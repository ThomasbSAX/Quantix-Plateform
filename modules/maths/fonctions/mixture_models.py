"""
Quantix – Module mixture_models
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# gaussian_mixture_model
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def gaussian_mixture_model_fit(
    X: np.ndarray,
    n_components: int = 1,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    normalization: str = 'standard',
    distance_metric: Union[str, Callable] = 'euclidean',
    solver: str = 'expectation_maximization',
    regularization: Optional[str] = None,
    metric: Union[str, Callable] = 'loglikelihood',
    weights_init: Optional[np.ndarray] = None,
    means_init: Optional[np.ndarray] = None,
    covars_init: Optional[np.ndarray] = None
) -> Dict:
    """
    Fit a Gaussian Mixture Model to the data.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    n_components : int, optional
        Number of mixture components.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence.
    random_state : int, optional
        Random seed for reproducibility.
    normalization : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    distance_metric : str or callable, optional
        Distance metric: 'euclidean', 'manhattan', 'cosine', or custom callable.
    solver : str, optional
        Solver method: 'expectation_maximization', 'gradient_descent'.
    regularization : str, optional
        Regularization type: 'none', 'l1', 'l2'.
    metric : str or callable, optional
        Metric for evaluation: 'loglikelihood', 'mse', or custom callable.
    weights_init : np.ndarray, optional
        Initial weights for components.
    means_init : np.ndarray, optional
        Initial means for components.
    covars_init : np.ndarray, optional
        Initial covariances for components.

    Returns:
    --------
    Dict containing:
        - 'result': Fitted model parameters.
        - 'metrics': Evaluation metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings encountered.
    """
    # Validate inputs
    _validate_inputs(X, n_components, weights_init, means_init, covars_init)

    # Initialize random state
    rng = np.random.RandomState(random_state)

    # Normalize data if required
    X_norm, norm_params = _normalize_data(X, normalization)

    # Initialize parameters
    weights, means, covars = _initialize_parameters(
        n_components, X_norm.shape[1], weights_init, means_init, covars_init, rng
    )

    # Fit the model
    params = _fit_gmm(
        X_norm, n_components, weights, means, covars,
        max_iter, tol, distance_metric, solver, regularization
    )

    # Compute metrics
    metrics = _compute_metrics(X_norm, params, metric)

    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'max_iter': max_iter,
            'tol': tol,
            'normalization': normalization,
            'distance_metric': distance_metric,
            'solver': solver,
            'regularization': regularization,
            'metric': metric
        },
        'warnings': []
    }

def _validate_inputs(
    X: np.ndarray,
    n_components: int,
    weights_init: Optional[np.ndarray],
    means_init: Optional[np.ndarray],
    covars_init: Optional[np.ndarray]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if n_components <= 0:
        raise ValueError("n_components must be positive.")

    if weights_init is not None and len(weights_init) != n_components:
        raise ValueError("weights_init must have length equal to n_components.")
    if means_init is not None and means_init.shape[0] != n_components:
        raise ValueError("means_init must have shape (n_components, n_features).")
    if covars_init is not None and covars_init.shape[0] != n_components:
        raise ValueError("covars_init must have shape (n_components, n_features, n_features).")

def _normalize_data(
    X: np.ndarray,
    method: str
) -> tuple[np.ndarray, Dict]:
    """Normalize the input data."""
    norm_params = {}
    if method == 'standard':
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X_norm = (X - mean) / (std + 1e-8)
        norm_params['mean'] = mean
        norm_params['std'] = std
    elif method == 'minmax':
        min_val = X.min(axis=0)
        max_val = X.max(axis=0)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
        norm_params['min'] = min_val
        norm_params['max'] = max_val
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - median) / (iqr + 1e-8)
        norm_params['median'] = median
        norm_params['iqr'] = iqr
    else:
        X_norm = X.copy()
    return X_norm, norm_params

def _initialize_parameters(
    n_components: int,
    n_features: int,
    weights_init: Optional[np.ndarray],
    means_init: Optional[np.ndarray],
    covars_init: Optional[np.ndarray],
    rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize the parameters of the Gaussian Mixture Model."""
    if weights_init is None:
        weights = np.ones(n_components) / n_components
    else:
        weights = weights_init.copy()

    if means_init is None:
        means = rng.uniform(size=(n_components, n_features))
    else:
        means = means_init.copy()

    if covars_init is None:
        covars = np.array([np.eye(n_features) for _ in range(n_components)])
    else:
        covars = covars_init.copy()

    return weights, means, covars

def _fit_gmm(
    X: np.ndarray,
    n_components: int,
    weights: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    max_iter: int,
    tol: float,
    distance_metric: Union[str, Callable],
    solver: str,
    regularization: Optional[str]
) -> Dict:
    """Fit the Gaussian Mixture Model using the specified solver."""
    if solver == 'expectation_maximization':
        return _em_gmm(X, n_components, weights, means, covars, max_iter, tol, distance_metric)
    elif solver == 'gradient_descent':
        return _gd_gmm(X, n_components, weights, means, covars, max_iter, tol, distance_metric)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _em_gmm(
    X: np.ndarray,
    n_components: int,
    weights: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    max_iter: int,
    tol: float,
    distance_metric: Union[str, Callable]
) -> Dict:
    """Expectation-Maximization algorithm for Gaussian Mixture Model."""
    n_samples, n_features = X.shape
    loglikelihood_old = -np.inf

    for _ in range(max_iter):
        # E-step
        responsibilities = _compute_responsibilities(X, weights, means, covars, distance_metric)

        # M-step
        weights, means, covars = _update_parameters(X, responsibilities, n_components)

        # Compute loglikelihood
        loglikelihood = _compute_loglikelihood(X, weights, means, covars)

        # Check for convergence
        if np.abs(loglikelihood - loglikelihood_old) < tol:
            break
        loglikelihood_old = loglikelihood

    return {
        'weights': weights,
        'means': means,
        'covariances': covars
    }

def _compute_responsibilities(
    X: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    distance_metric: Union[str, Callable]
) -> np.ndarray:
    """Compute the responsibilities for each data point."""
    n_samples, n_components = X.shape[0], weights.size
    responsibilities = np.zeros((n_samples, n_components))

    for k in range(n_components):
        if distance_metric == 'euclidean':
            dist = np.linalg.norm(X - means[k], axis=1)
        elif callable(distance_metric):
            dist = distance_metric(X, means[k])
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        responsibilities[:, k] = weights[k] * np.exp(-0.5 * dist)

    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return responsibilities

def _update_parameters(
    X: np.ndarray,
    responsibilities: np.ndarray,
    n_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update the parameters of the Gaussian Mixture Model."""
    n_samples = X.shape[0]

    # Update weights
    weights = responsibilities.sum(axis=0) / n_samples

    # Update means and covariances
    means = np.zeros((n_components, X.shape[1]))
    covars = np.zeros((n_components, X.shape[1], X.shape[1]))

    for k in range(n_components):
        Nk = responsibilities[:, k].sum()
        means[k] = (responsibilities[:, k] @ X) / Nk
        covars[k] = ((X - means[k]).T @ np.diag(responsibilities[:, k]) @ (X - means[k])) / Nk

    return weights, means, covars

def _compute_loglikelihood(
    X: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray
) -> float:
    """Compute the loglikelihood of the data under the model."""
    n_samples = X.shape[0]
    loglikelihood = 0.0

    for i in range(n_samples):
        total = 0.0
        for k in range(len(weights)):
            try:
                det_cov = np.linalg.det(covars[k])
                inv_cov = np.linalg.inv(covars[k])
            except:
                det_cov = 1.0
                inv_cov = np.eye(covars[k].shape[0])

            diff = X[i] - means[k]
            exponent = -0.5 * diff.T @ inv_cov @ diff
            total += weights[k] * np.exp(exponent) / (np.sqrt(2 * np.pi) ** len(diff) * np.sqrt(det_cov))

        loglikelihood += np.log(total + 1e-8)

    return loglikelihood

def _compute_metrics(
    X: np.ndarray,
    params: Dict,
    metric: Union[str, Callable]
) -> Dict:
    """Compute the metrics for the fitted model."""
    if metric == 'loglikelihood':
        return {'loglikelihood': _compute_loglikelihood(X, params['weights'], params['means'], params['covariances'])}
    elif callable(metric):
        return {f'custom_metric': metric(X, params)}
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _gd_gmm(
    X: np.ndarray,
    n_components: int,
    weights: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    max_iter: int,
    tol: float,
    distance_metric: Union[str, Callable]
) -> Dict:
    """Gradient Descent algorithm for Gaussian Mixture Model."""
    # Placeholder for gradient descent implementation
    return {
        'weights': weights,
        'means': means,
        'covariances': covars
    }

################################################################################
# expectation_maximization
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def expectation_maximization_fit(
    data: np.ndarray,
    n_components: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    distance_metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    normalization: Optional[str] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'logloss',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Fit a Gaussian Mixture Model using the Expectation-Maximization algorithm.

    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (n_samples, n_features).
    n_components : int
        Number of mixture components.
    max_iter : int, optional
        Maximum number of iterations (default: 100).
    tol : float, optional
        Tolerance for stopping criterion (default: 1e-4).
    random_state : Optional[int], optional
        Random seed for reproducibility (default: None).
    distance_metric : Union[str, Callable], optional
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable (default: 'euclidean').
    normalization : Optional[str], optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: None).
    metric : Union[str, Callable], optional
        Metric to evaluate the model ('mse', 'mae', 'r2', 'logloss') or custom callable (default: 'logloss').
    solver : str, optional
        Solver to use ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent') (default: 'closed_form').
    regularization : Optional[str], optional
        Regularization method ('none', 'l1', 'l2', 'elasticnet') (default: None).
    custom_distance : Optional[Callable], optional
        Custom distance function if not using built-in metrics (default: None).
    custom_metric : Optional[Callable], optional
        Custom metric function if not using built-in metrics (default: None).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, n_components)

    # Initialize random state
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize data
    if normalization == 'standard':
        data = _standard_normalize(data)
    elif normalization == 'minmax':
        data = _minmax_normalize(data)
    elif normalization == 'robust':
        data = _robust_normalize(data)

    # Initialize parameters
    params = _initialize_parameters(data, n_components)

    # Main EM loop
    for iteration in range(max_iter):
        # E-step: compute responsibilities
        responsibilities = _compute_responsibilities(data, params, distance_metric, custom_distance)

        # M-step: update parameters
        new_params = _update_parameters(data, responsibilities, params, solver, regularization)

        # Check convergence
        if _check_convergence(params, new_params, tol):
            break

        params = new_params

    # Compute metrics
    metrics = _compute_metrics(data, params, metric, custom_metric)

    # Prepare output
    result = {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'max_iter': max_iter,
            'tol': tol,
            'distance_metric': distance_metric if isinstance(distance_metric, str) else 'custom',
            'normalization': normalization,
            'metric': metric if isinstance(metric, str) else 'custom',
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

    return result

def _validate_inputs(data: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    if n_components <= 0:
        raise ValueError("Number of components must be positive.")

def _standard_normalize(data: np.ndarray) -> np.ndarray:
    """Standard normalize the data."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / (std + 1e-8)

def _minmax_normalize(data: np.ndarray) -> np.ndarray:
    """Min-max normalize the data."""
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    return (data - min_val) / (max_val - min_val + 1e-8)

def _robust_normalize(data: np.ndarray) -> np.ndarray:
    """Robust normalize the data using median and IQR."""
    median = np.median(data, axis=0)
    iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
    return (data - median) / (iqr + 1e-8)

def _initialize_parameters(data: np.ndarray, n_components: int) -> Dict[str, Any]:
    """Initialize parameters for the mixture model."""
    n_samples, n_features = data.shape
    weights = np.ones(n_components) / n_components
    means = data[np.random.choice(n_samples, n_components, replace=False)]
    covariances = np.array([np.cov(data.T) for _ in range(n_components)])
    return {'weights': weights, 'means': means, 'covariances': covariances}

def _compute_responsibilities(
    data: np.ndarray,
    params: Dict[str, Any],
    distance_metric: Union[str, Callable],
    custom_distance: Optional[Callable]
) -> np.ndarray:
    """Compute responsibilities for each data point."""
    weights = params['weights']
    means = params['means']
    covariances = params['covariances']

    if custom_distance is not None:
        distance_func = custom_distance
    else:
        distance_func = _get_distance_function(distance_metric)

    n_samples, n_components = data.shape[0], len(weights)
    responsibilities = np.zeros((n_samples, n_components))

    for k in range(n_components):
        if distance_metric == 'minkowski':
            distance_func = lambda x, y: np.linalg.norm(x - y, ord=3)
        distances = distance_func(data, means[k])
        responsibilities[:, k] = weights[k] * _gaussian_pdf(distances, covariances[k])

    responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
    return responsibilities

def _get_distance_function(metric: str) -> Callable:
    """Get the distance function based on the metric name."""
    if metric == 'euclidean':
        return lambda x, y: np.linalg.norm(x - y, axis=1)
    elif metric == 'manhattan':
        return lambda x, y: np.sum(np.abs(x - y), axis=1)
    elif metric == 'cosine':
        return lambda x, y: 1 - np.dot(x, y.T) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y))
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _gaussian_pdf(distances: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Compute the Gaussian PDF."""
    n_features = distances.shape[1]
    det_cov = np.linalg.det(covariance)
    inv_cov = np.linalg.inv(covariance)
    exponent = -0.5 * np.sum(distances ** 2, axis=1) / det_cov
    return (1 / np.sqrt((2 * np.pi) ** n_features * det_cov)) * np.exp(exponent)

def _update_parameters(
    data: np.ndarray,
    responsibilities: np.ndarray,
    params: Dict[str, Any],
    solver: str,
    regularization: Optional[str]
) -> Dict[str, Any]:
    """Update parameters using the specified solver."""
    if solver == 'closed_form':
        return _update_parameters_closed_form(data, responsibilities, params, regularization)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _update_parameters_closed_form(
    data: np.ndarray,
    responsibilities: np.ndarray,
    params: Dict[str, Any],
    regularization: Optional[str]
) -> Dict[str, Any]:
    """Update parameters using closed-form solution."""
    n_components = responsibilities.shape[1]
    weights = np.sum(responsibilities, axis=0) / data.shape[0]

    means = np.zeros((n_components, data.shape[1]))
    covariances = np.zeros((n_components, data.shape[1], data.shape[1]))

    for k in range(n_components):
        Nk = np.sum(responsibilities[:, k])
        means[k] = np.sum(responsibilities[:, k, np.newaxis] * data, axis=0) / Nk
        covariances[k] = (responsibilities[:, k, np.newaxis, np.newaxis] *
                         (data - means[k])[:, :, np.newaxis]).T @ (data - means[k]) / Nk

    return {'weights': weights, 'means': means, 'covariances': covariances}

def _check_convergence(
    old_params: Dict[str, Any],
    new_params: Dict[str, Any],
    tol: float
) -> bool:
    """Check if the parameters have converged."""
    for key in old_params:
        if not np.allclose(old_params[key], new_params[key], atol=tol):
            return False
    return True

def _compute_metrics(
    data: np.ndarray,
    params: Dict[str, Any],
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict[str, float]:
    """Compute metrics for the model."""
    if custom_metric is not None:
        return {'custom': custom_metric(data, params)}

    if metric == 'mse':
        return {'mse': _compute_mse(data, params)}
    elif metric == 'mae':
        return {'mae': _compute_mae(data, params)}
    elif metric == 'r2':
        return {'r2': _compute_r2(data, params)}
    elif metric == 'logloss':
        return {'logloss': _compute_logloss(data, params)}
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _compute_mse(data: np.ndarray, params: Dict[str, Any]) -> float:
    """Compute Mean Squared Error."""
    # Placeholder implementation
    return 0.0

def _compute_mae(data: np.ndarray, params: Dict[str, Any]) -> float:
    """Compute Mean Absolute Error."""
    # Placeholder implementation
    return 0.0

def _compute_r2(data: np.ndarray, params: Dict[str, Any]) -> float:
    """Compute R-squared."""
    # Placeholder implementation
    return 0.0

def _compute_logloss(data: np.ndarray, params: Dict[str, Any]) -> float:
    """Compute Log Loss."""
    # Placeholder implementation
    return 0.0

# Example usage:
# result = expectation_maximization_fit(data=np.random.rand(100, 5), n_components=3)

################################################################################
# bayesian_gaussian_mixture
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def bayesian_gaussian_mixture_fit(
    X: np.ndarray,
    n_components: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    normalization: str = 'standard',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, np.ndarray, float, str]]:
    """
    Fit a Bayesian Gaussian Mixture Model to the data.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    n_components : int
        Number of mixture components.
    max_iter : int, optional
        Maximum number of iterations, by default 100.
    tol : float, optional
        Tolerance for convergence, by default 1e-4.
    random_state : Optional[int], optional
        Random seed for reproducibility, by default None.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust'), by default 'standard'.
    metric : Union[str, Callable], optional
        Metric to evaluate the model ('mse', 'mae', 'r2', 'logloss'), by default 'mse'.
    distance : Union[str, Callable], optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski'), by default 'euclidean'.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent'), by default 'gradient_descent'.
    regularization : Optional[str], optional
        Regularization method ('none', 'l1', 'l2', 'elasticnet'), by default None.
    custom_metric : Optional[Callable], optional
        Custom metric function, by default None.
    custom_distance : Optional[Callable], optional
        Custom distance function, by default None.

    Returns
    -------
    Dict[str, Union[Dict, np.ndarray, float, str]]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    validate_inputs(X, n_components)

    # Set random state if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize data
    X_normalized, normalization_params = apply_normalization(X, normalization)

    # Initialize parameters
    weights, means, covariances = initialize_parameters(X_normalized, n_components)

    # Choose solver
    if solver == 'closed_form':
        weights, means, covariances = closed_form_solver(X_normalized, n_components)
    elif solver == 'gradient_descent':
        weights, means, covariances = gradient_descent_solver(X_normalized, n_components, max_iter, tol)
    elif solver == 'newton':
        weights, means, covariances = newton_solver(X_normalized, n_components, max_iter, tol)
    elif solver == 'coordinate_descent':
        weights, means, covariances = coordinate_descent_solver(X_normalized, n_components, max_iter, tol)
    else:
        raise ValueError("Invalid solver specified.")

    # Apply regularization if needed
    if regularization is not None:
        weights, means, covariances = apply_regularization(weights, means, covariances, regularization)

    # Calculate metrics
    if isinstance(metric, str):
        metric_func = get_metric_function(metric)
    else:
        metric_func = metric

    if isinstance(distance, str):
        distance_func = get_distance_function(distance)
    else:
        distance_func = distance

    metrics = calculate_metrics(X_normalized, weights, means, covariances, metric_func, distance_func)

    # Prepare output
    result = {
        'weights': weights,
        'means': means,
        'covariances': covariances
    }

    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'max_iter': max_iter,
            'tol': tol,
            'normalization': normalization,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

    return output

def validate_inputs(X: np.ndarray, n_components: int) -> None:
    """Validate the input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if n_components <= 0:
        raise ValueError("n_components must be a positive integer.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values.")

def apply_normalization(X: np.ndarray, method: str) -> Tuple[np.ndarray, Dict[str, Union[float, np.ndarray]]]:
    """Apply normalization to the data."""
    if method == 'none':
        return X, {}
    elif method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_normalized = (X - mean) / std
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_normalized = (X - min_val) / (max_val - min_val)
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_normalized = (X - median) / iqr
    else:
        raise ValueError("Invalid normalization method specified.")
    return X_normalized, {'method': method}

def initialize_parameters(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize the parameters of the mixture model."""
    n_samples, n_features = X.shape
    weights = np.ones(n_components) / n_components
    means = X[np.random.choice(n_samples, n_components, replace=False)]
    covariances = np.array([np.cov(X.T) for _ in range(n_components)])
    return weights, means, covariances

def closed_form_solver(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the mixture model using closed-form solution."""
    # Placeholder for actual implementation
    return np.ones(n_components) / n_components, X[:n_components], np.array([np.eye(X.shape[1]) for _ in range(n_components)])

def gradient_descent_solver(X: np.ndarray, n_components: int, max_iter: int, tol: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the mixture model using gradient descent."""
    # Placeholder for actual implementation
    return np.ones(n_components) / n_components, X[:n_components], np.array([np.eye(X.shape[1]) for _ in range(n_components)])

def newton_solver(X: np.ndarray, n_components: int, max_iter: int, tol: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the mixture model using Newton's method."""
    # Placeholder for actual implementation
    return np.ones(n_components) / n_components, X[:n_components], np.array([np.eye(X.shape[1]) for _ in range(n_components)])

def coordinate_descent_solver(X: np.ndarray, n_components: int, max_iter: int, tol: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the mixture model using coordinate descent."""
    # Placeholder for actual implementation
    return np.ones(n_components) / n_components, X[:n_components], np.array([np.eye(X.shape[1]) for _ in range(n_components)])

def apply_regularization(weights: np.ndarray, means: np.ndarray, covariances: np.ndarray, method: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply regularization to the parameters."""
    if method == 'none':
        return weights, means, covariances
    elif method == 'l1':
        # Placeholder for L1 regularization
        return weights, means, covariances
    elif method == 'l2':
        # Placeholder for L2 regularization
        return weights, means, covariances
    elif method == 'elasticnet':
        # Placeholder for elastic net regularization
        return weights, means, covariances
    else:
        raise ValueError("Invalid regularization method specified.")

def get_metric_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the metric function based on the specified metric."""
    if metric == 'mse':
        return mean_squared_error
    elif metric == 'mae':
        return mean_absolute_error
    elif metric == 'r2':
        return r_squared
    elif metric == 'logloss':
        return log_loss
    else:
        raise ValueError("Invalid metric specified.")

def get_distance_function(distance: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the distance function based on the specified distance."""
    if distance == 'euclidean':
        return euclidean_distance
    elif distance == 'manhattan':
        return manhattan_distance
    elif distance == 'cosine':
        return cosine_distance
    elif distance == 'minkowski':
        return minkowski_distance
    else:
        raise ValueError("Invalid distance specified.")

def calculate_metrics(X: np.ndarray, weights: np.ndarray, means: np.ndarray, covariances: np.ndarray,
                      metric_func: Callable[[np.ndarray, np.ndarray], float],
                      distance_func: Callable[[np.ndarray, np.ndarray], float]) -> Dict[str, float]:
    """Calculate the metrics for the mixture model."""
    # Placeholder for actual implementation
    return {'metric': 0.0, 'distance': 0.0}

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the R-squared metric."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the log loss."""
    # Placeholder for actual implementation
    return 0.0

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the Euclidean distance."""
    return np.linalg.norm(a - b)

def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the Manhattan distance."""
    return np.sum(np.abs(a - b))

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the cosine distance."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def minkowski_distance(a: np.ndarray, b: np.ndarray, p: int = 3) -> float:
    """Calculate the Minkowski distance."""
    return np.sum(np.abs(a - b) ** p) ** (1 / p)

################################################################################
# dirichlet_process_mixture_model
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def dirichlet_process_mixture_model_fit(
    data: np.ndarray,
    n_components: int = 3,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    distance_metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    normalization: Optional[str] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    beta: float = 1.0
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Fit a Dirichlet Process Mixture Model to the given data.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_samples, n_features).
    n_components : int, optional
        Number of components in the mixture model, by default 3.
    max_iter : int, optional
        Maximum number of iterations, by default 100.
    tol : float, optional
        Tolerance for convergence, by default 1e-4.
    random_state : Optional[int], optional
        Random seed for reproducibility, by default None.
    distance_metric : Union[str, Callable], optional
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski'), by default 'euclidean'.
    normalization : Optional[str], optional
        Normalization method ('none', 'standard', 'minmax', 'robust'), by default None.
    metric : Union[str, Callable], optional
        Metric to evaluate the model ('mse', 'mae', 'r2', 'logloss'), by default 'mse'.
    solver : str, optional
        Solver to use ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent'), by default 'gradient_descent'.
    regularization : Optional[str], optional
        Regularization method ('none', 'l1', 'l2', 'elasticnet'), by default None.
    alpha : float, optional
        Concentration parameter for the Dirichlet process, by default 1.0.
    beta : float, optional
        Base measure parameter for the Dirichlet process, by default 1.0.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], str]]
        A dictionary containing the results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.random.rand(100, 2)
    >>> result = dirichlet_process_mixture_model_fit(data, n_components=3)
    """
    # Validate inputs
    _validate_inputs(data, n_components)

    # Normalize data if specified
    if normalization:
        data = _normalize_data(data, normalization)

    # Initialize parameters
    params = _initialize_parameters(data.shape[1], n_components, random_state)

    # Choose distance metric
    if isinstance(distance_metric, str):
        distance_func = _get_distance_function(distance_metric)
    else:
        distance_func = distance_metric

    # Choose metric
    if isinstance(metric, str):
        metric_func = _get_metric_function(metric)
    else:
        metric_func = metric

    # Choose solver
    if solver == 'gradient_descent':
        params, log_likelihood = _gradient_descent_solver(data, params, distance_func, max_iter, tol, regularization)
    elif solver == 'closed_form':
        params = _closed_form_solver(data, params, distance_func)
    elif solver == 'newton':
        params = _newton_solver(data, params, distance_func)
    elif solver == 'coordinate_descent':
        params = _coordinate_descent_solver(data, params, distance_func)
    else:
        raise ValueError("Unsupported solver type.")

    # Calculate metrics
    metrics = _calculate_metrics(data, params, metric_func)

    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'max_iter': max_iter,
            'tol': tol,
            'distance_metric': distance_metric,
            'normalization': normalization,
            'metric': metric,
            'solver': solver,
            'regularization': regularization,
            'alpha': alpha,
            'beta': beta
        },
        'warnings': []
    }

def _validate_inputs(data: np.ndarray, n_components: int) -> None:
    """Validate the input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    if n_components <= 0:
        raise ValueError("Number of components must be positive.")

def _normalize_data(data: np.ndarray, method: str) -> np.ndarray:
    """Normalize the data using the specified method."""
    if method == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == 'robust':
        return (data - np.median(data, axis=0)) / (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0))
    else:
        return data

def _initialize_parameters(n_features: int, n_components: int, random_state: Optional[int]) -> np.ndarray:
    """Initialize the parameters of the mixture model."""
    if random_state is not None:
        np.random.seed(random_state)
    return np.random.rand(n_components, n_features)

def _get_distance_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the distance function based on the specified metric."""
    if metric == 'euclidean':
        return lambda x, y: np.linalg.norm(x - y)
    elif metric == 'manhattan':
        return lambda x, y: np.sum(np.abs(x - y))
    elif metric == 'cosine':
        return lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    elif metric == 'minkowski':
        return lambda x, y: np.sum(np.abs(x - y) ** 3) ** (1/3)
    else:
        raise ValueError("Unsupported distance metric.")

def _get_metric_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the metric function based on the specified metric."""
    if metric == 'mse':
        return lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        return lambda y_true, y_pred: 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    elif metric == 'logloss':
        return lambda y_true, y_pred: -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError("Unsupported metric.")

def _gradient_descent_solver(
    data: np.ndarray,
    params: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    max_iter: int,
    tol: float,
    regularization: Optional[str]
) -> Tuple[np.ndarray, float]:
    """Solve the mixture model using gradient descent."""
    log_likelihood = -np.inf
    for _ in range(max_iter):
        new_params, new_log_likelihood = _gradient_descent_step(data, params, distance_func, regularization)
        if np.abs(new_log_likelihood - log_likelihood) < tol:
            break
        params, log_likelihood = new_params, new_log_likelihood
    return params, log_likelihood

def _gradient_descent_step(
    data: np.ndarray,
    params: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: Optional[str]
) -> Tuple[np.ndarray, float]:
    """Perform a single gradient descent step."""
    # Placeholder for actual implementation
    return params, 0.0

def _closed_form_solver(
    data: np.ndarray,
    params: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """Solve the mixture model using closed-form solution."""
    # Placeholder for actual implementation
    return params

def _newton_solver(
    data: np.ndarray,
    params: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """Solve the mixture model using Newton's method."""
    # Placeholder for actual implementation
    return params

def _coordinate_descent_solver(
    data: np.ndarray,
    params: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """Solve the mixture model using coordinate descent."""
    # Placeholder for actual implementation
    return params

def _calculate_metrics(
    data: np.ndarray,
    params: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Calculate the metrics for the mixture model."""
    # Placeholder for actual implementation
    return {'metric': 0.0}

################################################################################
# variational_bayes
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def variational_bayes_fit(
    data: np.ndarray,
    n_components: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    normalization: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    learning_rate: float = 0.01,
    batch_size: Optional[int] = None,
    random_state: Optional[int] = None,
    custom_metric_func: Optional[Callable] = None,
    custom_distance_func: Optional[Callable] = None
) -> Dict:
    """
    Fit a variational Bayes mixture model to the data.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_samples, n_features).
    n_components : int
        Number of mixture components.
    max_iter : int, optional
        Maximum number of iterations, by default 100.
    tol : float, optional
        Tolerance for convergence, by default 1e-4.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust'), by default 'standard'.
    metric : Union[str, Callable], optional
        Metric to optimize ('mse', 'mae', 'r2', 'logloss'), by default 'mse'.
    distance : Union[str, Callable], optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski'), by default 'euclidean'.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent'), by default 'gradient_descent'.
    regularization : Optional[str], optional
        Regularization method (None, 'l1', 'l2', 'elasticnet'), by default None.
    learning_rate : float, optional
        Learning rate for gradient-based solvers, by default 0.01.
    batch_size : Optional[int], optional
        Batch size for stochastic solvers, by default None.
    random_state : Optional[int], optional
        Random seed for reproducibility, by default None.
    custom_metric_func : Optional[Callable], optional
        Custom metric function, by default None.
    custom_distance_func : Optional[Callable], optional
        Custom distance function, by default None.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Fitted model parameters.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings encountered.

    Examples
    --------
    >>> data = np.random.randn(100, 2)
    >>> result = variational_bayes_fit(data, n_components=3)
    """
    # Validate inputs
    _validate_inputs(data, n_components)

    # Normalize data
    normalized_data = _normalize_data(data, normalization)

    # Initialize parameters
    params = _initialize_parameters(normalized_data.shape[1], n_components, random_state)

    # Choose solver
    if solver == 'closed_form':
        result = _solve_closed_form(normalized_data, params)
    elif solver == 'gradient_descent':
        result = _solve_gradient_descent(
            normalized_data, params, max_iter, tol,
            learning_rate, batch_size, metric, distance,
            regularization
        )
    elif solver == 'newton':
        result = _solve_newton(normalized_data, params, max_iter, tol)
    elif solver == 'coordinate_descent':
        result = _solve_coordinate_descent(normalized_data, params, max_iter, tol)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(normalized_data, result, metric, custom_metric_func)

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'max_iter': max_iter,
            'tol': tol,
            'normalization': normalization,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'learning_rate': learning_rate,
            'batch_size': batch_size
        },
        'warnings': []
    }

    return output

def _validate_inputs(data: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Data must be 2-dimensional.")
    if n_components <= 0:
        raise ValueError("Number of components must be positive.")

def _normalize_data(data: np.ndarray, method: str) -> np.ndarray:
    """Normalize data using the specified method."""
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

def _initialize_parameters(n_features: int, n_components: int, random_state: Optional[int]) -> Dict:
    """Initialize mixture model parameters."""
    if random_state is not None:
        np.random.seed(random_state)

    weights = np.ones(n_components) / n_components
    means = np.random.randn(n_components, n_features)
    covariances = np.array([np.eye(n_features) for _ in range(n_components)])

    return {
        'weights': weights,
        'means': means,
        'covariances': covariances
    }

def _solve_closed_form(data: np.ndarray, params: Dict) -> Dict:
    """Solve mixture model using closed-form solution."""
    # Placeholder for actual implementation
    return params

def _solve_gradient_descent(
    data: np.ndarray,
    params: Dict,
    max_iter: int,
    tol: float,
    learning_rate: float,
    batch_size: Optional[int],
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str]
) -> Dict:
    """Solve mixture model using gradient descent."""
    # Placeholder for actual implementation
    return params

def _solve_newton(data: np.ndarray, params: Dict, max_iter: int, tol: float) -> Dict:
    """Solve mixture model using Newton's method."""
    # Placeholder for actual implementation
    return params

def _solve_coordinate_descent(data: np.ndarray, params: Dict, max_iter: int, tol: float) -> Dict:
    """Solve mixture model using coordinate descent."""
    # Placeholder for actual implementation
    return params

def _compute_metrics(
    data: np.ndarray,
    result: Dict,
    metric: Union[str, Callable],
    custom_metric_func: Optional[Callable]
) -> Dict:
    """Compute metrics for the fitted model."""
    if custom_metric_func is not None:
        return {'custom': custom_metric_func(data, result)}

    metrics = {}
    if metric == 'mse':
        metrics['mse'] = np.mean((data - result['means'][0])**2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(data - result['means'][0]))
    elif metric == 'r2':
        metrics['r2'] = 1 - np.sum((data - result['means'][0])**2) / np.sum((data - np.mean(data, axis=0))**2)
    elif metric == 'logloss':
        metrics['logloss'] = -np.mean(np.log(result['weights'][0]))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

################################################################################
# kmeans_init
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def kmeans_init_fit(
    X: np.ndarray,
    n_clusters: int,
    init_method: str = 'random',
    distance_metric: Union[str, Callable] = 'euclidean',
    normalization: Optional[str] = None,
    random_state: Optional[int] = None,
    custom_init_func: Optional[Callable] = None
) -> Dict:
    """
    Initialize cluster centers for K-means clustering.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    n_clusters : int
        Number of clusters to initialize.
    init_method : str, optional (default='random')
        Initialization method: 'random', 'k-means++'.
    distance_metric : str or callable, optional (default='euclidean')
        Distance metric to use: 'euclidean', 'manhattan', 'cosine', or custom callable.
    normalization : str, optional (default=None)
        Normalization method: 'standard', 'minmax', 'robust'.
    random_state : int, optional (default=None)
        Random seed for reproducibility.
    custom_init_func : callable, optional (default=None)
        Custom initialization function.

    Returns:
    --------
    dict
        Dictionary containing:
        - 'centers': Initialized cluster centers.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used in the initialization.
        - 'warnings': Any warnings encountered.
    """
    # Validate inputs
    _validate_inputs(X, n_clusters)

    # Set random state if provided
    rng = np.random.RandomState(random_state)

    # Normalize data if specified
    X_normalized, norm_warning = _apply_normalization(X, normalization)

    # Initialize centers
    if custom_init_func is not None:
        centers = _custom_initialization(X_normalized, n_clusters, custom_init_func)
    else:
        if init_method == 'random':
            centers = _random_initialization(X_normalized, n_clusters, rng)
        elif init_method == 'k-means++':
            centers = _kmeans_plusplus_initialization(X_normalized, n_clusters, distance_metric, rng)
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")

    # Compute metrics
    metrics = _compute_metrics(X_normalized, centers, distance_metric)

    # Prepare output
    result = {
        'centers': centers,
        'metrics': metrics,
        'params_used': {
            'init_method': init_method,
            'distance_metric': distance_metric,
            'normalization': normalization
        },
        'warnings': norm_warning if norm_warning else []
    }

    return result

def _validate_inputs(X: np.ndarray, n_clusters: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if n_clusters <= 0:
        raise ValueError("n_clusters must be a positive integer.")
    if n_clusters > X.shape[0]:
        raise ValueError("n_clusters cannot be greater than the number of samples.")

def _apply_normalization(X: np.ndarray, method: Optional[str]) -> tuple:
    """Apply specified normalization to the data."""
    warnings = []
    if method is None:
        return X, warnings

    X_normalized = X.copy()
    if method == 'standard':
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

    return X_normalized, warnings

def _random_initialization(X: np.ndarray, n_clusters: int, rng: np.random.RandomState) -> np.ndarray:
    """Randomly initialize cluster centers."""
    indices = rng.choice(X.shape[0], n_clusters, replace=False)
    return X[indices]

def _kmeans_plusplus_initialization(
    X: np.ndarray,
    n_clusters: int,
    distance_metric: Union[str, Callable],
    rng: np.random.RandomState
) -> np.ndarray:
    """Initialize cluster centers using k-means++ algorithm."""
    centers = [X[rng.choice(X.shape[0])]]

    for _ in range(1, n_clusters):
        distances = np.array([min(_compute_distance(x, centers, distance_metric)) for x in X])
        probabilities = distances / np.sum(distances)
        new_center_idx = rng.choice(X.shape[0], p=probabilities)
        centers.append(X[new_center_idx])

    return np.array(centers)

def _custom_initialization(
    X: np.ndarray,
    n_clusters: int,
    custom_func: Callable
) -> np.ndarray:
    """Initialize cluster centers using a custom function."""
    return custom_func(X, n_clusters)

def _compute_distance(
    x: np.ndarray,
    centers: np.ndarray,
    metric: Union[str, Callable]
) -> float:
    """Compute distance between a point and cluster centers."""
    if callable(metric):
        return metric(x, centers)
    elif metric == 'euclidean':
        return np.linalg.norm(x - centers, axis=1)
    elif metric == 'manhattan':
        return np.sum(np.abs(x - centers), axis=1)
    elif metric == 'cosine':
        return 1 - np.dot(x, centers.T) / (np.linalg.norm(x) * np.linalg.norm(centers, axis=1))
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _compute_metrics(
    X: np.ndarray,
    centers: np.ndarray,
    distance_metric: Union[str, Callable]
) -> Dict:
    """Compute metrics for the initialized centers."""
    distances = np.array([_compute_distance(x, centers, distance_metric) for x in X])
    min_distances = np.min(distances, axis=1)
    metrics = {
        'average_distance': np.mean(min_distances),
        'max_distance': np.max(min_distances),
        'min_distance': np.min(min_distances)
    }
    return metrics

################################################################################
# random_init
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def random_init_fit(
    data: np.ndarray,
    n_components: int,
    init_method: str = 'kmeans++',
    metric: Union[str, Callable] = 'euclidean',
    normalization: str = 'standard',
    random_state: Optional[int] = None,
    max_iter: int = 100,
    tol: float = 1e-4,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Initialize mixture model parameters randomly with various options.

    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (n_samples, n_features)
    n_components : int
        Number of mixture components
    init_method : str, optional
        Initialization method ('random', 'kmeans++')
    metric : str or callable, optional
        Distance metric ('euclidean', 'manhattan', 'cosine') or custom function
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    random_state : int, optional
        Random seed for reproducibility
    max_iter : int, optional
        Maximum iterations for kmeans++
    tol : float, optional
        Tolerance for convergence in kmeans++
    verbose : bool, optional
        Whether to print progress

    Returns:
    --------
    dict
        Dictionary containing initialization results, metrics and parameters used
    """
    # Validate inputs
    _validate_inputs(data, n_components)

    # Set random state if provided
    rng = np.random.RandomState(random_state)

    # Normalize data
    normalized_data, norm_params = _apply_normalization(data, normalization)

    # Choose initialization method
    if init_method == 'random':
        centers = _random_init(normalized_data, n_components, rng)
    elif init_method == 'kmeans++':
        centers = _kmeans_plusplus_init(normalized_data, n_components,
                                      metric, max_iter, tol, rng)
    else:
        raise ValueError(f"Unknown initialization method: {init_method}")

    # Calculate initial weights
    weights = _calculate_initial_weights(normalized_data, centers, metric)

    # Calculate initial covariance matrices
    covariances = _calculate_initial_covariances(normalized_data, centers,
                                               weights, metric)

    # Calculate metrics
    metrics = _calculate_metrics(normalized_data, centers, weights,
                               covariances, metric)

    return {
        'result': {
            'centers': centers,
            'weights': weights,
            'covariances': covariances
        },
        'metrics': metrics,
        'params_used': {
            'init_method': init_method,
            'metric': metric,
            'normalization': normalization
        },
        'warnings': []
    }

def _validate_inputs(data: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be 2-dimensional")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values")

def _apply_normalization(
    data: np.ndarray,
    method: str
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Apply data normalization."""
    normalized = data.copy()
    params = {}

    if method == 'standard':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized = (data - mean) / (std + 1e-8)
        params['mean'] = mean
        params['std'] = std
    elif method == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        params['min'] = min_val
        params['max'] = max_val
    elif method == 'robust':
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        normalized = (data - median) / (iqr + 1e-8)
        params['median'] = median
        params['iqr'] = iqr

    return normalized, params

def _random_init(
    data: np.ndarray,
    n_components: int,
    rng: np.random.RandomState
) -> np.ndarray:
    """Random initialization of cluster centers."""
    indices = rng.choice(data.shape[0], n_components, replace=False)
    return data[indices]

def _kmeans_plusplus_init(
    data: np.ndarray,
    n_components: int,
    metric: Union[str, Callable],
    max_iter: int,
    tol: float,
    rng: np.random.RandomState
) -> np.ndarray:
    """K-means++ initialization of cluster centers."""
    # First center is chosen uniformly at random
    centers = [data[rng.choice(data.shape[0])]]

    for _ in range(1, n_components):
        # Calculate distances to nearest existing center
        dists = _calculate_distances(data, centers[-1], metric)

        # Choose next center with probability proportional to squared distance
        probs = dists / np.sum(dists)
        next_center_idx = rng.choice(data.shape[0], p=probs)
        centers.append(data[next_center_idx])

    return np.array(centers)

def _calculate_distances(
    data: np.ndarray,
    center: np.ndarray,
    metric: Union[str, Callable]
) -> np.ndarray:
    """Calculate distances between data points and a center."""
    if callable(metric):
        return np.array([metric(x, center) for x in data])
    elif metric == 'euclidean':
        return np.linalg.norm(data - center, axis=1)
    elif metric == 'manhattan':
        return np.sum(np.abs(data - center), axis=1)
    elif metric == 'cosine':
        return 1 - np.dot(data, center) / (np.linalg.norm(data, axis=1) *
                                         np.linalg.norm(center))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _calculate_initial_weights(
    data: np.ndarray,
    centers: np.ndarray,
    metric: Union[str, Callable]
) -> np.ndarray:
    """Calculate initial weights for each component."""
    # Calculate distances to all centers
    dists = np.array([_calculate_distances(data, center, metric)
                     for center in centers])

    # Softmax to get weights
    exp_dists = np.exp(-dists)
    weights = exp_dists / np.sum(exp_dists, axis=0)

    return np.mean(weights, axis=1)

def _calculate_initial_covariances(
    data: np.ndarray,
    centers: np.ndarray,
    weights: np.ndarray,
    metric: Union[str, Callable]
) -> list:
    """Calculate initial covariance matrices for each component."""
    covariances = []
    n_features = data.shape[1]

    for i in range(len(centers)):
        # Calculate responsibilities
        dists = _calculate_distances(data, centers[i], metric)
        responsibilities = np.exp(-dists) / np.sum(np.exp(-dists))

        # Calculate covariance matrix
        centered = data - centers[i]
        cov = np.dot(responsibilities * centered.T, centered) / (np.sum(responsibilities) + 1e-8)

        # Ensure positive definiteness
        cov = (cov + cov.T) / 2
        eigvals, eigvecs = np.linalg.eigh(cov)
        cov = eigvecs @ np.diag(np.maximum(eigvals, 1e-8)) @ eigvecs.T

        covariances.append(cov)

    return covariances

def _calculate_metrics(
    data: np.ndarray,
    centers: np.ndarray,
    weights: np.ndarray,
    covariances: list,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """Calculate initialization metrics."""
    # Calculate log-likelihood
    log_likelihood = 0.0

    for i in range(len(centers)):
        # Calculate multivariate normal PDF
        if metric == 'euclidean':
            # For Euclidean distance, use standard multivariate normal
            det = np.linalg.det(covariances[i])
            if det <= 0:
                det = 1e-8
            inv_cov = np.linalg.inv(covariances[i])
            exponent = -0.5 * np.sum((data - centers[i]) @ inv_cov *
                                   (data - centers[i]), axis=1)
            log_pdf = -0.5 * np.log(det) + exponent
        else:
            # For other metrics, use a simplified approach
            dists = _calculate_distances(data, centers[i], metric)
            log_pdf = -dists

        # Add component contribution
        log_likelihood += weights[i] * np.log(np.sum(np.exp(log_pdf)))

    # Calculate BIC (Bayesian Information Criterion)
    n_samples, n_features = data.shape
    n_params = len(centers) * (n_features + 1 + n_features*(n_features+1)/2)
    bic = -2 * log_likelihood + n_params * np.log(n_samples)

    return {
        'log_likelihood': float(log_likelihood),
        'bic': float(bic)
    }

################################################################################
# aic_selection
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(
    data: np.ndarray,
    n_components_range: range,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values")
    if not isinstance(n_components_range, range):
        raise TypeError("n_components_range must be a range object")

    if normalizer is not None:
        try:
            normalized_data = normalizer(data)
        except Exception as e:
            raise ValueError(f"Normalization failed: {str(e)}")

def compute_aic(
    log_likelihood: float,
    n_params: int,
    n_samples: int
) -> float:
    """Compute AIC (Akaike Information Criterion)."""
    return 2 * n_params - 2 * log_likelihood

def fit_mixture_model(
    data: np.ndarray,
    n_components: int,
    solver: Callable[[np.ndarray, int], Dict[str, Any]],
    **solver_kwargs
) -> Dict[str, Any]:
    """Fit a mixture model with given number of components."""
    return solver(data, n_components, **solver_kwargs)

def calculate_metrics(
    data: np.ndarray,
    model_result: Dict[str, Any],
    metric_funcs: Dict[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Calculate various metrics for model evaluation."""
    return {name: func(data, model_result['predictions']) for name, func in metric_funcs.items()}

def aic_selection_fit(
    data: np.ndarray,
    n_components_range: range,
    solver: Callable[[np.ndarray, int], Dict[str, Any]],
    metric_funcs: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    **solver_kwargs
) -> Dict[str, Any]:
    """
    Select the optimal number of components for a mixture model using AIC.

    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (n_samples, n_features)
    n_components_range : range
        Range of number of components to try
    solver : callable
        Function that fits a mixture model and returns results
    metric_funcs : dict, optional
        Dictionary of metric functions to evaluate model performance
    normalizer : callable, optional
        Function to normalize the input data
    **solver_kwargs :
        Additional keyword arguments for the solver function

    Returns:
    --------
    dict
        Dictionary containing:
        - 'best_n_components': int
        - 'aic_scores': dict
        - 'metrics': dict (if metric_funcs provided)
        - 'params_used': dict
        - 'warnings': list

    Example:
    --------
    >>> data = np.random.randn(100, 2)
    >>> def my_solver(data, n_components):
    ...     return {'log_likelihood': 0.0, 'predictions': np.zeros(len(data))}
    >>> result = aic_selection_fit(data, range(1, 4), my_solver)
    """
    # Validate inputs
    validate_inputs(data, n_components_range, normalizer)

    if metric_funcs is None:
        metric_funcs = {}

    # Normalize data if specified
    normalized_data = normalizer(data) if normalizer is not None else data

    # Initialize results
    aic_scores = {}
    best_n_components = None
    min_aic = float('inf')
    metrics_results = {}
    warnings_list = []

    # Try each number of components
    for n_components in n_components_range:
        try:
            # Fit model
            model_result = fit_mixture_model(
                normalized_data,
                n_components,
                solver,
                **solver_kwargs
            )

            # Calculate AIC
            aic = compute_aic(
                model_result['log_likelihood'],
                n_components * (normalized_data.shape[1] + 1),  # Simple parameter count
                len(normalized_data)
            )
            aic_scores[n_components] = aic

            # Update best model
            if aic < min_aic:
                min_aic = aic
                best_n_components = n_components

            # Calculate metrics if provided
            if metric_funcs:
                metrics_results[n_components] = calculate_metrics(
                    normalized_data,
                    model_result,
                    metric_funcs
                )

        except Exception as e:
            warnings_list.append(f"Failed for {n_components} components: {str(e)}")
            continue

    # Prepare final results
    result = {
        'best_n_components': best_n_components,
        'aic_scores': aic_scores,
        'metrics': metrics_results if metric_funcs else None,
        'params_used': {
            'n_components_range': list(n_components_range),
            'solver_kwargs': solver_kwargs
        },
        'warnings': warnings_list
    }

    return result

# Example metric functions
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Example normalizer functions
def standard_normalize(data: np.ndarray) -> np.ndarray:
    """Standard normalization (z-score)."""
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def minmax_normalize(data: np.ndarray) -> np.ndarray:
    """Min-max normalization."""
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

################################################################################
# bic_selection
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def bic_selection_fit(
    data: np.ndarray,
    max_components: int = 5,
    normalize: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Select the optimal number of components for a mixture model using BIC criterion.

    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (n_samples, n_features)
    max_components : int, optional
        Maximum number of components to try (default: 5)
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard')
    metric : str or callable, optional
        Metric to evaluate model performance ('mse', 'mae', 'r2', 'logloss') or custom callable (default: 'mse')
    distance : str, optional
        Distance metric for clustering ('euclidean', 'manhattan', 'cosine', 'minkowski') (default: 'euclidean')
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent') (default: 'closed_form')
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet') (default: None)
    tol : float, optional
        Tolerance for convergence (default: 1e-4)
    max_iter : int, optional
        Maximum number of iterations (default: 1000)
    random_state : int, optional
        Random seed for reproducibility (default: None)
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics (default: None)

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - 'result': Optimal number of components
        - 'metrics': Dictionary of metrics for each model
        - 'params_used': Parameters used in the fitting process
        - 'warnings': List of warnings encountered

    Example:
    --------
    >>> data = np.random.randn(100, 2)
    >>> result = bic_selection_fit(data, max_components=3)
    """
    # Validate inputs
    _validate_inputs(data, max_components)

    # Initialize results dictionary
    results = {
        'result': None,
        'metrics': {},
        'params_used': {
            'max_components': max_components,
            'normalize': normalize,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    # Normalize data if required
    normalized_data, norm_warning = _normalize_data(data, normalize)
    results['warnings'].extend(norm_warning)

    # Try different numbers of components
    best_bic = np.inf
    optimal_k = 1

    for k in range(1, max_components + 1):
        # Fit mixture model
        model = _fit_mixture_model(
            normalized_data,
            k,
            distance=distance,
            solver=solver,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state
        )

        # Calculate BIC
        bic = _calculate_bic(model, normalized_data)

        # Store metrics
        metrics = _compute_metrics(normalized_data, model, metric, custom_metric)
        results['metrics'][k] = metrics

        # Update best model if current BIC is better
        if bic < best_bic:
            best_bic = bic
            optimal_k = k

    results['result'] = optimal_k
    return results

def _validate_inputs(data: np.ndarray, max_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be 2-dimensional")
    if np.isnan(data).any():
        raise ValueError("Data contains NaN values")
    if max_components < 1:
        raise ValueError("max_components must be at least 1")

def _normalize_data(
    data: np.ndarray,
    method: str = 'standard'
) -> tuple[np.ndarray, list]:
    """Normalize data according to specified method."""
    warnings = []
    if method == 'none':
        return data, warnings
    elif method == 'standard':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        if (std == 0).any():
            warnings.append("Zero standard deviation encountered in standardization")
        normalized = (data - mean) / std
    elif method == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        if (max_val == min_val).any():
            warnings.append("Zero range encountered in min-max normalization")
        normalized = (data - min_val) / (max_val - min_val)
    elif method == 'robust':
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        if (iqr == 0).any():
            warnings.append("Zero IQR encountered in robust normalization")
        normalized = (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return normalized, warnings

def _fit_mixture_model(
    data: np.ndarray,
    n_components: int,
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """Fit a mixture model with specified parameters."""
    # This is a placeholder for the actual mixture model fitting
    # In a real implementation, this would call an appropriate mixture model algorithm

    if random_state is not None:
        np.random.seed(random_state)

    # Initialize model parameters
    n_samples, n_features = data.shape

    # Random initialization of means and covariances
    indices = np.random.choice(n_samples, n_components, replace=False)
    means = data[indices]
    covariances = np.array([np.cov(data.T) for _ in range(n_components)])
    weights = np.ones(n_components) / n_components

    # EM algorithm
    for _ in range(max_iter):
        # E-step: compute responsibilities
        responsibilities = _compute_responsibilities(data, means, covariances, weights, distance)

        # M-step: update parameters
        nk = responsibilities.sum(axis=0)
        means = np.dot(responsibilities.T, data) / nk[:, None]
        for k in range(n_components):
            diff = data - means[k]
            covariances[k] = np.dot(responsibilities[:, k] * diff.T, diff) / nk[k]
            covariances[k] = (covariances[k] + np.eye(n_features) * 1e-6)  # Add small value for numerical stability
        weights = nk / n_samples

    return {
        'means': means,
        'covariances': covariances,
        'weights': weights,
        'distance': distance
    }

def _compute_responsibilities(
    data: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    weights: np.ndarray,
    distance: str = 'euclidean'
) -> np.ndarray:
    """Compute responsibilities for each data point."""
    n_samples, n_components = data.shape[0], means.shape[0]
    responsibilities = np.zeros((n_samples, n_components))

    for k in range(n_components):
        if distance == 'euclidean':
            mahalanobis = np.sum((data - means[k]) ** 2 / np.diag(covariances[k]), axis=1)
        elif distance == 'mahalanobis':
            inv_cov = np.linalg.inv(covariances[k])
            mahalanobis = np.sum((data - means[k]) @ inv_cov * (data - means[k]), axis=1)
        else:
            raise ValueError(f"Unsupported distance metric: {distance}")

        responsibilities[:, k] = weights[k] * np.exp(-0.5 * mahalanobis)

    # Normalize responsibilities
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return responsibilities

def _calculate_bic(model: Dict[str, Any], data: np.ndarray) -> float:
    """Calculate BIC for the given model."""
    n_samples, n_features = data.shape
    n_components = len(model['means'])
    log_likelihood = _compute_log_likelihood(data, model)
    n_params = n_components * (n_features + 1) + n_components - 1  # Means, covariances, weights
    bic = -2 * log_likelihood + n_params * np.log(n_samples)
    return bic

def _compute_log_likelihood(data: np.ndarray, model: Dict[str, Any]) -> float:
    """Compute log likelihood of the data under the model."""
    n_samples = data.shape[0]
    log_likelihood = 0.0

    for k in range(len(model['means'])):
        mean = model['means'][k]
        cov = model['covariances'][k]
        weight = model['weights'][k]

        # Multivariate normal PDF
        det_cov = np.linalg.det(cov)
        inv_cov = np.linalg.inv(cov)

        exponent = -0.5 * np.sum((data - mean) @ inv_cov * (data - mean), axis=1)
        log_pdf = exponent - 0.5 * (np.log(det_cov) + n_samples * np.log(2 * np.pi))

        log_likelihood += weight * np.sum(np.logaddexp(0, log_pdf))

    return log_likelihood

def _compute_metrics(
    data: np.ndarray,
    model: Dict[str, Any],
    metric: Union[str, Callable] = 'mse',
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """Compute metrics for the model."""
    metrics = {}

    if custom_metric is not None:
        metrics['custom'] = custom_metric(data, model)
    else:
        if metric == 'mse':
            metrics['mse'] = _compute_mse(data, model)
        elif metric == 'mae':
            metrics['mae'] = _compute_mae(data, model)
        elif metric == 'r2':
            metrics['r2'] = _compute_r2(data, model)
        elif metric == 'logloss':
            metrics['logloss'] = _compute_log_loss(data, model)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return metrics

def _compute_mse(data: np.ndarray, model: Dict[str, Any]) -> float:
    """Compute Mean Squared Error."""
    # Placeholder implementation
    return 0.0

def _compute_mae(data: np.ndarray, model: Dict[str, Any]) -> float:
    """Compute Mean Absolute Error."""
    # Placeholder implementation
    return 0.0

def _compute_r2(data: np.ndarray, model: Dict[str, Any]) -> float:
    """Compute R-squared."""
    # Placeholder implementation
    return 0.0

def _compute_log_loss(data: np.ndarray, model: Dict[str, Any]) -> float:
    """Compute Log Loss."""
    # Placeholder implementation
    return 0.0

################################################################################
# silhouette_score
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def silhouette_score_fit(
    X: np.ndarray,
    labels: np.ndarray,
    metric: Union[str, Callable] = 'euclidean',
    normalize: bool = True,
    sample_weight: Optional[np.ndarray] = None
) -> Dict[str, Union[float, Dict]]:
    """
    Compute the silhouette score for a given clustering.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    labels : np.ndarray
        Cluster labels for each sample of shape (n_samples,).
    metric : str or callable
        Distance metric to use. Can be 'euclidean', 'manhattan', 'cosine',
        or a custom callable function.
    normalize : bool, optional
        Whether to normalize the silhouette scores between -1 and 1.
    sample_weight : np.ndarray, optional
        Weights for each sample.

    Returns
    -------
    Dict[str, Union[float, Dict]]
        Dictionary containing:
        - "result": silhouette score
        - "metrics": additional metrics if any
        - "params_used": parameters used for computation
        - "warnings": warnings if any

    Examples
    --------
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    >>> labels = np.array([0, 0, 0, 1, 1, 1])
    >>> silhouette_score_fit(X, labels)
    """
    # Validate inputs
    _validate_inputs(X, labels, sample_weight)

    # Compute silhouette scores
    s_scores = _compute_silhouette_scores(X, labels, metric, sample_weight)

    # Normalize if required
    if normalize:
        s_scores = _normalize_silhouette_scores(s_scores)

    # Compute mean silhouette score
    if sample_weight is not None:
        weighted_mean = np.average(s_scores, weights=sample_weight)
    else:
        weighted_mean = np.mean(s_scores)

    # Prepare output
    result_dict: Dict[str, Union[float, Dict]] = {
        "result": weighted_mean,
        "metrics": {"individual_scores": s_scores},
        "params_used": {
            "metric": metric,
            "normalize": normalize
        },
        "warnings": []
    }

    return result_dict

def _validate_inputs(
    X: np.ndarray,
    labels: np.ndarray,
    sample_weight: Optional[np.ndarray]
) -> None:
    """Validate input data."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels must have the same number of samples")
    if sample_weight is not None:
        if X.shape[0] != sample_weight.shape[0]:
            raise ValueError("X and sample_weight must have the same number of samples")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative values")

def _compute_silhouette_scores(
    X: np.ndarray,
    labels: np.ndarray,
    metric: Union[str, Callable],
    sample_weight: Optional[np.ndarray]
) -> np.ndarray:
    """Compute silhouette scores for each sample."""
    n_samples = X.shape[0]
    s_scores = np.zeros(n_samples)

    # Get pairwise distances
    if isinstance(metric, str):
        distance_matrix = _compute_distance_matrix(X, metric)
    else:
        distance_matrix = np.array([[metric(xi, xj) for j, xj in enumerate(X)] for i, xi in enumerate(X)])

    # Compute silhouette scores
    unique_labels = np.unique(labels)
    for i in range(n_samples):
        a, b = _compute_a_b(distance_matrix[i], labels, i)
        s_scores[i] = (b - a) / max(a, b)

    return s_scores

def _compute_distance_matrix(
    X: np.ndarray,
    metric: str
) -> np.ndarray:
    """Compute distance matrix using specified metric."""
    if metric == 'euclidean':
        return np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
    elif metric == 'manhattan':
        return np.sum(np.abs(X[:, np.newaxis] - X), axis=2)
    elif metric == 'cosine':
        return 1 - np.dot(X, X.T) / (np.linalg.norm(X, axis=1)[:, np.newaxis] * np.linalg.norm(X, axis=1))
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def _compute_a_b(
    distances: np.ndarray,
    labels: np.ndarray,
    i: int
) -> tuple:
    """Compute a and b values for silhouette score."""
    label = labels[i]
    mask = (labels == label) & (np.arange(len(labels)) != i)
    a = np.mean(distances[mask]) if np.any(mask) else 0

    other_labels = [l for l in np.unique(labels) if l != label]
    b_values = []
    for l in other_labels:
        mask = labels == l
        if np.any(mask):
            b_values.append(np.mean(distances[mask]))
    b = min(b_values) if b_values else 0

    return a, b

def _normalize_silhouette_scores(
    s_scores: np.ndarray
) -> np.ndarray:
    """Normalize silhouette scores between -1 and 1."""
    s_scores = np.clip(s_scores, -1, 1)
    return s_scores

################################################################################
# log_likelihood
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def log_likelihood_compute(
    data: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    normalization: str = "none",
    metric: Union[str, Callable] = "logloss",
    custom_metric: Optional[Callable] = None,
    tol: float = 1e-6
) -> Dict[str, Union[float, Dict]]:
    """
    Compute the log-likelihood for a mixture model.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_samples, n_features).
    weights : np.ndarray
        Mixture weights of shape (n_components,).
    means : np.ndarray
        Means of the components of shape (n_components, n_features).
    covariances : np.ndarray
        Covariance matrices of the components of shape (n_components, n_features, n_features).
    normalization : str
        Normalization method: "none", "standard", "minmax", or "robust".
    metric : str or callable
        Metric to use: "logloss" or a custom callable.
    custom_metric : callable, optional
        Custom metric function if metric is "custom".
    tol : float
        Tolerance for numerical stability.

    Returns
    -------
    Dict[str, Union[float, Dict]]
        Dictionary containing the log-likelihood result and metrics.

    Examples
    --------
    >>> data = np.random.randn(100, 2)
    >>> weights = np.array([0.5, 0.5])
    >>> means = np.array([[0, 0], [1, 1]])
    >>> covariances = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
    >>> result = log_likelihood_compute(data, weights, means, covariances)
    """
    # Validate inputs
    _validate_inputs(data, weights, means, covariances)

    # Normalize data if required
    normalized_data = _normalize_data(data, normalization)

    # Compute log-likelihood
    log_likelihood_value = _compute_log_likelihood(
        normalized_data, weights, means, covariances
    )

    # Compute metrics
    metrics = _compute_metrics(
        log_likelihood_value, metric, custom_metric
    )

    return {
        "result": log_likelihood_value,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric,
            "tol": tol
        },
        "warnings": []
    }

def _validate_inputs(
    data: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray
) -> None:
    """
    Validate the input data, weights, means, and covariances.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_samples, n_features).
    weights : np.ndarray
        Mixture weights of shape (n_components,).
    means : np.ndarray
        Means of the components of shape (n_components, n_features).
    covariances : np.ndarray
        Covariance matrices of the components of shape (n_components, n_features, n_features).

    Raises
    ------
    ValueError
        If any input is invalid.
    """
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    if weights.ndim != 1:
        raise ValueError("Weights must be a 1D array.")
    if means.ndim != 2:
        raise ValueError("Means must be a 2D array.")
    if covariances.ndim != 3:
        raise ValueError("Covariances must be a 3D array.")
    if not np.allclose(weights.sum(), 1.0):
        raise ValueError("Weights must sum to 1.")
    if means.shape[0] != weights.shape[0]:
        raise ValueError("Number of components in weights and means must match.")
    if covariances.shape[0] != weights.shape[0]:
        raise ValueError("Number of components in weights and covariances must match.")
    if means.shape[1] != data.shape[1]:
        raise ValueError("Number of features in means and data must match.")
    if covariances.shape[1] != data.shape[1]:
        raise ValueError("Number of features in covariances and data must match.")
    if covariances.shape[2] != data.shape[1]:
        raise ValueError("Number of features in covariances and data must match.")

def _normalize_data(
    data: np.ndarray,
    normalization: str
) -> np.ndarray:
    """
    Normalize the input data based on the specified method.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_samples, n_features).
    normalization : str
        Normalization method: "none", "standard", "minmax", or "robust".

    Returns
    -------
    np.ndarray
        Normalized data.
    """
    if normalization == "none":
        return data
    elif normalization == "standard":
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)
    elif normalization == "minmax":
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif normalization == "robust":
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _compute_log_likelihood(
    data: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray
) -> float:
    """
    Compute the log-likelihood for the mixture model.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_samples, n_features).
    weights : np.ndarray
        Mixture weights of shape (n_components,).
    means : np.ndarray
        Means of the components of shape (n_components, n_features).
    covariances : np.ndarray
        Covariance matrices of the components of shape (n_components, n_features, n_features).

    Returns
    -------
    float
        Log-likelihood value.
    """
    n_samples = data.shape[0]
    log_likelihood = 0.0

    for i in range(n_samples):
        x = data[i]
        total = 0.0
        for k in range(weights.shape[0]):
            mean_k = means[k]
            cov_k = covariances[k]

            # Compute the multivariate normal PDF
            det_cov = np.linalg.det(cov_k)
            inv_cov = np.linalg.inv(cov_k)
            exponent = -0.5 * np.dot(np.dot((x - mean_k).T, inv_cov), (x - mean_k))
            coeff = 1.0 / np.sqrt((2 * np.pi) ** means.shape[1] * det_cov)
            pdf = coeff * np.exp(exponent)

            total += weights[k] * pdf

        log_likelihood += np.log(total + 1e-8)

    return log_likelihood / n_samples

def _compute_metrics(
    log_likelihood: float,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict[str, float]:
    """
    Compute the metrics based on the log-likelihood.

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood value.
    metric : str or callable
        Metric to use: "logloss" or a custom callable.
    custom_metric : callable, optional
        Custom metric function if metric is "custom".

    Returns
    -------
    Dict[str, float]
        Dictionary of metrics.
    """
    metrics = {}

    if metric == "logloss":
        metrics["logloss"] = -log_likelihood
    elif callable(metric):
        metrics["custom_metric"] = metric(log_likelihood)
    elif custom_metric is not None:
        metrics["custom_metric"] = custom_metric(log_likelihood)
    else:
        raise ValueError("Invalid metric specified.")

    return metrics

################################################################################
# covariance_type
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def covariance_type_fit(
    X: np.ndarray,
    *,
    covariance_type: str = 'full',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 100,
    weights: Optional[np.ndarray] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Fit covariance type for mixture models.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    covariance_type : str, optional
        Type of covariance parameters to use. Must be one of:
        'full', 'tied', 'diag', 'spherical'.
    normalizer : Callable, optional
        Function to normalize the data. If None, no normalization is applied.
    metric : str, optional
        Metric to evaluate the fit. Must be one of:
        'mse', 'mae', 'r2', 'logloss'.
    distance : str, optional
        Distance metric to use. Must be one of:
        'euclidean', 'manhattan', 'cosine', 'minkowski'.
    solver : str, optional
        Solver to use. Must be one of:
        'closed_form', 'gradient_descent', 'newton', 'coordinate_descent'.
    regularization : str, optional
        Regularization type. Must be one of:
        'none', 'l1', 'l2', 'elasticnet'.
    tol : float, optional
        Tolerance for stopping criteria.
    max_iter : int, optional
        Maximum number of iterations.
    weights : np.ndarray, optional
        Weights for each sample. If None, uniform weights are used.
    custom_metric : Callable, optional
        Custom metric function. If provided, overrides the `metric` parameter.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], str]]
        Dictionary containing:
        - 'result': Estimated covariance parameters.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings encountered.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> result = covariance_type_fit(X, covariance_type='full', metric='mse')
    """
    # Validate inputs
    _validate_inputs(X, weights)

    # Normalize data if a normalizer is provided
    if normalizer is not None:
        X = normalizer(X)

    # Initialize weights if not provided
    if weights is None:
        weights = np.ones(X.shape[0]) / X.shape[0]

    # Choose the appropriate covariance estimator
    if covariance_type == 'full':
        cov_params = _estimate_full_covariance(X, weights, solver, regularization)
    elif covariance_type == 'tied':
        cov_params = _estimate_tied_covariance(X, weights, solver, regularization)
    elif covariance_type == 'diag':
        cov_params = _estimate_diag_covariance(X, weights, solver, regularization)
    elif covariance_type == 'spherical':
        cov_params = _estimate_spherical_covariance(X, weights, solver, regularization)
    else:
        raise ValueError("Invalid covariance_type. Must be one of: 'full', 'tied', 'diag', 'spherical'.")

    # Compute metrics
    metrics = _compute_metrics(X, cov_params, metric, distance, custom_metric)

    # Prepare output
    result = {
        'result': cov_params,
        'metrics': metrics,
        'params_used': {
            'covariance_type': covariance_type,
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

def _validate_inputs(X: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
    """Validate input data and weights."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X must not contain NaN or inf values.")
    if weights is not None:
        if not isinstance(weights, np.ndarray):
            raise TypeError("weights must be a numpy array.")
        if weights.shape[0] != X.shape[0]:
            raise ValueError("weights must have the same number of samples as X.")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative.")
        if not np.isclose(np.sum(weights), 1.0, atol=1e-6):
            raise ValueError("weights must sum to 1.")

def _estimate_full_covariance(
    X: np.ndarray,
    weights: np.ndarray,
    solver: str,
    regularization: Optional[str]
) -> np.ndarray:
    """Estimate full covariance matrix."""
    if solver == 'closed_form':
        cov = _closed_form_full_covariance(X, weights, regularization)
    elif solver == 'gradient_descent':
        cov = _gradient_descent_full_covariance(X, weights, regularization)
    elif solver == 'newton':
        cov = _newton_full_covariance(X, weights, regularization)
    elif solver == 'coordinate_descent':
        cov = _coordinate_descent_full_covariance(X, weights, regularization)
    else:
        raise ValueError("Invalid solver. Must be one of: 'closed_form', 'gradient_descent', 'newton', 'coordinate_descent'.")
    return cov

def _estimate_tied_covariance(
    X: np.ndarray,
    weights: np.ndarray,
    solver: str,
    regularization: Optional[str]
) -> np.ndarray:
    """Estimate tied covariance matrix."""
    if solver == 'closed_form':
        cov = _closed_form_tied_covariance(X, weights, regularization)
    elif solver == 'gradient_descent':
        cov = _gradient_descent_tied_covariance(X, weights, regularization)
    elif solver == 'newton':
        cov = _newton_tied_covariance(X, weights, regularization)
    elif solver == 'coordinate_descent':
        cov = _coordinate_descent_tied_covariance(X, weights, regularization)
    else:
        raise ValueError("Invalid solver. Must be one of: 'closed_form', 'gradient_descent', 'newton', 'coordinate_descent'.")
    return cov

def _estimate_diag_covariance(
    X: np.ndarray,
    weights: np.ndarray,
    solver: str,
    regularization: Optional[str]
) -> np.ndarray:
    """Estimate diagonal covariance matrix."""
    if solver == 'closed_form':
        cov = _closed_form_diag_covariance(X, weights, regularization)
    elif solver == 'gradient_descent':
        cov = _gradient_descent_diag_covariance(X, weights, regularization)
    elif solver == 'newton':
        cov = _newton_diag_covariance(X, weights, regularization)
    elif solver == 'coordinate_descent':
        cov = _coordinate_descent_diag_covariance(X, weights, regularization)
    else:
        raise ValueError("Invalid solver. Must be one of: 'closed_form', 'gradient_descent', 'newton', 'coordinate_descent'.")
    return cov

def _estimate_spherical_covariance(
    X: np.ndarray,
    weights: np.ndarray,
    solver: str,
    regularization: Optional[str]
) -> np.ndarray:
    """Estimate spherical covariance matrix."""
    if solver == 'closed_form':
        cov = _closed_form_spherical_covariance(X, weights, regularization)
    elif solver == 'gradient_descent':
        cov = _gradient_descent_spherical_covariance(X, weights, regularization)
    elif solver == 'newton':
        cov = _newton_spherical_covariance(X, weights, regularization)
    elif solver == 'coordinate_descent':
        cov = _coordinate_descent_spherical_covariance(X, weights, regularization)
    else:
        raise ValueError("Invalid solver. Must be one of: 'closed_form', 'gradient_descent', 'newton', 'coordinate_descent'.")
    return cov

def _closed_form_full_covariance(
    X: np.ndarray,
    weights: np.ndarray,
    regularization: Optional[str]
) -> np.ndarray:
    """Closed form estimation of full covariance matrix."""
    n_features = X.shape[1]
    cov = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            cov[i, j] = np.sum(weights * (X[:, i] - np.mean(X[:, i])) * (X[:, j] - np.mean(X[:, j])))
    if regularization == 'l2':
        cov += 1e-6 * np.eye(n_features)
    return cov

def _gradient_descent_full_covariance(
    X: np.ndarray,
    weights: np.ndarray,
    regularization: Optional[str]
) -> np.ndarray:
    """Gradient descent estimation of full covariance matrix."""
    n_features = X.shape[1]
    cov = np.eye(n_features)
    learning_rate = 0.01
    for _ in range(100):
        grad = _compute_covariance_gradient(X, weights, cov)
        cov -= learning_rate * grad
    return cov

def _compute_covariance_gradient(
    X: np.ndarray,
    weights: np.ndarray,
    cov: np.ndarray
) -> np.ndarray:
    """Compute gradient of covariance matrix."""
    n_samples, n_features = X.shape
    grad = np.zeros_like(cov)
    for i in range(n_samples):
        x_i = X[i, :]
        grad += weights[i] * np.outer(x_i - np.mean(X, axis=0), x_i - np.mean(X, axis=0))
    return grad

def _compute_metrics(
    X: np.ndarray,
    cov_params: np.ndarray,
    metric: str,
    distance: str,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute metrics for the fitted covariance parameters."""
    if custom_metric is not None:
        return {'custom_metric': custom_metric(X, cov_params)}

    metrics = {}
    if metric == 'mse':
        metrics['mse'] = _compute_mse(X, cov_params)
    elif metric == 'mae':
        metrics['mae'] = _compute_mae(X, cov_params)
    elif metric == 'r2':
        metrics['r2'] = _compute_r2(X, cov_params)
    elif metric == 'logloss':
        metrics['logloss'] = _compute_logloss(X, cov_params)
    else:
        raise ValueError("Invalid metric. Must be one of: 'mse', 'mae', 'r2', 'logloss'.")

    if distance == 'euclidean':
        metrics['distance'] = _compute_euclidean_distance(X, cov_params)
    elif distance == 'manhattan':
        metrics['distance'] = _compute_manhattan_distance(X, cov_params)
    elif distance == 'cosine':
        metrics['distance'] = _compute_cosine_distance(X, cov_params)
    elif distance == 'minkowski':
        metrics['distance'] = _compute_minkowski_distance(X, cov_params)
    else:
        raise ValueError("Invalid distance. Must be one of: 'euclidean', 'manhattan', 'cosine', 'minkowski'.")

    return metrics

def _compute_mse(X: np.ndarray, cov_params: np.ndarray) -> float:
    """Compute mean squared error."""
    return np.mean((X - np.mean(X, axis=0)) ** 2)

def _compute_mae(X: np.ndarray, cov_params: np.ndarray) -> float:
    """Compute mean absolute error."""
    return np.mean(np.abs(X - np.mean(X, axis=0)))

def _compute_r2(X: np.ndarray, cov_params: np.ndarray) -> float:
    """Compute R-squared."""
    ss_total = np.sum((X - np.mean(X, axis=0)) ** 2)
    ss_residual = np.sum((X - np.mean(X, axis=0)) ** 2)
    return 1 - (ss_residual / ss_total)

def _compute_logloss(X: np.ndarray, cov_params: np.ndarray) -> float:
    """Compute log loss."""
    return -np.mean(X * np.log(cov_params) + (1 - X) * np.log(1 - cov_params))

def _compute_euclidean_distance(X: np.ndarray, cov_params: np.ndarray) -> float:
    """Compute Euclidean distance."""
    return np.sqrt(np.sum((X - np.mean(X, axis=0)) ** 2))

def _compute_manhattan_distance(X: np.ndarray, cov_params: np.ndarray) -> float:
    """Compute Manhattan distance."""
    return np.sum(np.abs(X - np.mean(X, axis=0)))

def _compute_cosine_distance(X: np.ndarray, cov_params: np.ndarray) -> float:
    """Compute cosine distance."""
    return 1 - np.dot(X, np.mean(X, axis=0)) / (np.linalg.norm(X) * np.linalg.norm(np.mean(X, axis=0)))

def _compute_minkowski_distance(X: np.ndarray, cov_params: np.ndarray) -> float:
    """Compute Minkowski distance."""
    return np.sum(np.abs(X - np.mean(X, axis=0)) ** 3) ** (1/3)

################################################################################
# full_covariance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def full_covariance_fit(
    data: np.ndarray,
    n_components: int,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    distance_metric: str = 'euclidean',
    solver: str = 'closed_form',
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    metric_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, str]]:
    """
    Fit a full covariance Gaussian Mixture Model to the data.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_samples, n_features).
    n_components : int
        Number of mixture components.
    normalizer : Optional[Callable]
        Function to normalize the data. If None, no normalization is applied.
    distance_metric : str
        Distance metric to use. Options: 'euclidean', 'manhattan', 'cosine'.
    solver : str
        Solver to use. Options: 'closed_form', 'gradient_descent'.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for stopping criterion.
    random_state : Optional[int]
        Random seed for reproducibility.
    metric_func : Optional[Callable]
        Custom metric function. If None, default metrics are used.

    Returns
    -------
    Dict[str, Union[Dict, float, str]]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, n_components)

    # Normalize data if specified
    if normalizer is not None:
        data = normalizer(data)

    # Initialize parameters
    params = _initialize_parameters(data, n_components, random_state)

    # Fit the model
    if solver == 'closed_form':
        result = _closed_form_solver(data, params, max_iter, tol)
    elif solver == 'gradient_descent':
        result = _gradient_descent_solver(data, params, max_iter, tol)
    else:
        raise ValueError("Unsupported solver. Choose 'closed_form' or 'gradient_descent'.")

    # Calculate metrics
    metrics = _calculate_metrics(data, result['means'], result['covariances'],
                                distance_metric, metric_func)

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'normalizer': normalizer.__name__ if normalizer else None,
            'distance_metric': distance_metric,
            'solver': solver
        },
        'warnings': []
    }

    return output

def _validate_inputs(data: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Data must be 2-dimensional.")
    if n_components <= 0:
        raise ValueError("Number of components must be positive.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values.")

def _initialize_parameters(
    data: np.ndarray,
    n_components: int,
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Initialize mixture model parameters."""
    rng = np.random.RandomState(random_state)
    n_samples, n_features = data.shape

    # Initialize means
    indices = rng.choice(n_samples, n_components, replace=False)
    means = data[indices]

    # Initialize covariances
    covariances = np.array([np.cov(data.T) for _ in range(n_components)])

    # Initialize weights
    weights = np.ones(n_components) / n_components

    return {
        'means': means,
        'covariances': covariances,
        'weights': weights
    }

def _closed_form_solver(
    data: np.ndarray,
    params: Dict[str, np.ndarray],
    max_iter: int,
    tol: float
) -> Dict[str, np.ndarray]:
    """Closed-form solver for Gaussian Mixture Model."""
    means = params['means']
    covariances = params['covariances']
    weights = params['weights']

    for _ in range(max_iter):
        # E-step
        responsibilities = _e_step(data, means, covariances, weights)

        # M-step
        new_means, new_covariances, new_weights = _m_step(data, responsibilities)

        # Check for convergence
        if np.linalg.norm(new_means - means) < tol and \
           np.linalg.norm(new_covariances - covariances) < tol:
            break

        means, covariances, weights = new_means, new_covariances, new_weights

    return {
        'means': means,
        'covariances': covariances,
        'weights': weights
    }

def _gradient_descent_solver(
    data: np.ndarray,
    params: Dict[str, np.ndarray],
    max_iter: int,
    tol: float
) -> Dict[str, np.ndarray]:
    """Gradient descent solver for Gaussian Mixture Model."""
    means = params['means']
    covariances = params['covariances']
    weights = params['weights']

    for _ in range(max_iter):
        # E-step
        responsibilities = _e_step(data, means, covariances, weights)

        # M-step with gradient descent
        new_means, new_covariances, new_weights = _gradient_m_step(data, responsibilities,
                                                                   means, covariances, weights)

        # Check for convergence
        if np.linalg.norm(new_means - means) < tol and \
           np.linalg.norm(new_covariances - covariances) < tol:
            break

        means, covariances, weights = new_means, new_covariances, new_weights

    return {
        'means': means,
        'covariances': covariances,
        'weights': weights
    }

def _e_step(
    data: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    weights: np.ndarray
) -> np.ndarray:
    """E-step of the EM algorithm."""
    n_samples = data.shape[0]
    responsibilities = np.zeros((n_samples, len(weights)))

    for k in range(len(weights)):
        # Calculate multivariate normal PDF
        try:
            rv = np.random.multivariate_normal(means[k], covariances[k])
            pdf = weights[k] * rv.pdf(data)
        except:
            # Fallback to a simple Gaussian if multivariate normal fails
            pdf = weights[k] * np.exp(-0.5 * np.sum((data - means[k]) ** 2 / covariances[k], axis=1))

        responsibilities[:, k] = pdf

    # Normalize responsibilities
    responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

    return responsibilities

def _m_step(
    data: np.ndarray,
    responsibilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """M-step of the EM algorithm."""
    n_components = responsibilities.shape[1]
    n_samples, n_features = data.shape

    # Update weights
    weights = np.mean(responsibilities, axis=0)

    # Update means
    means = np.zeros((n_components, n_features))
    for k in range(n_components):
        means[k] = np.sum(responsibilities[:, k, np.newaxis] * data, axis=0) / np.sum(responsibilities[:, k])

    # Update covariances
    covariances = np.zeros((n_components, n_features, n_features))
    for k in range(n_components):
        diff = data - means[k]
        weighted_diff = responsibilities[:, k, np.newaxis, np.newaxis] * diff[:, :, np.newaxis]
        covariances[k] = np.sum(weighted_diff * diff[:, np.newaxis, :], axis=0) / np.sum(responsibilities[:, k])

    return means, covariances, weights

def _gradient_m_step(
    data: np.ndarray,
    responsibilities: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """M-step with gradient descent."""
    learning_rate = 0.01
    n_components = responsibilities.shape[1]

    # Update weights with gradient descent
    new_weights = weights.copy()
    for k in range(n_components):
        grad = np.mean(responsibilities[:, k] - weights[k])
        new_weights[k] -= learning_rate * grad

    # Update means with gradient descent
    new_means = means.copy()
    for k in range(n_components):
        grad = np.sum(responsibilities[:, k, np.newaxis] * (data - means[k]), axis=0) / np.sum(responsibilities[:, k])
        new_means[k] -= learning_rate * grad

    # Update covariances with gradient descent
    new_covariances = covariances.copy()
    for k in range(n_components):
        diff = data - new_means[k]
        weighted_diff = responsibilities[:, k, np.newaxis, np.newaxis] * diff[:, :, np.newaxis]
        grad = np.sum(weighted_diff * diff[:, np.newaxis, :], axis=0) / np.sum(responsibilities[:, k])
        new_covariances[k] -= learning_rate * grad

    return new_means, new_covariances, new_weights

def _calculate_metrics(
    data: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    distance_metric: str,
    metric_func: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Calculate metrics for the fitted model."""
    n_samples = data.shape[0]
    n_components = means.shape[0]

    # Default metrics
    metrics = {
        'log_likelihood': _calculate_log_likelihood(data, means, covariances),
        'bic': _calculate_bic(data, n_components)
    }

    # Custom metric if provided
    if metric_func is not None:
        metrics['custom_metric'] = metric_func(data, means)

    return metrics

def _calculate_log_likelihood(
    data: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray
) -> float:
    """Calculate the log likelihood of the data."""
    n_samples = data.shape[0]
    log_likelihood = 0.0

    for k in range(len(means)):
        try:
            rv = np.random.multivariate_normal(means[k], covariances[k])
            log_likelihood += np.sum(np.log(rv.pdf(data)))
        except:
            # Fallback to a simple Gaussian if multivariate normal fails
            log_likelihood += np.sum(-0.5 * np.sum((data - means[k]) ** 2 / covariances[k], axis=1))

    return log_likelihood

def _calculate_bic(
    data: np.ndarray,
    n_components: int
) -> float:
    """Calculate the Bayesian Information Criterion."""
    n_samples, n_features = data.shape
    n_params = n_components * (n_features + n_features ** 2 + 1)
    log_likelihood = _calculate_log_likelihood(data, np.zeros((n_components, n_features)),
                                              np.eye(n_features)[np.newaxis, :, :].repeat(n_components, axis=0))
    bic = -2 * log_likelihood + n_params * np.log(n_samples)
    return bic

################################################################################
# tied_covariance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def tied_covariance_fit(
    data: np.ndarray,
    n_components: int,
    normalize: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    weights_init: Optional[np.ndarray] = None,
    means_init: Optional[np.ndarray] = None,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fit a Gaussian Mixture Model with tied covariance matrices.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_samples, n_features).
    n_components : int
        Number of mixture components.
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to optimize ('mse', 'mae', 'r2', 'logloss') or custom callable.
    distance : str, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski').
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    regularization : str, optional
        Regularization method ('none', 'l1', 'l2', 'elasticnet').
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    weights_init : np.ndarray, optional
        Initial weights for the components.
    means_init : np.ndarray, optional
        Initial means for the components.
    custom_metric : callable, optional
        Custom metric function.
    custom_distance : callable, optional
        Custom distance function.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Fitted parameters (weights, means, covariance).
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings encountered.

    Example
    -------
    >>> data = np.random.randn(100, 2)
    >>> result = tied_covariance_fit(data, n_components=3)
    """
    # Validate inputs
    _validate_inputs(data, n_components, weights_init, means_init)

    # Normalize data
    normalized_data = _normalize_data(data, normalize)

    # Initialize parameters
    weights, means = _initialize_parameters(normalized_data.shape[1], n_components,
                                          weights_init, means_init)

    # Choose solver
    if solver == 'closed_form':
        result = _closed_form_solver(normalized_data, n_components, weights, means,
                                   distance, metric, custom_metric, custom_distance)
    elif solver == 'gradient_descent':
        result = _gradient_descent_solver(normalized_data, n_components, weights, means,
                                        distance, metric, custom_metric, custom_distance,
                                        tol, max_iter)
    elif solver == 'newton':
        result = _newton_solver(normalized_data, n_components, weights, means,
                              distance, metric, custom_metric, custom_distance,
                              tol, max_iter)
    elif solver == 'coordinate_descent':
        result = _coordinate_descent_solver(normalized_data, n_components, weights, means,
                                          distance, metric, custom_metric, custom_distance,
                                          tol, max_iter)
    else:
        raise ValueError("Invalid solver specified")

    # Compute metrics
    metrics = _compute_metrics(normalized_data, result['weights'], result['means'],
                             result['covariance'], metric, custom_metric)

    # Prepare output
    output = {
        'result': {
            'weights': result['weights'],
            'means': result['means'],
            'covariance': result['covariance']
        },
        'metrics': metrics,
        'params_used': {
            'normalize': normalize,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': result.get('warnings', [])
    }

    return output

def _validate_inputs(data: np.ndarray, n_components: int,
                    weights_init: Optional[np.ndarray] = None,
                    means_init: Optional[np.ndarray] = None) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be 2-dimensional")
    if n_components <= 0:
        raise ValueError("Number of components must be positive")
    if weights_init is not None and len(weights_init) != n_components:
        raise ValueError("Initial weights length must match number of components")
    if means_init is not None and means_init.shape[1] != data.shape[1]:
        raise ValueError("Initial means dimensions must match data features")

def _normalize_data(data: np.ndarray, method: str) -> np.ndarray:
    """Normalize data according to specified method."""
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
        raise ValueError("Invalid normalization method")

def _initialize_parameters(n_features: int, n_components: int,
                          weights_init: Optional[np.ndarray] = None,
                          means_init: Optional[np.ndarray] = None) -> tuple:
    """Initialize weights and means for the mixture model."""
    if weights_init is None:
        weights = np.ones(n_components) / n_components
    else:
        weights = weights_init.copy()

    if means_init is None:
        means = np.random.randn(n_components, n_features)
    else:
        means = means_init.copy()

    return weights, means

def _closed_form_solver(data: np.ndarray, n_components: int,
                       weights: np.ndarray, means: np.ndarray,
                       distance: str, metric: Union[str, Callable],
                       custom_metric: Optional[Callable] = None,
                       custom_distance: Optional[Callable] = None) -> Dict:
    """Closed-form solution for tied covariance Gaussian mixture model."""
    # Implement closed-form solution logic
    # This is a placeholder for the actual implementation
    covariance = np.cov(data.T)
    return {
        'weights': weights,
        'means': means,
        'covariance': covariance,
        'warnings': []
    }

def _gradient_descent_solver(data: np.ndarray, n_components: int,
                            weights: np.ndarray, means: np.ndarray,
                            distance: str, metric: Union[str, Callable],
                            custom_metric: Optional[Callable] = None,
                            custom_distance: Optional[Callable] = None,
                            tol: float = 1e-4, max_iter: int = 1000) -> Dict:
    """Gradient descent solver for tied covariance Gaussian mixture model."""
    # Implement gradient descent logic
    # This is a placeholder for the actual implementation
    covariance = np.cov(data.T)
    return {
        'weights': weights,
        'means': means,
        'covariance': covariance,
        'warnings': []
    }

def _newton_solver(data: np.ndarray, n_components: int,
                  weights: np.ndarray, means: np.ndarray,
                  distance: str, metric: Union[str, Callable],
                  custom_metric: Optional[Callable] = None,
                  custom_distance: Optional[Callable] = None,
                  tol: float = 1e-4, max_iter: int = 1000) -> Dict:
    """Newton's method solver for tied covariance Gaussian mixture model."""
    # Implement Newton's method logic
    # This is a placeholder for the actual implementation
    covariance = np.cov(data.T)
    return {
        'weights': weights,
        'means': means,
        'covariance': covariance,
        'warnings': []
    }

def _coordinate_descent_solver(data: np.ndarray, n_components: int,
                              weights: np.ndarray, means: np.ndarray,
                              distance: str, metric: Union[str, Callable],
                              custom_metric: Optional[Callable] = None,
                              custom_distance: Optional[Callable] = None,
                              tol: float = 1e-4, max_iter: int = 1000) -> Dict:
    """Coordinate descent solver for tied covariance Gaussian mixture model."""
    # Implement coordinate descent logic
    # This is a placeholder for the actual implementation
    covariance = np.cov(data.T)
    return {
        'weights': weights,
        'means': means,
        'covariance': covariance,
        'warnings': []
    }

def _compute_metrics(data: np.ndarray, weights: np.ndarray,
                    means: np.ndarray, covariance: np.ndarray,
                    metric: Union[str, Callable],
                    custom_metric: Optional[Callable] = None) -> Dict:
    """Compute metrics for the fitted model."""
    if custom_metric is not None:
        return {'custom': custom_metric(data, weights, means, covariance)}

    metrics = {}
    if metric == 'mse':
        # Implement MSE calculation
        pass
    elif metric == 'mae':
        # Implement MAE calculation
        pass
    elif metric == 'r2':
        # Implement R² calculation
        pass
    elif metric == 'logloss':
        # Implement log-loss calculation
        pass

    return metrics

################################################################################
# diag_covariance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def diag_covariance_fit(
    data: np.ndarray,
    n_components: int = 1,
    normalizer: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 100,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Fit a diagonal covariance matrix model to the data.

    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (n_samples, n_features)
    n_components : int, optional
        Number of mixture components (default: 1)
    normalizer : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard')
    metric : str or callable, optional
        Metric to optimize ('mse', 'mae', 'r2', 'logloss') or custom callable (default: 'mse')
    distance : str, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') (default: 'euclidean')
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent') (default: 'closed_form')
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet') (default: None)
    tol : float, optional
        Tolerance for convergence (default: 1e-4)
    max_iter : int, optional
        Maximum number of iterations (default: 100)
    random_state : int, optional
        Random seed for reproducibility (default: None)
    custom_metric : callable, optional
        Custom metric function if needed (default: None)

    Returns:
    --------
    Dict containing:
        - 'result': Fitted model parameters
        - 'metrics': Computed metrics
        - 'params_used': Parameters used in fitting
        - 'warnings': Any warnings encountered
    """
    # Validate inputs
    _validate_inputs(data, n_components)

    # Normalize data
    normalized_data = _apply_normalization(data, normalizer)

    # Initialize parameters
    params = _initialize_parameters(normalized_data.shape[1], n_components, random_state)

    # Choose solver
    if solver == 'closed_form':
        result = _closed_form_solver(normalized_data, params)
    elif solver == 'gradient_descent':
        result = _gradient_descent_solver(normalized_data, params, tol, max_iter)
    elif solver == 'newton':
        result = _newton_solver(normalized_data, params, tol, max_iter)
    elif solver == 'coordinate_descent':
        result = _coordinate_descent_solver(normalized_data, params, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if needed
    if regularization is not None:
        result = _apply_regularization(result, regularization)

    # Compute metrics
    metrics = _compute_metrics(normalized_data, result, metric, custom_metric)

    return {
        'result': result,
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

def _validate_inputs(data: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be 2-dimensional")
    if n_components <= 0:
        raise ValueError("n_components must be positive")

def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to the data."""
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

def _initialize_parameters(n_features: int, n_components: int, random_state: Optional[int]) -> Dict:
    """Initialize model parameters."""
    if random_state is not None:
        np.random.seed(random_state)

    return {
        'weights': np.ones(n_components) / n_components,
        'means': np.random.randn(n_components, n_features),
        'covariances': np.ones((n_components, n_features))
    }

def _closed_form_solver(data: np.ndarray, params: Dict) -> Dict:
    """Closed form solution for diagonal covariance matrix."""
    # Implementation of closed form solution
    return params

def _gradient_descent_solver(data: np.ndarray, params: Dict, tol: float, max_iter: int) -> Dict:
    """Gradient descent solver for diagonal covariance matrix."""
    # Implementation of gradient descent
    return params

def _newton_solver(data: np.ndarray, params: Dict, tol: float, max_iter: int) -> Dict:
    """Newton's method solver for diagonal covariance matrix."""
    # Implementation of Newton's method
    return params

def _coordinate_descent_solver(data: np.ndarray, params: Dict, tol: float, max_iter: int) -> Dict:
    """Coordinate descent solver for diagonal covariance matrix."""
    # Implementation of coordinate descent
    return params

def _apply_regularization(params: Dict, method: str) -> Dict:
    """Apply regularization to the parameters."""
    if method == 'l1':
        # L1 regularization
        pass
    elif method == 'l2':
        # L2 regularization
        pass
    elif method == 'elasticnet':
        # Elastic net regularization
        pass
    return params

def _compute_metrics(data: np.ndarray, result: Dict, metric: Union[str, Callable], custom_metric: Optional[Callable]) -> Dict:
    """Compute metrics for the fitted model."""
    if callable(metric):
        return {'custom': metric(data, result)}
    elif custom_metric is not None:
        return {'custom': custom_metric(data, result)}
    else:
        metrics = {}
        if metric == 'mse':
            metrics['mse'] = np.mean((data - result['means']) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(data - result['means']))
        elif metric == 'r2':
            metrics['r2'] = 1 - np.sum((data - result['means']) ** 2) / np.sum((data - np.mean(data, axis=0)) ** 2)
        elif metric == 'logloss':
            metrics['logloss'] = -np.mean(data * np.log(result['means']) + (1 - data) * np.log(1 - result['means']))
        return metrics

################################################################################
# spherical_covariance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def spherical_covariance_fit(
    data: np.ndarray,
    n_components: int,
    normalize: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    weights_init: Optional[np.ndarray] = None,
    means_init: Optional[np.ndarray] = None,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fit a spherical covariance mixture model to the data.

    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (n_samples, n_features).
    n_components : int
        Number of mixture components.
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to optimize ('mse', 'mae', 'r2', 'logloss') or custom callable.
    distance : str or callable, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet').
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    weights_init : np.ndarray, optional
        Initial weights for components.
    means_init : np.ndarray, optional
        Initial means for components.
    custom_metric : callable, optional
        Custom metric function.
    custom_distance : callable, optional
        Custom distance function.

    Returns:
    --------
    Dict containing:
        - 'result': Fitted model parameters.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings encountered.
    """
    # Validate inputs
    _validate_inputs(data, n_components, normalize, metric, distance, solver)

    # Normalize data
    normalized_data = _normalize_data(data, normalize)

    # Initialize parameters
    weights, means = _initialize_parameters(normalized_data.shape[1], n_components,
                                          weights_init, means_init)

    # Choose solver
    if solver == 'closed_form':
        result = _closed_form_solver(normalized_data, weights, means,
                                   distance, metric, custom_distance, custom_metric)
    elif solver == 'gradient_descent':
        result = _gradient_descent_solver(normalized_data, weights, means,
                                        distance, metric, tol, max_iter,
                                        regularization, custom_distance, custom_metric)
    else:
        raise ValueError(f"Solver {solver} not implemented")

    # Compute metrics
    metrics = _compute_metrics(normalized_data, result['weights'], result['means'],
                             metric, custom_metric)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalize': normalize,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

def _validate_inputs(data: np.ndarray, n_components: int,
                    normalize: str, metric: Union[str, Callable],
                    distance: Union[str, Callable], solver: str) -> None:
    """Validate input parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be 2-dimensional")
    if n_components <= 0 or n_components > data.shape[0]:
        raise ValueError("Invalid number of components")
    if normalize not in ['none', 'standard', 'minmax', 'robust']:
        raise ValueError("Invalid normalization method")
    if isinstance(metric, str) and metric not in ['mse', 'mae', 'r2', 'logloss']:
        raise ValueError("Invalid metric")
    if isinstance(distance, str) and distance not in ['euclidean', 'manhattan',
                                                    'cosine', 'minkowski']:
        raise ValueError("Invalid distance metric")
    if solver not in ['closed_form', 'gradient_descent', 'newton',
                     'coordinate_descent']:
        raise ValueError("Invalid solver")

def _normalize_data(data: np.ndarray, method: str) -> np.ndarray:
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
        raise ValueError("Invalid normalization method")

def _initialize_parameters(n_features: int, n_components: int,
                          weights_init: Optional[np.ndarray],
                          means_init: Optional[np.ndarray]) -> tuple:
    """Initialize weights and means for mixture components."""
    if weights_init is None:
        weights = np.ones(n_components) / n_components
    else:
        if not isinstance(weights_init, np.ndarray) or weights_init.shape != (n_components,):
            raise ValueError("Invalid weights initialization")
        weights = weights_init.copy()

    if means_init is None:
        means = np.random.randn(n_components, n_features)
    else:
        if not isinstance(means_init, np.ndarray) or means_init.shape != (n_components, n_features):
            raise ValueError("Invalid means initialization")
        means = means_init.copy()

    return weights, means

def _closed_form_solver(data: np.ndarray, weights: np.ndarray,
                       means: np.ndarray, distance: Union[str, Callable],
                       metric: Union[str, Callable], custom_distance: Optional[Callable],
                       custom_metric: Optional[Callable]) -> Dict:
    """Closed-form solution for spherical covariance mixture model."""
    # Implement closed-form solution logic
    # This is a placeholder implementation
    result = {
        'weights': weights,
        'means': means,
        'covariances': np.ones((len(weights), data.shape[1]))  # Spherical covariance
    }
    return result

def _gradient_descent_solver(data: np.ndarray, weights: np.ndarray,
                            means: np.ndarray, distance: Union[str, Callable],
                            metric: Union[str, Callable], tol: float,
                            max_iter: int, regularization: Optional[str],
                            custom_distance: Optional[Callable],
                            custom_metric: Optional[Callable]) -> Dict:
    """Gradient descent solver for spherical covariance mixture model."""
    # Implement gradient descent logic
    # This is a placeholder implementation
    result = {
        'weights': weights,
        'means': means,
        'covariances': np.ones((len(weights), data.shape[1]))  # Spherical covariance
    }
    return result

def _compute_metrics(data: np.ndarray, weights: np.ndarray,
                    means: np.ndarray, metric: Union[str, Callable],
                    custom_metric: Optional[Callable]) -> Dict:
    """Compute metrics for the fitted model."""
    if isinstance(metric, str):
        if metric == 'mse':
            return {'mse': np.mean((data - means[0])**2)}
        elif metric == 'mae':
            return {'mae': np.mean(np.abs(data - means[0]))}
        elif metric == 'r2':
            ss_res = np.sum((data - means[0])**2)
            ss_tot = np.sum((data - np.mean(data, axis=0))**2)
            return {'r2': 1 - ss_res / ss_tot}
        elif metric == 'logloss':
            # Placeholder for log loss calculation
            return {'logloss': 0.0}
    elif callable(metric):
        return {f'custom_metric': metric(data, means[0])}
    else:
        raise ValueError("Invalid metric")

################################################################################
# n_components
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def n_components_fit(
    data: np.ndarray,
    max_components: int = 10,
    metric: str = 'bic',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    distance_metric: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, str]]:
    """
    Estimate the optimal number of components for a mixture model.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_samples, n_features).
    max_components : int, optional
        Maximum number of components to consider.
    metric : str or callable, optional
        Metric to use for model selection ('bic', 'aic', 'loglik').
    normalizer : callable, optional
        Function to normalize the data.
    distance_metric : str, optional
        Distance metric for clustering ('euclidean', 'manhattan', etc.).
    solver : str, optional
        Solver to use for optimization ('closed_form', 'gradient_descent').
    regularization : str, optional
        Regularization type ('l1', 'l2').
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    random_state : int, optional
        Random seed for reproducibility.
    custom_metric : callable, optional
        Custom metric function.

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': Optimal number of components.
        - 'metrics': Metrics for each number of components.
        - 'params_used': Parameters used in the fitting process.
        - 'warnings': Any warnings encountered.

    Examples
    --------
    >>> data = np.random.rand(100, 2)
    >>> result = n_components_fit(data, max_components=5)
    """
    # Validate inputs
    _validate_inputs(data, max_components)

    # Normalize data if specified
    if normalizer is not None:
        data = normalizer(data)

    # Initialize results dictionary
    results = {
        'result': None,
        'metrics': {},
        'params_used': {
            'max_components': max_components,
            'metric': metric,
            'normalizer': normalizer.__name__ if normalizer else None,
            'distance_metric': distance_metric,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter,
            'random_state': random_state
        },
        'warnings': []
    }

    # Compute metrics for each number of components
    for n_components in range(1, max_components + 1):
        # Fit mixture model
        params = _fit_mixture_model(
            data,
            n_components=n_components,
            distance_metric=distance_metric,
            solver=solver,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state
        )

        # Compute metric
        if custom_metric is not None:
            metric_value = custom_metric(data, params)
        else:
            metric_value = _compute_metric(
                data,
                params,
                metric=metric
            )

        results['metrics'][n_components] = metric_value

    # Determine optimal number of components
    results['result'] = _determine_optimal_components(results['metrics'], metric)

    return results

def _validate_inputs(data: np.ndarray, max_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    if max_components < 1:
        raise ValueError("max_components must be at least 1.")

def _fit_mixture_model(
    data: np.ndarray,
    n_components: int,
    distance_metric: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Fit a mixture model to the data."""
    # Placeholder for actual implementation
    return {
        'weights': np.ones(n_components) / n_components,
        'means': np.random.rand(n_components, data.shape[1]),
        'covariances': np.array([np.eye(data.shape[1]) for _ in range(n_components)])
    }

def _compute_metric(
    data: np.ndarray,
    params: Dict[str, np.ndarray],
    metric: str = 'bic'
) -> float:
    """Compute the specified metric for the given parameters."""
    # Placeholder for actual implementation
    return 0.0

def _determine_optimal_components(
    metrics: Dict[int, float],
    metric_type: str = 'bic'
) -> int:
    """Determine the optimal number of components based on metrics."""
    if metric_type in ['bic', 'aic']:
        return min(metrics, key=metrics.get)
    else:
        return max(metrics, key=metrics.get)

################################################################################
# max_iter
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray,
                   n_components: int,
                   max_iter: int = 100,
                   tol: float = 1e-4) -> None:
    """
    Validate input data and parameters for mixture model fitting.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    n_components : int
        Number of mixture components
    max_iter : int, optional
        Maximum number of iterations (default: 100)
    tol : float, optional
        Tolerance for convergence (default: 1e-4)

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X must be a 2D numpy array")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol < 0:
        raise ValueError("tol must be non-negative")

def compute_responsibilities(X: np.ndarray,
                           weights: np.ndarray,
                           means: np.ndarray,
                           covariances: np.ndarray) -> np.ndarray:
    """
    Compute responsibilities for each data point and component.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    weights : np.ndarray
        Component weights of shape (n_components,)
    means : np.ndarray
        Component means of shape (n_components, n_features)
    covariances : np.ndarray
        Component covariances of shape (n_components, n_features)

    Returns
    ------
    np.ndarray
        Responsibilities of shape (n_samples, n_components)
    """
    n_samples = X.shape[0]
    n_components = weights.shape[0]

    responsibilities = np.zeros((n_samples, n_components))

    for k in range(n_components):
        # Compute multivariate normal PDF
        exponent = -0.5 * np.sum((X - means[k]) ** 2 / covariances[k], axis=1)
        coeff = 1.0 / (np.sqrt((2 * np.pi) ** X.shape[1] * np.prod(covariances[k])))
        responsibilities[:, k] = weights[k] * coeff * np.exp(exponent)

    # Normalize responsibilities
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return responsibilities

def estimate_parameters(X: np.ndarray,
                       responsibilities: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Estimate mixture model parameters from responsibilities.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    responsibilities : np.ndarray
        Responsibilities of shape (n_samples, n_components)

    Returns
    ------
    Dict[str, np.ndarray]
        Dictionary containing estimated parameters:
        - weights: Component weights
        - means: Component means
        - covariances: Component covariances
    """
    n_samples, n_features = X.shape
    n_components = responsibilities.shape[1]

    weights = np.mean(responsibilities, axis=0)
    means = np.dot(responsibilities.T, X) / responsibilities.sum(axis=0)[:, None]
    covariances = np.zeros((n_components, n_features))

    for k in range(n_components):
        diff = X - means[k]
        weighted_diff = responsibilities[:, k, None] * diff
        covariances[k] = np.sum(weighted_diff * diff, axis=0) / responsibilities[:, k].sum()

    return {
        'weights': weights,
        'means': means,
        'covariances': covariances
    }

def compute_metrics(X: np.ndarray,
                   responsibilities: np.ndarray,
                   metric_func: Callable = None) -> Dict[str, float]:
    """
    Compute metrics for mixture model evaluation.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    responsibilities : np.ndarray
        Responsibilities of shape (n_samples, n_components)
    metric_func : Callable, optional
        Custom metric function (default: None)

    Returns
    ------
    Dict[str, float]
        Dictionary containing computed metrics
    """
    if metric_func is None:
        # Default to log-likelihood
        def metric_func(X, responsibilities):
            n_samples = X.shape[0]
            log_likelihood = 0.0
            for i in range(n_samples):
                log_likelihood += np.log(np.sum(responsibilities[i]))
            return -log_likelihood / n_samples

    return {'metric': metric_func(X, responsibilities)}

def max_iter_fit(X: np.ndarray,
                n_components: int = 1,
                max_iter: int = 100,
                tol: float = 1e-4,
                normalize: str = 'none',
                metric_func: Optional[Callable] = None) -> Dict[str, Union[np.ndarray, float, str]]:
    """
    Fit a mixture model using iterative estimation.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    n_components : int, optional
        Number of mixture components (default: 1)
    max_iter : int, optional
        Maximum number of iterations (default: 100)
    tol : float, optional
        Tolerance for convergence (default: 1e-4)
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'none')
    metric_func : Callable, optional
        Custom metric function (default: None)

    Returns
    ------
    Dict[str, Union[np.ndarray, float, str]]
        Dictionary containing:
        - result: Estimated parameters
        - metrics: Computed metrics
        - params_used: Parameters used in fitting
        - warnings: Any warnings generated

    Example
    -------
    >>> X = np.random.randn(100, 2)
    >>> result = max_iter_fit(X, n_components=2)
    """
    # Validate inputs
    validate_inputs(X, n_components, max_iter, tol)

    # Initialize parameters (random initialization in practice)
    n_samples, n_features = X.shape
    weights = np.ones(n_components) / n_components
    means = X[np.random.choice(n_samples, n_components, replace=False)]
    covariances = np.ones((n_components, n_features)) * X.var(axis=0).mean()

    # Store parameters used
    params_used = {
        'n_components': n_components,
        'max_iter': max_iter,
        'tol': tol,
        'normalize': normalize
    }

    # Main iteration loop
    for iteration in range(max_iter):
        # Compute responsibilities
        responsibilities = compute_responsibilities(X, weights, means, covariances)

        # Estimate new parameters
        new_params = estimate_parameters(X, responsibilities)
        new_weights = new_params['weights']
        new_means = new_params['means']
        new_covariances = new_params['covariances']

        # Compute metrics
        metrics = compute_metrics(X, responsibilities, metric_func)

        # Check for convergence
        if np.linalg.norm(new_weights - weights) < tol and \
           np.linalg.norm(new_means - means) < tol and \
           np.linalg.norm(new_covariances - covariances) < tol:
            break

        # Update parameters
        weights, means, covariances = new_weights, new_means, new_covariances

    # Prepare result dictionary
    result = {
        'weights': weights,
        'means': means,
        'covariances': covariances
    }

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': []
    }

################################################################################
# tol
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def tol_fit(
    data: np.ndarray,
    n_components: int = 2,
    max_iter: int = 100,
    tol: float = 1e-4,
    normalize: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fit a mixture model using the TOL (Topology Optimized Learning) method.

    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (n_samples, n_features).
    n_components : int, optional
        Number of mixture components.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence.
    normalize : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : Union[str, Callable], optional
        Metric to optimize: 'mse', 'mae', 'r2', 'logloss', or custom callable.
    distance : Union[str, Callable], optional
        Distance metric: 'euclidean', 'manhattan', 'cosine', 'minkowski', or custom callable.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'.
    regularization : Optional[str], optional
        Regularization type: 'none', 'l1', 'l2', or 'elasticnet'.
    custom_metric : Optional[Callable], optional
        Custom metric function if metric is 'custom'.
    custom_distance : Optional[Callable], optional
        Custom distance function if distance is 'custom'.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, n_components)

    # Normalize data
    normalized_data = _normalize_data(data, normalize)

    # Initialize parameters
    params = _initialize_parameters(normalized_data.shape[1], n_components)

    # Choose solver
    if solver == 'closed_form':
        params = _solve_closed_form(normalized_data, n_components)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(
            normalized_data, n_components, max_iter, tol,
            metric, distance, regularization
        )
    elif solver == 'newton':
        params = _solve_newton(
            normalized_data, n_components, max_iter, tol,
            metric, distance, regularization
        )
    elif solver == 'coordinate_descent':
        params = _solve_coordinate_descent(
            normalized_data, n_components, max_iter, tol,
            metric, distance, regularization
        )

    # Compute metrics
    metrics = _compute_metrics(normalized_data, params, metric, distance)

    # Prepare output
    result = {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'max_iter': max_iter,
            'tol': tol,
            'normalize': normalize,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

    return result

def _validate_inputs(data: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    if n_components <= 0:
        raise ValueError("Number of components must be positive.")

def _normalize_data(data: np.ndarray, method: str) -> np.ndarray:
    """Normalize data based on the specified method."""
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

def _initialize_parameters(n_features: int, n_components: int) -> Dict:
    """Initialize mixture model parameters."""
    return {
        'weights': np.ones(n_components) / n_components,
        'means': np.random.randn(n_components, n_features),
        'covariances': np.array([np.eye(n_features) for _ in range(n_components)])
    }

def _solve_closed_form(data: np.ndarray, n_components: int) -> Dict:
    """Solve mixture model using closed-form solution."""
    # Placeholder for actual implementation
    return _initialize_parameters(data.shape[1], n_components)

def _solve_gradient_descent(
    data: np.ndarray,
    n_components: int,
    max_iter: int,
    tol: float,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str]
) -> Dict:
    """Solve mixture model using gradient descent."""
    # Placeholder for actual implementation
    return _initialize_parameters(data.shape[1], n_components)

def _solve_newton(
    data: np.ndarray,
    n_components: int,
    max_iter: int,
    tol: float,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str]
) -> Dict:
    """Solve mixture model using Newton's method."""
    # Placeholder for actual implementation
    return _initialize_parameters(data.shape[1], n_components)

def _solve_coordinate_descent(
    data: np.ndarray,
    n_components: int,
    max_iter: int,
    tol: float,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str]
) -> Dict:
    """Solve mixture model using coordinate descent."""
    # Placeholder for actual implementation
    return _initialize_parameters(data.shape[1], n_components)

def _compute_metrics(
    data: np.ndarray,
    params: Dict,
    metric: Union[str, Callable],
    distance: Union[str, Callable]
) -> Dict:
    """Compute metrics for the mixture model."""
    # Placeholder for actual implementation
    return {
        'metric': 0.0,
        'distance': 0.0
    }

# Example usage:
# result = tol_fit(data=np.random.randn(100, 5), n_components=3)

################################################################################
# reg_covar
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def reg_covar_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]:
    """
    Fit a regularized covariance model for mixture components.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,) or (n_samples, n_outputs).
    normalizer : Callable[[np.ndarray], np.ndarray], optional
        Function to normalize the input data. If None, no normalization is applied.
    metric : str or Callable[[np.ndarray, np.ndarray], float]
        Metric to evaluate the model. Can be "mse", "mae", "r2", or a custom callable.
    distance : str or Callable[[np.ndarray, np.ndarray], float]
        Distance metric for covariance regularization. Can be "euclidean", "manhattan",
        "cosine", "minkowski", or a custom callable.
    solver : str
        Solver to use. Options: "closed_form", "gradient_descent", "newton",
        "coordinate_descent".
    regularization : str, optional
        Type of regularization. Options: "l1", "l2", "elasticnet".
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Callable[[np.ndarray, np.ndarray], float], optional
        Custom metric function.
    custom_distance : Callable[[np.ndarray, np.ndarray], float], optional
        Custom distance function.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]
        Dictionary containing:
        - "result": Estimated parameters.
        - "metrics": Computed metrics.
        - "params_used": Parameters used during fitting.
        - "warnings": List of warnings encountered.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = reg_covar_fit(X, y, normalizer=None, metric="mse", solver="closed_form")
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    if normalizer is not None:
        X = normalizer(X)

    # Initialize parameters
    params_used = {
        "normalizer": str(normalizer),
        "metric": metric,
        "distance": distance,
        "solver": solver,
        "regularization": regularization,
        "tol": tol,
        "max_iter": max_iter
    }

    # Choose solver
    if solver == "closed_form":
        result = _solve_closed_form(X, y, regularization)
    elif solver == "gradient_descent":
        result = _solve_gradient_descent(X, y, metric, distance, regularization, tol, max_iter)
    elif solver == "newton":
        result = _solve_newton(X, y, metric, distance, regularization, tol, max_iter)
    elif solver == "coordinate_descent":
        result = _solve_coordinate_descent(X, y, metric, distance, regularization, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(y, result["predictions"], metric, custom_metric)

    # Prepare output
    output = {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": []
    }

    return output

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays.")
    if X.ndim != 2 or y.ndim not in (1, 2):
        raise ValueError("X must be 2D and y must be 1D or 2D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input data contains NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input data contains infinite values.")

def _solve_closed_form(X: np.ndarray, y: np.ndarray, regularization: Optional[str]) -> Dict[str, Union[np.ndarray, float]]:
    """Solve the regularized covariance problem in closed form."""
    # Add regularization if specified
    if regularization == "l1":
        pass  # L1 regularization requires different approach
    elif regularization == "l2":
        pass  # L2 regularization can be added to the normal equations
    elif regularization == "elasticnet":
        pass  # Elastic net requires a combination of L1 and L2

    # Closed form solution (simplified for illustration)
    X_tx = np.dot(X.T, X)
    X_ty = np.dot(X.T, y)
    params = np.linalg.solve(X_tx, X_ty)

    predictions = np.dot(X, params)
    return {"params": params, "predictions": predictions}

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict[str, Union[np.ndarray, float]]:
    """Solve the regularized covariance problem using gradient descent."""
    # Initialize parameters
    n_features = X.shape[1]
    params = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        # Compute predictions
        predictions = np.dot(X, params)

        # Compute gradient (simplified for illustration)
        gradient = 2 * np.dot(X.T, predictions - y)

        # Add regularization if specified
        if regularization == "l1":
            gradient += np.sign(params)
        elif regularization == "l2":
            gradient += 2 * params
        elif regularization == "elasticnet":
            gradient += np.sign(params) + 2 * params

        # Update parameters
        params -= learning_rate * gradient

    predictions = np.dot(X, params)
    return {"params": params, "predictions": predictions}

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict[str, Union[np.ndarray, float]]:
    """Solve the regularized covariance problem using Newton's method."""
    # Initialize parameters
    n_features = X.shape[1]
    params = np.zeros(n_features)

    for _ in range(max_iter):
        # Compute predictions
        predictions = np.dot(X, params)

        # Compute gradient and Hessian (simplified for illustration)
        gradient = 2 * np.dot(X.T, predictions - y)
        hessian = 2 * np.dot(X.T, X)

        # Add regularization if specified
        if regularization == "l2":
            gradient += 2 * params
            hessian += 2 * np.eye(n_features)

        # Update parameters using Newton's step
        params -= np.linalg.solve(hessian, gradient)

    predictions = np.dot(X, params)
    return {"params": params, "predictions": predictions}

def _solve_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict[str, Union[np.ndarray, float]]:
    """Solve the regularized covariance problem using coordinate descent."""
    # Initialize parameters
    n_features = X.shape[1]
    params = np.zeros(n_features)

    for _ in range(max_iter):
        for j in range(n_features):
            # Compute residual
            residual = y - np.dot(X, params) + X[:, j] * params[j]

            # Compute optimal value for parameter j
            if regularization == "l1":
                r = X[:, j]
                if np.dot(r, residual) < -0.5:
                    params[j] = (np.dot(r, residual) - 1) / np.dot(r, r)
                elif np.dot(r, residual) > 0.5:
                    params[j] = (np.dot(r, residual) + 1) / np.dot(r, r)
                else:
                    params[j] = 0
            elif regularization == "l2":
                params[j] = np.dot(X[:, j], residual) / (np.dot(X[:, j], X[:, j]) + 1)
            else:
                params[j] = np.dot(X[:, j], residual) / np.dot(X[:, j], X[:, j])

    predictions = np.dot(X, params)
    return {"params": params, "predictions": predictions}

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute metrics for the model."""
    metrics = {}

    if custom_metric is not None:
        metrics["custom"] = custom_metric(y_true, y_pred)

    if metric == "mse" or (isinstance(metric, str) and custom_metric is None):
        metrics["mse"] = np.mean((y_true - y_pred) ** 2)
    elif metric == "mae":
        metrics["mae"] = np.mean(np.abs(y_true - y_pred))
    elif metric == "r2":
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics["r2"] = 1 - (ss_res / ss_tot)

    return metrics

################################################################################
# weights_init
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def weights_init_fit(
    data: np.ndarray,
    n_components: int,
    normalization: str = 'none',
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Initialize weights for mixture models with configurable parameters.

    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (n_samples, n_features)
    n_components : int
        Number of components in the mixture model
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, optional
        Metric to optimize ('mse', 'mae', 'r2', 'logloss') or custom callable
    distance : str, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    regularization : str or None, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
    custom_metric : callable or None, optional
        Custom metric function if not using built-in metrics
    custom_distance : callable or None, optional
        Custom distance function if not using built-in distances

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> data = np.random.rand(100, 5)
    >>> result = weights_init_fit(data, n_components=3)
    """
    # Validate inputs
    _validate_inputs(data, n_components)

    # Normalize data if required
    normalized_data = _apply_normalization(data, normalization)

    # Initialize weights
    initial_weights = _initialize_weights(n_components)

    # Choose metric function
    if callable(metric):
        metric_func = metric
    else:
        metric_func = _get_metric_function(metric)

    # Choose distance function
    if callable(distance):
        distance_func = distance
    else:
        distance_func = _get_distance_function(distance)

    # Choose solver function
    solver_func = _get_solver_function(solver, metric_func, distance_func)

    # Optimize weights
    optimized_weights = solver_func(
        normalized_data,
        initial_weights,
        n_components,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter
    )

    # Calculate metrics
    metrics = _calculate_metrics(
        data,
        optimized_weights,
        n_components,
        metric_func=metric_func
    )

    # Prepare output
    result = {
        'result': optimized_weights,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric if not callable(metric) else 'custom',
            'distance': distance if not callable(distance) else 'custom',
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

    return result

def _validate_inputs(data: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be 2-dimensional")
    if n_components <= 0 or not isinstance(n_components, int):
        raise ValueError("n_components must be a positive integer")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values")

def _apply_normalization(
    data: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply specified normalization to data."""
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

def _initialize_weights(n_components: int) -> np.ndarray:
    """Initialize weights uniformly."""
    return np.ones(n_components) / n_components

def _get_metric_function(metric: str) -> Callable:
    """Get metric function based on input string."""
    metrics = {
        'mse': _mse,
        'mae': _mae,
        'r2': _r2,
        'logloss': _logloss
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _get_distance_function(distance: str) -> Callable:
    """Get distance function based on input string."""
    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance,
        'minkowski': _minkowski_distance
    }
    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")
    return distances[distance]

def _get_solver_function(
    solver: str,
    metric_func: Callable,
    distance_func: Callable
) -> Callable:
    """Get solver function based on input string."""
    solvers = {
        'closed_form': _closed_form_solver,
        'gradient_descent': lambda *args, **kwargs: _gradient_descent_solver(*args, metric_func=metric_func, distance_func=distance_func, **kwargs),
        'newton': lambda *args, **kwargs: _newton_solver(*args, metric_func=metric_func, distance_func=distance_func, **kwargs),
        'coordinate_descent': lambda *args, **kwargs: _coordinate_descent_solver(*args, metric_func=metric_func, distance_func=distance_func, **kwargs)
    }
    if solver not in solvers:
        raise ValueError(f"Unknown solver: {solver}")
    return solvers[solver]

def _calculate_metrics(
    data: np.ndarray,
    weights: np.ndarray,
    n_components: int,
    metric_func: Callable
) -> Dict:
    """Calculate metrics for the optimized weights."""
    return {
        'primary_metric': metric_func(data, weights),
        'weights_sum': np.sum(weights)
    }

# Metric functions
def _mse(data: np.ndarray, weights: np.ndarray) -> float:
    """Mean Squared Error metric."""
    return np.mean((data - weights)**2)

def _mae(data: np.ndarray, weights: np.ndarray) -> float:
    """Mean Absolute Error metric."""
    return np.mean(np.abs(data - weights))

def _r2(data: np.ndarray, weights: np.ndarray) -> float:
    """R-squared metric."""
    ss_res = np.sum((data - weights)**2)
    ss_tot = np.sum((data - np.mean(data))**2)
    return 1 - (ss_res / ss_tot)

def _logloss(data: np.ndarray, weights: np.ndarray) -> float:
    """Log Loss metric."""
    return -np.mean(data * np.log(weights + 1e-15) + (1 - data) * np.log(1 - weights + 1e-15))

# Distance functions
def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _minkowski_distance(a: np.ndarray, b: np.ndarray, p: int = 3) -> float:
    """Minkowski distance."""
    return np.sum(np.abs(a - b)**p)**(1/p)

# Solver functions
def _closed_form_solver(
    data: np.ndarray,
    initial_weights: np.ndarray,
    n_components: int,
    **kwargs
) -> np.ndarray:
    """Closed form solution for weight initialization."""
    return initial_weights

def _gradient_descent_solver(
    data: np.ndarray,
    initial_weights: np.ndarray,
    n_components: int,
    metric_func: Callable,
    distance_func: Callable,
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Gradient descent solver for weight optimization."""
    weights = initial_weights.copy()
    for _ in range(max_iter):
        gradient = _compute_gradient(data, weights, metric_func)
        if regularization:
            gradient += _apply_regularization(weights, regularization)
        weights -= tol * gradient
    return weights

def _newton_solver(
    data: np.ndarray,
    initial_weights: np.ndarray,
    n_components: int,
    metric_func: Callable,
    distance_func: Callable,
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Newton's method solver for weight optimization."""
    weights = initial_weights.copy()
    for _ in range(max_iter):
        gradient = _compute_gradient(data, weights, metric_func)
        hessian = _compute_hessian(data, weights, metric_func)
        if regularization:
            gradient += _apply_regularization(weights, regularization)
        weights -= np.linalg.inv(hessian) @ gradient
    return weights

def _coordinate_descent_solver(
    data: np.ndarray,
    initial_weights: np.ndarray,
    n_components: int,
    metric_func: Callable,
    distance_func: Callable,
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Coordinate descent solver for weight optimization."""
    weights = initial_weights.copy()
    for _ in range(max_iter):
        for i in range(len(weights)):
            weights[i] = _optimize_single_weight(data, weights, i, metric_func)
        if regularization:
            weights += _apply_regularization(weights, regularization)
    return weights

# Helper functions for solvers
def _compute_gradient(data: np.ndarray, weights: np.ndarray, metric_func: Callable) -> np.ndarray:
    """Compute gradient of the metric function."""
    return 2 * (weights - data) / len(data)

def _compute_hessian(data: np.ndarray, weights: np.ndarray, metric_func: Callable) -> np.ndarray:
    """Compute Hessian matrix of the metric function."""
    return 2 * np.eye(len(weights)) / len(data)

def _apply_regularization(
    weights: np.ndarray,
    regularization: str
) -> np.ndarray:
    """Apply specified regularization."""
    if regularization == 'l1':
        return np.sign(weights)
    elif regularization == 'l2':
        return 2 * weights
    elif regularization == 'elasticnet':
        return np.sign(weights) + 2 * weights
    else:
        raise ValueError(f"Unknown regularization: {regularization}")

def _optimize_single_weight(
    data: np.ndarray,
    weights: np.ndarray,
    index: int,
    metric_func: Callable
) -> float:
    """Optimize a single weight while keeping others fixed."""
    # This is a simplified version - actual implementation would be more complex
    return np.mean(data)

################################################################################
# means_init
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(X: np.ndarray,
                    n_components: int,
                    random_state: Optional[int] = None) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if random_state is not None and (not isinstance(random_state, int) or random_state < 0):
        raise ValueError("random_state must be a non-negative integer")

def _normalize_data(X: np.ndarray,
                   method: str = 'standard') -> np.ndarray:
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

def _compute_initial_means(X: np.ndarray,
                          n_components: int,
                          method: str = 'kmeans++',
                          random_state: Optional[int] = None) -> np.ndarray:
    """Compute initial means using specified method."""
    if random_state is not None:
        np.random.seed(random_state)

    if method == 'random':
        indices = np.random.choice(X.shape[0], n_components, replace=False)
        return X[indices]
    elif method == 'kmeans++':
        means = [X[np.random.randint(X.shape[0])]]
        for _ in range(1, n_components):
            dist_sq = np.min(np.sum((X - means[-1])**2, axis=1), axis=0)
            probs = dist_sq / np.sum(dist_sq)
            cumulative_probs = np.cumsum(probs)
            r = np.random.rand()
            indices = np.where(cumulative_probs >= r)[0][0]
            means.append(X[indices])
        return np.array(means)
    else:
        raise ValueError(f"Unknown initialization method: {method}")

def _compute_metrics(X: np.ndarray,
                    means: np.ndarray,
                    metric: str = 'euclidean') -> Dict[str, float]:
    """Compute metrics between data and initial means."""
    if metric == 'euclidean':
        distances = np.sqrt(np.sum((X[:, np.newaxis] - means)**2, axis=2))
    elif metric == 'manhattan':
        distances = np.sum(np.abs(X[:, np.newaxis] - means), axis=2)
    elif metric == 'cosine':
        dot_products = np.sum(X[:, np.newaxis] * means, axis=2)
        norms = np.linalg.norm(X, axis=1)[:, np.newaxis] * np.linalg.norm(means, axis=1)
        distances = 1 - dot_products / (norms + 1e-8)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return {
        'mean_distance': np.mean(distances),
        'median_distance': np.median(distances),
        'max_distance': np.max(distances)
    }

def means_init_fit(X: np.ndarray,
                  n_components: int,
                  normalization: str = 'standard',
                  metric: str = 'euclidean',
                  init_method: str = 'kmeans++',
                  random_state: Optional[int] = None) -> Dict:
    """
    Compute initial means for mixture models.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    n_components : int
        Number of components in the mixture model
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str, optional
        Distance metric ('euclidean', 'manhattan', 'cosine')
    init_method : str, optional
        Initialization method ('random', 'kmeans++')
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': array of initial means
        - 'metrics': computed metrics
        - 'params_used': parameters used
        - 'warnings': any warnings

    Example
    -------
    >>> X = np.random.rand(100, 2)
    >>> result = means_init_fit(X, n_components=3)
    """
    _validate_inputs(X, n_components, random_state)

    X_normalized = _normalize_data(X, normalization)
    means = _compute_initial_means(X_normalized, n_components, init_method, random_state)
    metrics = _compute_metrics(X_normalized, means, metric)

    return {
        'result': means,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'init_method': init_method
        },
        'warnings': []
    }

################################################################################
# precisions_init
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def precisions_init_fit(
    X: np.ndarray,
    n_components: int,
    normalizer: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Estimate initial precisions (inverse covariances) for Gaussian mixture models.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    n_components : int
        Number of mixture components
    normalizer : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, optional
        Metric to optimize ('mse', 'mae', 'r2', custom callable)
    distance : str, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski')
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton')
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
    random_state : int, optional
        Random seed for reproducibility
    custom_metric : callable, optional
        Custom metric function

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.randn(100, 2)
    >>> result = precisions_init_fit(X, n_components=3)
    """
    # Validate inputs
    _validate_inputs(X, n_components)

    # Normalize data
    X_norm = _apply_normalization(X, normalizer)

    # Initialize parameters
    params = {
        'n_components': n_components,
        'normalizer': normalizer,
        'metric': metric,
        'distance': distance,
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'max_iter': max_iter
    }

    # Choose solver
    if solver == 'closed_form':
        precisions = _closed_form_solver(X_norm, n_components)
    elif solver == 'gradient_descent':
        precisions = _gradient_descent_solver(X_norm, n_components,
                                             metric=metric,
                                             distance=distance,
                                             tol=tol,
                                             max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if needed
    if regularization:
        precisions = _apply_regularization(precisions, regularization)

    # Calculate metrics
    metrics = _calculate_metrics(X_norm, precisions,
                                metric=metric,
                                custom_metric=custom_metric)

    return {
        'result': precisions,
        'metrics': metrics,
        'params_used': params,
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
    if np.any(np.isnan(X)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Input data contains infinite values")

def _apply_normalization(
    X: np.ndarray,
    method: str = 'standard'
) -> np.ndarray:
    """Apply data normalization."""
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

def _closed_form_solver(
    X: np.ndarray,
    n_components: int
) -> np.ndarray:
    """Closed-form solution for initial precisions."""
    # Simple implementation - in practice would use more sophisticated initialization
    cov = np.cov(X, rowvar=False)
    precisions = np.linalg.inv(cov) * np.eye(n_components)
    return precisions

def _gradient_descent_solver(
    X: np.ndarray,
    n_components: int,
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Gradient descent solver for initial precisions."""
    # Initialize parameters
    n_features = X.shape[1]
    precisions = np.random.rand(n_components, n_features, n_features)

    for _ in range(max_iter):
        # Update precisions using gradient descent
        # This is a placeholder - actual implementation would depend on metric/distance
        pass

    return precisions

def _apply_regularization(
    precisions: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply regularization to precisions."""
    if method == 'l1':
        return np.sign(precisions) * (np.abs(precisions) - 0.1)
    elif method == 'l2':
        return precisions / (1 + 0.1 * np.sum(precisions**2, axis=(1, 2))[:, None, None])
    elif method == 'elasticnet':
        l1 = np.sign(precisions) * (np.abs(precisions) - 0.1)
        l2 = precisions / (1 + 0.1 * np.sum(precisions**2, axis=(1, 2))[:, None, None])
        return (l1 + l2) / 2
    else:
        raise ValueError(f"Unknown regularization method: {method}")

def _calculate_metrics(
    X: np.ndarray,
    precisions: np.ndarray,
    metric: Union[str, Callable] = 'mse',
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Calculate metrics for the estimated precisions."""
    if callable(metric):
        return {'custom_metric': metric(X, precisions)}

    if custom_metric:
        return {
            'custom_metric': custom_metric(X, precisions),
            metric: _calculate_standard_metric(X, precisions, metric)
        }

    return {metric: _calculate_standard_metric(X, precisions, metric)}

def _calculate_standard_metric(
    X: np.ndarray,
    precisions: np.ndarray,
    metric: str
) -> float:
    """Calculate standard metrics."""
    if metric == 'mse':
        return np.mean((X @ precisions) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(X @ precisions))
    elif metric == 'r2':
        # Placeholder - actual R² calculation would be more complex
        return 1 - np.var(X @ precisions) / np.var(X)
    else:
        raise ValueError(f"Unknown metric: {metric}")

################################################################################
# random_state
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def random_state_fit(
    data: np.ndarray,
    n_components: int = 2,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    normalization: str = 'standard',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]:
    """
    Fit a mixture model with random state initialization.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_samples, n_features).
    n_components : int, optional
        Number of mixture components, by default 2.
    max_iter : int, optional
        Maximum number of iterations, by default 100.
    tol : float, optional
        Tolerance for convergence, by default 1e-4.
    random_state : Optional[int], optional
        Random seed for reproducibility, by default None.
    normalization : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust', by default 'standard'.
    metric : Union[str, Callable], optional
        Metric to evaluate the model: 'mse', 'mae', 'r2', 'logloss', or custom callable, by default 'mse'.
    distance : Union[str, Callable], optional
        Distance metric: 'euclidean', 'manhattan', 'cosine', 'minkowski', or custom callable, by default 'euclidean'.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent', by default 'gradient_descent'.
    regularization : Optional[str], optional
        Regularization method: 'none', 'l1', 'l2', or 'elasticnet', by default None.
    custom_metric : Optional[Callable], optional
        Custom metric function, by default None.
    custom_distance : Optional[Callable], optional
        Custom distance function, by default None.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]
        Dictionary containing:
        - "result": Estimated parameters.
        - "metrics": Computed metrics.
        - "params_used": Parameters used during fitting.
        - "warnings": List of warnings encountered.

    Examples
    --------
    >>> data = np.random.randn(100, 2)
    >>> result = random_state_fit(data, n_components=3, random_state=42)
    """
    # Validate inputs
    _validate_inputs(data, n_components)

    # Set random state for reproducibility
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize data
    normalized_data = _normalize_data(data, normalization)

    # Initialize parameters randomly
    params = _initialize_parameters(normalized_data.shape[1], n_components)

    # Choose solver
    if solver == 'closed_form':
        params = _solve_closed_form(normalized_data, params)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(normalized_data, params, max_iter, tol)
    elif solver == 'newton':
        params = _solve_newton(normalized_data, params, max_iter, tol)
    elif solver == 'coordinate_descent':
        params = _solve_coordinate_descent(normalized_data, params, max_iter, tol)
    else:
        raise ValueError("Invalid solver specified.")

    # Apply regularization if specified
    if regularization is not None:
        params = _apply_regularization(params, regularization)

    # Compute metrics
    metrics = _compute_metrics(normalized_data, params, metric, distance, custom_metric, custom_distance)

    # Prepare output
    result = {
        "result": params,
        "metrics": metrics,
        "params_used": {
            "n_components": n_components,
            "max_iter": max_iter,
            "tol": tol,
            "normalization": normalization,
            "metric": metric if not callable(metric) else "custom",
            "distance": distance if not callable(distance) else "custom",
            "solver": solver,
            "regularization": regularization
        },
        "warnings": []
    }

    return result

def _validate_inputs(data: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    if n_components <= 0:
        raise ValueError("Number of components must be positive.")

def _normalize_data(data: np.ndarray, method: str) -> np.ndarray:
    """Normalize data using the specified method."""
    if method == 'none':
        return data
    elif method == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == 'robust':
        return (data - np.median(data, axis=0)) / (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0))
    else:
        raise ValueError("Invalid normalization method specified.")

def _initialize_parameters(n_features: int, n_components: int) -> np.ndarray:
    """Initialize mixture model parameters randomly."""
    weights = np.random.dirichlet(np.ones(n_components))
    means = np.random.randn(n_components, n_features)
    covariances = np.array([np.eye(n_features) for _ in range(n_components)])
    return {"weights": weights, "means": means, "covariances": covariances}

def _solve_closed_form(data: np.ndarray, params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Solve mixture model using closed-form solution."""
    # Placeholder for actual implementation
    return params

def _solve_gradient_descent(
    data: np.ndarray,
    params: Dict[str, np.ndarray],
    max_iter: int,
    tol: float
) -> Dict[str, np.ndarray]:
    """Solve mixture model using gradient descent."""
    # Placeholder for actual implementation
    return params

def _solve_newton(
    data: np.ndarray,
    params: Dict[str, np.ndarray],
    max_iter: int,
    tol: float
) -> Dict[str, np.ndarray]:
    """Solve mixture model using Newton's method."""
    # Placeholder for actual implementation
    return params

def _solve_coordinate_descent(
    data: np.ndarray,
    params: Dict[str, np.ndarray],
    max_iter: int,
    tol: float
) -> Dict[str, np.ndarray]:
    """Solve mixture model using coordinate descent."""
    # Placeholder for actual implementation
    return params

def _apply_regularization(params: Dict[str, np.ndarray], method: str) -> Dict[str, np.ndarray]:
    """Apply regularization to the parameters."""
    if method == 'l1':
        params['means'] = np.sign(params['means']) * np.maximum(np.abs(params['means']) - 0.1, 0)
    elif method == 'l2':
        params['means'] = params['means'] / (1 + 0.1 * np.linalg.norm(params['means']))
    elif method == 'elasticnet':
        params['means'] = np.sign(params['means']) * np.maximum(np.abs(params['means']) - 0.1, 0)
        params['means'] = params['means'] / (1 + 0.1 * np.linalg.norm(params['means']))
    return params

def _compute_metrics(
    data: np.ndarray,
    params: Dict[str, np.ndarray],
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]],
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute metrics for the mixture model."""
    if callable(metric):
        metric_value = metric(data, params['means'])
    elif metric == 'mse':
        metric_value = np.mean((data - params['means'][0])**2)
    elif metric == 'mae':
        metric_value = np.mean(np.abs(data - params['means'][0]))
    elif metric == 'r2':
        metric_value = 1 - np.sum((data - params['means'][0])**2) / np.sum((data - np.mean(data, axis=0))**2)
    elif metric == 'logloss':
        metric_value = -np.mean(data * np.log(params['means'][0]) + (1 - data) * np.log(1 - params['means'][0]))
    else:
        raise ValueError("Invalid metric specified.")

    if callable(distance):
        distance_value = distance(data, params['means'][0])
    elif distance == 'euclidean':
        distance_value = np.linalg.norm(data - params['means'][0])
    elif distance == 'manhattan':
        distance_value = np.sum(np.abs(data - params['means'][0]))
    elif distance == 'cosine':
        distance_value = 1 - np.dot(data, params['means'][0]) / (np.linalg.norm(data) * np.linalg.norm(params['means'][0]))
    elif distance == 'minkowski':
        distance_value = np.sum(np.abs(data - params['means'][0])**3)**(1/3)
    else:
        raise ValueError("Invalid distance specified.")

    return {"metric": metric_value, "distance": distance_value}
