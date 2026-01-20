"""
Quantix – Module regression_poisson
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# model_assumptions
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def model_assumptions_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    normalization: str = "none",
    metrics: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    distance: str = "euclidean",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict[str, float], Dict[str, str], list]]:
    """
    Fit model assumptions for Poisson regression.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    normalization : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    metrics : str or callable, optional
        Metric to compute ("mse", "mae", "r2", "logloss") or custom callable.
    distance : str, optional
        Distance metric ("euclidean", "manhattan", "cosine", "minkowski").
    custom_metric : callable, optional
        Custom metric function.
    custom_distance : callable, optional
        Custom distance function.

    Returns
    -------
    Dict[str, Union[Dict[str, float], Dict[str, str], list]]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Normalize data if needed
    y_true_norm, y_pred_norm = _apply_normalization(y_true, y_pred, normalization)

    # Compute metrics
    metric_results = _compute_metrics(y_true_norm, y_pred_norm, metrics, custom_metric)

    # Compute distances
    distance_results = _compute_distances(y_true_norm, y_pred_norm, distance, custom_distance)

    # Check assumptions
    assumption_checks = _check_assumptions(y_true_norm, y_pred_norm)

    # Prepare output
    result = {
        "result": {
            **metric_results,
            **distance_results,
            **assumption_checks
        },
        "metrics": {
            "normalization": normalization,
            "metric_used": metrics if isinstance(metrics, str) else "custom",
            "distance_used": distance if custom_distance is None else "custom"
        },
        "params_used": {
            "normalization": normalization,
            "metrics": metrics if isinstance(metrics, str) else "custom",
            "distance": distance
        },
        "warnings": []
    }

    return result

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Inputs must not contain NaN values.")
    if np.any(y_true < 0) or np.any(y_pred < 0):
        raise ValueError("Inputs must be non-negative for Poisson regression.")

def _apply_normalization(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalization: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply normalization to input arrays."""
    if normalization == "none":
        return y_true, y_pred
    elif normalization == "standard":
        mean = np.mean(y_true)
        std = np.std(y_true)
        y_true_norm = (y_true - mean) / std
        y_pred_norm = (y_pred - mean) / std
    elif normalization == "minmax":
        min_val = np.min(y_true)
        max_val = np.max(y_true)
        y_true_norm = (y_true - min_val) / (max_val - min_val)
        y_pred_norm = (y_pred - min_val) / (max_val - min_val)
    elif normalization == "robust":
        median = np.median(y_true)
        iqr = np.percentile(y_true, 75) - np.percentile(y_true, 25)
        y_true_norm = (y_true - median) / iqr
        y_pred_norm = (y_pred - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")
    return y_true_norm, y_pred_norm

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict[str, float]:
    """Compute metrics for model evaluation."""
    metric_results = {}

    if isinstance(metrics, str):
        if metrics == "mse":
            metric_results["mse"] = np.mean((y_true - y_pred) ** 2)
        elif metrics == "mae":
            metric_results["mae"] = np.mean(np.abs(y_true - y_pred))
        elif metrics == "r2":
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metric_results["r2"] = 1 - (ss_res / ss_tot)
        elif metrics == "logloss":
            metric_results["logloss"] = -np.mean(y_true * np.log(y_pred + 1e-10))
        else:
            raise ValueError(f"Unknown metric: {metrics}")
    elif callable(metrics):
        metric_results["custom_metric"] = metrics(y_true, y_pred)
    elif custom_metric is not None:
        metric_results["custom_metric"] = custom_metric(y_true, y_pred)
    else:
        raise ValueError("Either metrics or custom_metric must be provided.")

    return metric_results

def _compute_distances(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    distance: str,
    custom_distance: Optional[Callable]
) -> Dict[str, float]:
    """Compute distances between true and predicted values."""
    distance_results = {}

    if custom_distance is not None:
        distance_results["custom_distance"] = custom_distance(y_true, y_pred)
    else:
        if distance == "euclidean":
            distance_results["euclidean"] = np.linalg.norm(y_true - y_pred)
        elif distance == "manhattan":
            distance_results["manhattan"] = np.sum(np.abs(y_true - y_pred))
        elif distance == "cosine":
            dot_product = np.dot(y_true, y_pred)
            norm_true = np.linalg.norm(y_true)
            norm_pred = np.linalg.norm(y_pred)
            distance_results["cosine"] = 1 - (dot_product / (norm_true * norm_pred))
        elif distance == "minkowski":
            p = 3
            distance_results["minkowski"] = np.sum(np.abs(y_true - y_pred) ** p) ** (1 / p)
        else:
            raise ValueError(f"Unknown distance metric: {distance}")

    return distance_results

def _check_assumptions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, bool]:
    """Check assumptions of Poisson regression."""
    assumption_checks = {}

    # Check mean-variance relationship
    mean_true = np.mean(y_true)
    var_true = np.var(y_true)
    assumption_checks["mean_variance_relationship"] = abs(mean_true - var_true) < 0.1 * mean_true

    # Check non-negativity of predictions
    assumption_checks["non_negative_predictions"] = np.all(y_pred >= 0)

    return assumption_checks

################################################################################
# count_data
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def count_data_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    solver: str = 'closed_form',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit a Poisson regression model to count data.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,) containing count data.
    normalizer : Optional[Callable]
        Function to normalize the design matrix. If None, no normalization is applied.
    solver : str
        Solver to use: 'closed_form', 'gradient_descent', 'newton'.
    metric : Union[str, Callable]
        Metric to evaluate: 'mse', 'mae', 'r2', or custom callable.
    regularization : Optional[str]
        Regularization type: 'none', 'l1', 'l2', or 'elasticnet'.
    alpha : float
        Regularization strength.
    l1_ratio : float
        ElasticNet mixing parameter (0 <= l1_ratio <= 1).
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
        - 'result': Fitted parameters.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings encountered.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.poisson(lam=2, size=100)
    >>> result = count_data_fit(X, y, solver='gradient_descent', metric='mse')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Initialize parameters
    params_used = {
        'solver': solver,
        'metric': metric,
        'regularization': regularization,
        'alpha': alpha,
        'l1_ratio': l1_ratio,
        'max_iter': max_iter,
        'tol': tol
    }

    # Choose solver
    if solver == 'closed_form':
        result = _poisson_closed_form(X_normalized, y)
    elif solver == 'gradient_descent':
        result = _poisson_gradient_descent(
            X_normalized, y,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state
        )
    elif solver == 'newton':
        result = _poisson_newton(
            X_normalized, y,
            max_iter=max_iter,
            tol=tol
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(y, result, metric)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': []
    }

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
    if np.any(y < 0):
        raise ValueError("y must contain non-negative counts")

def _apply_normalization(
    X: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Apply normalization to the design matrix."""
    if normalizer is None:
        return X
    try:
        return normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _poisson_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form solution for Poisson regression."""
    XtX = X.T @ X
    if np.linalg.matrix_rank(XtX) < X.shape[1]:
        raise ValueError("Design matrix is not full rank")
    return np.linalg.solve(XtX, X.T @ y)

def _poisson_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None
) -> np.ndarray:
    """Gradient descent solver for Poisson regression."""
    if random_state is not None:
        np.random.seed(random_state)

    n_features = X.shape[1]
    beta = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradient = X.T @ (np.exp(X @ beta) - y)
        new_beta = beta - learning_rate * gradient

        if np.linalg.norm(new_beta - beta) < tol:
            break
        beta = new_beta

    return beta

def _poisson_newton(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> np.ndarray:
    """Newton-Raphson solver for Poisson regression."""
    n_features = X.shape[1]
    beta = np.zeros(n_features)

    for _ in range(max_iter):
        mu = np.exp(X @ beta)
        gradient = X.T @ (mu - y)
        hessian = X.T @ np.diag(mu) @ X

        try:
            delta = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break

        new_beta = beta - delta
        if np.linalg.norm(new_beta - beta) < tol:
            break
        beta = new_beta

    return beta

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    if callable(metric):
        return {'custom': metric(y_true, y_pred)}

    metrics = {}
    if metric == 'mse' or (isinstance(metric, str) and 'custom' not in metric):
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    if metric == 'mae' or (isinstance(metric, str) and 'custom' not in metric):
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    if metric == 'r2' or (isinstance(metric, str) and 'custom' not in metric):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - ss_res / ss_tot

    return metrics

################################################################################
# log_link_function
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate input arrays for log link function.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("Inputs must be numpy arrays")
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")
    if np.any(y < 0):
        raise ValueError("y must contain non-negative values for Poisson regression")

def log_link_function(X: np.ndarray, y: np.ndarray,
                     solver: str = 'newton',
                     max_iter: int = 100,
                     tol: float = 1e-4,
                     penalty: Optional[str] = None,
                     alpha: float = 1.0,
                     metric: Union[str, Callable] = 'mse',
                     normalize: bool = False,
                     custom_metric: Optional[Callable] = None) -> Dict:
    """
    Fit Poisson regression model with log link function.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    solver : str
        Solver to use ('newton', 'gradient_descent')
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for convergence
    penalty : str or None
        Penalty type ('l1', 'l2', 'elasticnet') or None
    alpha : float
        Regularization strength (0 <= alpha <= 1)
    metric : str or callable
        Metric to compute ('mse', 'mae') or custom function
    normalize : bool
        Whether to normalize features
    custom_metric : callable or None
        Custom metric function if needed

    Returns
    ------
    Dict
        Dictionary containing results, metrics, parameters used and warnings

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.poisson(2, 100)
    >>> result = log_link_function(X, y)
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize if requested
    if normalize:
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Initialize parameters
    n_features = X.shape[1]
    beta = np.zeros(n_features)

    # Solver selection
    if solver == 'newton':
        beta = _newton_solver(X, y, beta, max_iter, tol, penalty, alpha)
    elif solver == 'gradient_descent':
        beta = _gradient_descent_solver(X, y, beta, max_iter, tol, penalty, alpha)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute predictions
    mu = np.exp(X @ beta)

    # Metric computation
    if isinstance(metric, str):
        if metric == 'mse':
            m = _compute_mse(y, mu)
        elif metric == 'mae':
            m = _compute_mae(y, mu)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        m = metric(y, mu)
    else:
        raise ValueError("Metric must be string or callable")

    # Prepare results
    result = {
        'result': beta,
        'metrics': {'metric_value': m},
        'params_used': {
            'solver': solver,
            'max_iter': max_iter,
            'tol': tol,
            'penalty': penalty,
            'alpha': alpha,
            'metric': metric,
            'normalize': normalize
        },
        'warnings': []
    }

    return result

def _newton_solver(X: np.ndarray, y: np.ndarray, beta_init: np.ndarray,
                   max_iter: int, tol: float,
                   penalty: Optional[str], alpha: float) -> np.ndarray:
    """
    Newton-Raphson solver for Poisson regression.

    Parameters
    ----------
    X : np.ndarray
        Design matrix
    y : np.ndarray
        Target values
    beta_init : np.ndarray
        Initial parameters
    max_iter : int
        Maximum iterations
    tol : float
        Tolerance
    penalty : str or None
        Penalty type
    alpha : float
        Regularization strength

    Returns
    ------
    np.ndarray
        Estimated parameters
    """
    beta = beta_init.copy()
    n_samples, n_features = X.shape

    for _ in range(max_iter):
        # Compute current predictions
        mu = np.exp(X @ beta)

        # Compute gradient and Hessian
        grad = X.T @ (y - mu)
        hessian = -X.T @ np.diag(mu) @ X

        # Add penalty if needed
        if penalty == 'l2':
            grad -= 2 * alpha * beta
            hessian += 2 * alpha * np.eye(n_features)
        elif penalty == 'l1':
            grad -= alpha * np.sign(beta)
            hessian += alpha * np.diag(np.where(beta > 0, 1, -1))

        # Newton step
        delta = np.linalg.solve(hessian, grad)
        beta -= delta

        # Check convergence
        if np.linalg.norm(delta) < tol:
            break

    return beta

def _gradient_descent_solver(X: np.ndarray, y: np.ndarray, beta_init: np.ndarray,
                            max_iter: int, tol: float,
                            penalty: Optional[str], alpha: float) -> np.ndarray:
    """
    Gradient descent solver for Poisson regression.

    Parameters
    ----------
    X : np.ndarray
        Design matrix
    y : np.ndarray
        Target values
    beta_init : np.ndarray
        Initial parameters
    max_iter : int
        Maximum iterations
    tol : float
        Tolerance
    penalty : str or None
        Penalty type
    alpha : float
        Regularization strength

    Returns
    ------
    np.ndarray
        Estimated parameters
    """
    beta = beta_init.copy()
    learning_rate = 0.1

    for _ in range(max_iter):
        # Compute current predictions
        mu = np.exp(X @ beta)

        # Compute gradient
        grad = X.T @ (y - mu)
        if penalty == 'l2':
            grad -= 2 * alpha * beta
        elif penalty == 'l1':
            grad -= alpha * np.sign(beta)

        # Gradient step
        beta += learning_rate * grad

        # Check convergence
        if np.linalg.norm(grad) < tol:
            break

    return beta

def _compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute mean squared error.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values

    Returns
    ------
    float
        MSE value
    """
    return np.mean((y_true - y_pred) ** 2)

def _compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute mean absolute error.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values

    Returns
    ------
    float
        MAE value
    """
    return np.mean(np.abs(y_true - y_pred))

################################################################################
# dispersion_parameter
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def dispersion_parameter_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: str = "deviance",
    normalization: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Estimate the dispersion parameter for Poisson regression.

    Parameters
    ----------
    y_true : np.ndarray
        Array of observed counts.
    y_pred : np.ndarray
        Array of predicted means from Poisson regression.
    metric : str, optional
        Metric to use for dispersion calculation. Options: "deviance", "pearson".
        Default is "deviance".
    normalization : str, optional
        Normalization method. Options: None, "standard", "robust".
    custom_metric : Callable, optional
        Custom metric function if not using built-in options.
    **kwargs :
        Additional keyword arguments for specific metrics.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - "result": estimated dispersion parameter
        - "metrics": computed metrics
        - "params_used": parameters used in calculation
        - "warnings": any warnings generated

    Examples
    --------
    >>> y_true = np.array([10, 20, 30])
    >>> y_pred = np.array([9.5, 18.2, 31.0])
    >>> result = dispersion_parameter_fit(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Choose metric function
    if custom_metric is not None:
        metric_func = custom_metric
    else:
        metric_func = _get_metric_function(metric)

    # Compute residuals and metric
    residuals = y_true - y_pred
    metric_value = metric_func(y_true, y_pred)

    # Calculate dispersion parameter
    n_samples = len(y_true)
    if normalization == "standard":
        dispersion = metric_value / (n_samples - 1)
    elif normalization == "robust":
        dispersion = metric_value / (n_samples - 1) * (1 + 2/(n_samples - 1))
    else:
        dispersion = metric_value / n_samples

    # Prepare output
    result_dict: Dict[str, Any] = {
        "result": dispersion,
        "metrics": {"residual_sum_squares": np.sum(residuals**2)},
        "params_used": {
            "metric": metric,
            "normalization": normalization
        },
        "warnings": []
    }

    return result_dict

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if np.any(y_true < 0) or np.any(y_pred <= 0):
        raise ValueError("Counts must be non-negative and predictions positive")

def _get_metric_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the appropriate metric function."""
    if metric == "deviance":
        return _poisson_deviance
    elif metric == "pearson":
        return _pearson_chi2
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Poisson deviance."""
    return 2 * np.sum(y_true * np.log(y_true / y_pred) - (y_true - y_pred))

def _pearson_chi2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Pearson chi-squared statistic."""
    return np.sum(((y_true - y_pred) ** 2) / y_pred)

################################################################################
# overdispersion
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def overdispersion_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: Union[str, Callable] = "pearson",
    normalization: str = "standard",
    solver: str = "closed_form",
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Estimate overdispersion in Poisson regression models.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of observed counts.
    y_pred : np.ndarray
        Array of predicted counts from Poisson regression.
    metric : str or callable, optional
        Metric to use for overdispersion test. Options: "pearson", "deviance".
    normalization : str, optional
        Normalization method. Options: "standard", "none".
    solver : str, optional
        Solver to use. Options: "closed_form", "iterative".
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function if not using built-in options.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Choose metric function
    if isinstance(metric, str):
        if metric == "pearson":
            metric_func = _pearson_residuals
        elif metric == "deviance":
            metric_func = _deviance_residuals
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        metric_func = metric
    else:
        raise TypeError("Metric must be a string or callable")

    # Choose normalization
    if normalization == "standard":
        y_true, y_pred = _normalize_data(y_true, y_pred)
    elif normalization != "none":
        raise ValueError(f"Unknown normalization: {normalization}")

    # Choose solver
    if solver == "closed_form":
        result = _closed_form_solution(y_true, y_pred, metric_func)
    elif solver == "iterative":
        result = _iterative_solution(y_true, y_pred, metric_func, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(y_true, y_pred, metric_func)

    return {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "metric": metric,
            "normalization": normalization,
            "solver": solver,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or inf values")
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError("y_pred contains NaN or inf values")
    if np.any(y_true < 0) or np.any(y_pred < 0):
        raise ValueError("Counts cannot be negative")

def _pearson_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calculate Pearson residuals."""
    return (y_true - y_pred) / np.sqrt(y_pred)

def _deviance_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calculate deviance residuals."""
    return np.sign(y_true - y_pred) * np.sqrt(2 * (y_true * np.log(y_true / y_pred) - y_true + y_pred))

def _normalize_data(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Normalize the data."""
    mean = np.mean(y_true)
    std = np.std(y_true)
    if std == 0:
        return y_true, y_pred
    return (y_true - mean) / std, (y_pred - mean) / std

def _closed_form_solution(y_true: np.ndarray, y_pred: np.ndarray, metric_func: Callable) -> float:
    """Calculate overdispersion using closed form solution."""
    residuals = metric_func(y_true, y_pred)
    return np.sum(residuals**2) / (len(y_true) - 1)

def _iterative_solution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable,
    tol: float,
    max_iter: int
) -> float:
    """Calculate overdispersion using iterative solution."""
    residuals = metric_func(y_true, y_pred)
    dispersion = np.sum(residuals**2) / (len(y_true) - 1)

    for _ in range(max_iter):
        new_residuals = metric_func(y_true, y_pred * dispersion)
        new_dispersion = np.sum(new_residuals**2) / (len(y_true) - 1)
        if abs(new_dispersion - dispersion) < tol:
            break
        dispersion = new_dispersion

    return dispersion

def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, metric_func: Callable) -> Dict:
    """Calculate various metrics for overdispersion."""
    residuals = metric_func(y_true, y_pred)
    return {
        "mean_residual": np.mean(residuals),
        "std_residual": np.std(residuals),
        "residual_range": (np.min(residuals), np.max(residuals)),
        "dispersion": np.sum(residuals**2) / (len(y_true) - 1)
    }

################################################################################
# under_dispersion
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def under_dispersion_fit(
    y: np.ndarray,
    X: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = "mse",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Fit an under-dispersed Poisson regression model.

    Parameters
    ----------
    y : np.ndarray
        Target values (must be non-negative).
    X : np.ndarray
        Design matrix.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the design matrix, by default None.
    metric : str, optional
        Metric to evaluate model performance, by default "mse".
    solver : str, optional
        Solver to use for optimization, by default "closed_form".
    regularization : Optional[str], optional
        Type of regularization, by default None.
    tol : float, optional
        Tolerance for convergence, by default 1e-6.
    max_iter : int, optional
        Maximum number of iterations, by default 1000.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function, by default None.
    **kwargs : dict
        Additional solver-specific parameters.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], str]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> y = np.array([1, 2, 3])
    >>> X = np.array([[1, 0], [0, 1], [1, 1]])
    >>> result = under_dispersion_fit(y, X)
    """
    # Validate inputs
    _validate_inputs(y, X)

    # Normalize data if specified
    if normalizer is not None:
        X = normalizer(X)

    # Choose solver
    if solver == "closed_form":
        coefficients = _solve_closed_form(y, X)
    elif solver == "gradient_descent":
        coefficients = _solve_gradient_descent(y, X, tol=tol, max_iter=max_iter, **kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if specified
    if regularization is not None:
        coefficients = _apply_regularization(coefficients, X, regularization)

    # Calculate metrics
    y_pred = _predict(y, X, coefficients)
    metrics = _calculate_metrics(y, y_pred, metric=metric, custom_metric=custom_metric)

    # Prepare output
    result = {
        "result": coefficients,
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

def _validate_inputs(y: np.ndarray, X: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y, np.ndarray) or not isinstance(X, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if y.ndim != 1 or X.ndim != 2:
        raise ValueError("y must be 1D and X must be 2D.")
    if y.shape[0] != X.shape[0]:
        raise ValueError("y and X must have the same number of samples.")
    if np.any(y < 0):
        raise ValueError("Target values must be non-negative.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or inf values.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or inf values.")

def _solve_closed_form(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Solve using closed-form solution."""
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def _solve_gradient_descent(
    y: np.ndarray,
    X: np.ndarray,
    *,
    tol: float = 1e-6,
    max_iter: int = 1000,
    learning_rate: float = 0.01
) -> np.ndarray:
    """Solve using gradient descent."""
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ coefficients - y) / len(y)
        new_coefficients = coefficients - learning_rate * gradient
        if np.linalg.norm(new_coefficients - coefficients) < tol:
            break
        coefficients = new_coefficients
    return coefficients

def _apply_regularization(
    coefficients: np.ndarray,
    X: np.ndarray,
    regularization: str
) -> np.ndarray:
    """Apply regularization to coefficients."""
    if regularization == "l1":
        return np.sign(coefficients) * np.maximum(np.abs(coefficients) - 1, 0)
    elif regularization == "l2":
        return coefficients / (1 + np.linalg.norm(X.T @ X))
    elif regularization == "elasticnet":
        l1_coeffs = np.sign(coefficients) * np.maximum(np.abs(coefficients) - 1, 0)
        l2_coeffs = coefficients / (1 + np.linalg.norm(X.T @ X))
        return 0.5 * l1_coeffs + 0.5 * l2_coeffs
    else:
        raise ValueError(f"Unknown regularization: {regularization}")

def _predict(y: np.ndarray, X: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Predict using the fitted model."""
    return X @ coefficients

def _calculate_metrics(
    y: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: str = "mse",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Calculate metrics for the model."""
    metrics = {}
    if metric == "mse":
        metrics["mse"] = np.mean((y - y_pred) ** 2)
    elif metric == "mae":
        metrics["mae"] = np.mean(np.abs(y - y_pred))
    elif metric == "r2":
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics["r2"] = 1 - (ss_res / ss_tot)
    elif metric == "logloss":
        metrics["logloss"] = -np.mean(y * np.log(y_pred) - y_pred)
    if custom_metric is not None:
        metrics["custom"] = custom_metric(y, y_pred)
    return metrics

################################################################################
# maximum_likelihood_estimation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def maximum_likelihood_estimation_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    solver: str = 'newton',
    normalization: Optional[str] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'logloss',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_solver: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit a Poisson regression model using maximum likelihood estimation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    solver : str, optional
        Solver to use ('newton', 'gradient_descent'). Default is 'newton'.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax'). Default is None.
    metric : str or callable, optional
        Metric to evaluate ('mse', 'mae', 'r2', 'logloss'). Default is 'logloss'.
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2'). Default is None.
    tol : float, optional
        Tolerance for convergence. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    custom_solver : callable, optional
        Custom solver function. Default is None.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': Estimated coefficients.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used in the fitting process.
        - 'warnings': Any warnings encountered.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.poisson(lam=2, size=100)
    >>> result = maximum_likelihood_estimation_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_normalized = _apply_normalization(X, normalization)

    # Initialize parameters
    n_features = X_normalized.shape[1]
    beta_init = np.zeros(n_features)

    # Choose solver
    if custom_solver is not None:
        beta = custom_solver(X_normalized, y, tol=tol, max_iter=max_iter)
    else:
        if solver == 'newton':
            beta = _newton_raphson(X_normalized, y, beta_init, tol=tol, max_iter=max_iter)
        elif solver == 'gradient_descent':
            beta = _gradient_descent(X_normalized, y, beta_init, tol=tol, max_iter=max_iter)
        else:
            raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    y_pred = _predict(X_normalized, beta)
    metrics = _compute_metrics(y, y_pred, metric)

    # Prepare output
    result_dict = {
        'result': beta,
        'metrics': metrics,
        'params_used': {
            'solver': solver if custom_solver is None else 'custom',
            'normalization': normalization,
            'metric': metric if isinstance(metric, str) else 'custom',
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return result_dict

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
    if np.any(y < 0):
        raise ValueError("y must contain non-negative values for Poisson regression")

def _apply_normalization(X: np.ndarray, method: Optional[str]) -> np.ndarray:
    """Apply normalization to the design matrix."""
    if method is None or method == 'none':
        return X
    elif method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _newton_raphson(
    X: np.ndarray,
    y: np.ndarray,
    beta_init: np.ndarray,
    *,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> np.ndarray:
    """Newton-Raphson solver for Poisson regression."""
    beta = beta_init.copy()
    for _ in range(max_iter):
        eta = np.exp(X @ beta)
        gradient = X.T @ (y - eta)
        hessian = -X.T @ np.diag(eta) @ X
        delta = np.linalg.solve(hessian, gradient)
        beta -= delta
        if np.linalg.norm(delta) < tol:
            break
    return beta

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    beta_init: np.ndarray,
    *,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> np.ndarray:
    """Gradient descent solver for Poisson regression."""
    beta = beta_init.copy()
    learning_rate = 0.01
    for _ in range(max_iter):
        eta = np.exp(X @ beta)
        gradient = X.T @ (y - eta)
        beta += learning_rate * gradient
        if np.linalg.norm(gradient) < tol:
            break
    return beta

def _predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Predict using the Poisson regression model."""
    return np.exp(X @ beta)

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """Compute metrics for the Poisson regression model."""
    metrics_dict = {}
    if isinstance(metric, str):
        if metric == 'mse':
            metrics_dict['mse'] = np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            metrics_dict['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics_dict['r2'] = 1 - (ss_res / ss_tot)
        elif metric == 'logloss':
            metrics_dict['logloss'] = -np.mean(y_true * np.log(y_pred + 1e-8) - y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        metrics_dict['custom'] = metric(y_true, y_pred)
    return metrics_dict

################################################################################
# deviance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def deviance_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    normalization: str = "none",
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "standard",
    weights: Optional[np.ndarray] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Calculate the deviance for Poisson regression.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of observed counts.
    y_pred : np.ndarray
        Array of predicted counts (fitted values).
    normalization : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    metric : str or callable, optional
        Metric to use ("standard" for Poisson deviance, custom callable).
    weights : np.ndarray, optional
        Array of observation weights.
    custom_metric : callable, optional
        Custom metric function.

    Returns:
    --------
    dict
        Dictionary containing the deviance result, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred, weights)

    # Normalize data if required
    y_true_norm, y_pred_norm = _apply_normalization(y_true, y_pred, normalization)

    # Calculate deviance
    result = _calculate_deviance(y_true_norm, y_pred_norm, weights=weights)

    # Calculate metrics
    metrics = _calculate_metrics(y_true_norm, y_pred_norm, metric=metric, custom_metric=custom_metric)

    # Prepare output
    params_used = {
        "normalization": normalization,
        "metric": metric if isinstance(metric, str) else "custom",
    }

    warnings = _check_warnings(y_true_norm, y_pred_norm)

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> None:
    """
    Validate input arrays.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of observed counts.
    y_pred : np.ndarray
        Array of predicted counts (fitted values).
    weights : np.ndarray, optional
        Array of observation weights.

    Raises:
    -------
    ValueError
        If inputs are invalid.
    """
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-dimensional arrays.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if weights is not None:
        if len(weights) != len(y_true):
            raise ValueError("weights must have the same length as y_true.")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")
    if np.any(y_true < 0) or np.any(y_pred < 0):
        raise ValueError("Counts must be non-negative.")

def _apply_normalization(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalization: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply normalization to input arrays.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of observed counts.
    y_pred : np.ndarray
        Array of predicted counts (fitted values).
    normalization : str
        Normalization method ("none", "standard", "minmax", "robust").

    Returns:
    --------
    tuple
        Normalized y_true and y_pred arrays.
    """
    if normalization == "none":
        return y_true, y_pred
    elif normalization == "standard":
        mean = np.mean(y_true)
        std = np.std(y_true)
        y_true_norm = (y_true - mean) / std
        y_pred_norm = (y_pred - mean) / std
    elif normalization == "minmax":
        min_val = np.min(y_true)
        max_val = np.max(y_true)
        y_true_norm = (y_true - min_val) / (max_val - min_val)
        y_pred_norm = (y_pred - min_val) / (max_val - min_val)
    elif normalization == "robust":
        median = np.median(y_true)
        iqr = np.percentile(y_true, 75) - np.percentile(y_true, 25)
        y_true_norm = (y_true - median) / iqr
        y_pred_norm = (y_pred - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")
    return y_true_norm, y_pred_norm

def _calculate_deviance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None
) -> float:
    """
    Calculate the Poisson deviance.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of observed counts.
    y_pred : np.ndarray
        Array of predicted counts (fitted values).
    weights : np.ndarray, optional
        Array of observation weights.

    Returns:
    --------
    float
        The Poisson deviance.
    """
    if weights is None:
        weights = np.ones_like(y_true)
    deviance = 2 * np.sum(weights * (y_true * np.log(y_true / y_pred) - (y_true - y_pred)))
    return deviance

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "standard",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """
    Calculate additional metrics.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of observed counts.
    y_pred : np.ndarray
        Array of predicted counts (fitted values).
    metric : str or callable, optional
        Metric to use ("standard" for Poisson deviance, custom callable).
    custom_metric : callable, optional
        Custom metric function.

    Returns:
    --------
    dict
        Dictionary of calculated metrics.
    """
    metrics = {}
    if metric == "standard" or isinstance(metric, str):
        pass  # Deviance is already calculated
    elif callable(metric):
        metrics["custom_metric"] = metric(y_true, y_pred)
    if custom_metric is not None:
        metrics["custom_metric"] = custom_metric(y_true, y_pred)
    return metrics

def _check_warnings(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, str]:
    """
    Check for potential warnings.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of observed counts.
    y_pred : np.ndarray
        Array of predicted counts (fitted values).

    Returns:
    --------
    dict
        Dictionary of warnings.
    """
    warnings = {}
    if np.any(y_pred == 0):
        warnings["zero_predictions"] = "Some predicted values are zero, which may cause numerical issues."
    if np.any(y_true == 0):
        warnings["zero_observations"] = "Some observed values are zero, which may affect deviance calculation."
    return warnings

# Example usage:
"""
y_true = np.array([10, 20, 30])
y_pred = np.array([8, 22, 29])
result = deviance_fit(y_true, y_pred)
print(result)
"""

################################################################################
# AIC
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for Poisson regression."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")
    if np.any(y < 0):
        raise ValueError("y must contain non-negative values for Poisson regression")

def compute_aic(log_likelihood: float, k: int) -> float:
    """Compute AIC (Akaike Information Criterion)."""
    return 2 * k - 2 * log_likelihood

def poisson_log_likelihood(y: np.ndarray, mu: np.ndarray) -> float:
    """Compute Poisson log-likelihood."""
    return np.sum(y * np.log(mu) - mu - np.linalg.norm(y))

def AIC_fit(X: np.ndarray,
            y: np.ndarray,
            log_likelihood_func: Callable[[np.ndarray, np.ndarray], float] = poisson_log_likelihood,
            penalty: str = 'aic',
            **kwargs) -> Dict[str, Union[float, Dict]]:
    """
    Compute AIC for Poisson regression.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    log_likelihood_func : callable
        Function to compute log-likelihood (default: Poisson)
    penalty : str
        Type of information criterion ('aic' or 'bic')
    **kwargs :
        Additional parameters passed to the model fitting function

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': computed AIC value
        - 'metrics': dictionary of additional metrics
        - 'params_used': parameters used in computation
        - 'warnings': any warnings encountered

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.poisson(lam=2, size=100)
    >>> result = AIC_fit(X, y)
    """
    validate_inputs(X, y)

    # Default parameters
    params_used = {
        'penalty': penalty,
        'log_likelihood_func': log_likelihood_func.__name__ if callable(log_likelihood_func) else str(log_likelihood_func),
        'additional_params': kwargs
    }

    warnings = []

    # Compute log-likelihood (in practice, this would come from model fitting)
    mu = np.exp(X @ kwargs.get('coefficients', np.zeros(X.shape[1])))
    log_lik = log_likelihood_func(y, mu)

    # Number of parameters (including intercept if present)
    k = X.shape[1] + 1

    # Compute AIC or BIC
    if penalty.lower() == 'aic':
        aic = compute_aic(log_lik, k)
    elif penalty.lower() == 'bic':
        n = X.shape[0]
        aic = compute_aic(log_lik, k) + (np.log(n) * (k - 1))
    else:
        raise ValueError("penalty must be either 'aic' or 'bic'")

    metrics = {
        'log_likelihood': log_lik,
        'n_parameters': k
    }

    return {
        'result': aic,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

################################################################################
# BIC
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int
) -> None:
    """
    Validate the inputs for BIC computation.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    k : int
        Number of parameters in the model.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-dimensional arrays.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or inf values.")
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError("y_pred contains NaN or inf values.")

def compute_log_likelihood_poisson(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute the log-likelihood for Poisson regression.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    ------
    float
        Log-likelihood.
    """
    return np.sum(y_true * np.log(y_pred) - y_pred - np.linalg.norm(y_true))

def BIC_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int,
    log_likelihood_func: Callable = compute_log_likelihood_poisson
) -> Dict[str, Union[float, Dict]]:
    """
    Compute the Bayesian Information Criterion (BIC) for Poisson regression.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    k : int
        Number of parameters in the model.
    log_likelihood_func : Callable, optional
        Function to compute the log-likelihood. Default is Poisson regression.

    Returns
    ------
    Dict[str, Union[float, Dict]]
        Dictionary containing the BIC result and metrics.
    """
    validate_inputs(y_true, y_pred, k)

    log_likelihood = log_likelihood_func(y_true, y_pred)
    n_samples = len(y_true)

    bic = -2 * log_likelihood + k * np.log(n_samples)

    return {
        "result": bic,
        "metrics": {"log_likelihood": log_likelihood},
        "params_used": {
            "n_samples": n_samples,
            "k": k
        },
        "warnings": []
    }

################################################################################
# offset_variable
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    offset: Optional[np.ndarray] = None
) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if offset is not None:
        if not isinstance(offset, np.ndarray):
            raise TypeError("offset must be a numpy array")
        if offset.ndim != 1:
            raise ValueError("offset must be a 1D array")
        if X.shape[0] != offset.shape[0]:
            raise ValueError("X and offset must have the same number of samples")

def _compute_offset(
    X: np.ndarray,
    y: np.ndarray,
    offset: Optional[np.ndarray] = None,
    offset_func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Compute the offset term for Poisson regression."""
    if offset is not None:
        return offset
    if offset_func is not None:
        return offset_func(X, y)
    return np.zeros_like(y)

def _poisson_loss(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    offset: np.ndarray
) -> float:
    """Compute the Poisson deviance loss."""
    linear_predictor = np.dot(X, beta) + offset
    return np.sum(np.exp(linear_predictor)) - np.dot(y, linear_predictor)

def _gradient_poisson(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    offset: np.ndarray
) -> np.ndarray:
    """Compute the gradient of the Poisson deviance loss."""
    linear_predictor = np.dot(X, beta) + offset
    return -np.dot(X.T, y - np.exp(linear_predictor))

def _hessian_poisson(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    offset: np.ndarray
) -> np.ndarray:
    """Compute the Hessian of the Poisson deviance loss."""
    linear_predictor = np.dot(X, beta) + offset
    return np.dot(X.T, np.diag(np.exp(linear_predictor)) * X)

def _newton_raphson(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6
) -> np.ndarray:
    """Newton-Raphson optimization for Poisson regression."""
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        grad = _gradient_poisson(X, y, beta, offset)
        hess = _hessian_poisson(X, y, beta, offset)
        delta = np.linalg.solve(hess, grad)
        beta -= delta
        if np.linalg.norm(delta) < tol:
            break
    return beta

def offset_variable_fit(
    X: np.ndarray,
    y: np.ndarray,
    offset: Optional[np.ndarray] = None,
    offset_func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    solver: str = "newton",
    max_iter: int = 100,
    tol: float = 1e-6,
    metric_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Fit Poisson regression with optional offset variable.

    Parameters:
    -----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    offset : Optional[np.ndarray]
        Offset term of shape (n_samples,)
    offset_func : Optional[Callable]
        Function to compute offset from X and y
    solver : str
        Optimization algorithm ("newton")
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for convergence
    metric_func : Optional[Callable]
        Custom metric function

    Returns:
    --------
    Dict containing:
    - "result": fitted coefficients
    - "metrics": computed metrics
    - "params_used": parameters used
    - "warnings": any warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.poisson(5, 100)
    >>> result = offset_variable_fit(X, y)
    """
    _validate_inputs(X, y, offset)

    # Compute offset
    offset = _compute_offset(X, y, offset, offset_func)

    # Solve for coefficients
    if solver == "newton":
        beta = _newton_raphson(X, y, offset, max_iter, tol)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = {}
    if metric_func is not None:
        metrics["custom"] = metric_func(y, np.dot(X, beta) + offset)
    else:
        metrics["deviance"] = _poisson_loss(X, y, beta, offset)

    return {
        "result": beta,
        "metrics": metrics,
        "params_used": {
            "solver": solver,
            "max_iter": max_iter,
            "tol": tol
        },
        "warnings": []
    }

################################################################################
# zero_inflation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def zero_inflation_fit(
    y: np.ndarray,
    X: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = "logloss",
    solver: str = "coordinate_descent",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    verbose: bool = False
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit a zero-inflated Poisson regression model.

    Parameters
    ----------
    y : np.ndarray
        Target variable (1D array).
    X : np.ndarray
        Feature matrix (2D array).
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize features, by default None.
    metric : str, optional
        Metric to evaluate model performance, by default "logloss".
    solver : str, optional
        Solver to use for optimization, by default "coordinate_descent".
    regularization : Optional[str], optional
        Type of regularization, by default None.
    tol : float, optional
        Tolerance for convergence, by default 1e-4.
    max_iter : int, optional
        Maximum number of iterations, by default 1000.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function, by default None.
    verbose : bool, optional
        Whether to print progress, by default False.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> y = np.array([0, 1, 2, 0, 3])
    >>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    >>> result = zero_inflation_fit(y, X)
    """
    # Validate inputs
    _validate_inputs(y, X)

    # Normalize features if specified
    if normalizer is not None:
        X = normalizer(X)

    # Initialize parameters
    params = _initialize_parameters(X.shape[1])

    # Choose solver and fit model
    if solver == "coordinate_descent":
        params = _coordinate_descent(y, X, params, tol=tol, max_iter=max_iter)
    elif solver == "gradient_descent":
        params = _gradient_descent(y, X, params, tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if specified
    if regularization is not None:
        params = _apply_regularization(params, X, regularization)

    # Calculate metrics
    metrics = _calculate_metrics(y, X, params, metric=metric, custom_metric=custom_metric)

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

def _validate_inputs(y: np.ndarray, X: np.ndarray) -> None:
    """Validate input arrays."""
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if len(y) != X.shape[0]:
        raise ValueError("y and X must have the same number of samples.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or inf values.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or inf values.")

def _initialize_parameters(n_features: int) -> np.ndarray:
    """Initialize model parameters."""
    return np.zeros(2 * n_features + 1)  # Poisson and zero-inflation parameters

def _coordinate_descent(
    y: np.ndarray,
    X: np.ndarray,
    params: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Coordinate descent solver for zero-inflated Poisson regression."""
    # Implementation of coordinate descent
    return params

def _gradient_descent(
    y: np.ndarray,
    X: np.ndarray,
    params: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Gradient descent solver for zero-inflated Poisson regression."""
    # Implementation of gradient descent
    return params

def _apply_regularization(
    params: np.ndarray,
    X: np.ndarray,
    regularization: str
) -> np.ndarray:
    """Apply regularization to model parameters."""
    # Implementation of regularization
    return params

def _calculate_metrics(
    y: np.ndarray,
    X: np.ndarray,
    params: np.ndarray,
    *,
    metric: str = "logloss",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Calculate model metrics."""
    metrics = {}
    if custom_metric is not None:
        metrics["custom"] = custom_metric(y, X @ params[:X.shape[1]])

    if metric == "logloss":
        metrics["logloss"] = _compute_logloss(y, X @ params[:X.shape[1]])
    elif metric == "mse":
        metrics["mse"] = _compute_mse(y, X @ params[:X.shape[1]])
    elif metric == "mae":
        metrics["mae"] = _compute_mae(y, X @ params[:X.shape[1]])
    elif metric == "r2":
        metrics["r2"] = _compute_r2(y, X @ params[:X.shape[1]])

    return metrics

def _compute_logloss(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute log loss."""
    return -np.mean(y * np.log(y_pred) - y_pred)

def _compute_mse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error."""
    return np.mean((y - y_pred) ** 2)

def _compute_mae(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""
    return np.mean(np.abs(y - y_pred))

def _compute_r2(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared."""
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

################################################################################
# hurdle_model
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def hurdle_model_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    zero_threshold: float = 1e-8,
    selection_model: str = "logistic",
    count_model: str = "poisson",
    selection_solver: Callable = None,
    count_solver: Callable = None,
    selection_metric: str = "logloss",
    count_metric: str = "poisson_dev",
    selection_normalization: Optional[str] = None,
    count_normalization: Optional[str] = None,
    selection_penalty: str = "none",
    count_penalty: str = "none",
    selection_tol: float = 1e-4,
    count_tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict:
    """
    Fit a hurdle model to the given data.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    zero_threshold : float, default=1e-8
        Threshold to consider a value as zero
    selection_model : str, default="logistic"
        Model for the zero hurdle part ("logistic" or "probit")
    count_model : str, default="poisson"
        Model for the count part ("poisson" or "negative_binomial")
    selection_solver : Callable, optional
        Custom solver for the zero hurdle part
    count_solver : Callable, optional
        Custom solver for the count part
    selection_metric : str, default="logloss"
        Metric for the zero hurdle part ("logloss" or "auc")
    count_metric : str, default="poisson_dev"
        Metric for the count part ("poisson_dev" or "mse")
    selection_normalization : str, optional
        Normalization for the zero hurdle part (None, "standard", "minmax")
    count_normalization : str, optional
        Normalization for the count part (None, "standard", "minmax")
    selection_penalty : str, default="none"
        Penalty for the zero hurdle part ("none", "l1", "l2")
    count_penalty : str, default="none"
        Penalty for the count part ("none", "l1", "l2")
    selection_tol : float, default=1e-4
        Tolerance for the zero hurdle solver
    count_tol : float, default=1e-4
        Tolerance for the count solver
    max_iter : int, default=1000
        Maximum number of iterations
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    Dict containing:
        - "result": Fitted model parameters
        - "metrics": Computed metrics for both parts
        - "params_used": Parameters used during fitting
        - "warnings": Any warnings encountered

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.poisson(2, 100)
    >>> result = hurdle_model_fit(X, y)
    """
    # Input validation
    _validate_inputs(X, y)

    # Initialize random state if provided
    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    # Split data based on zero threshold
    y_zero = (y <= zero_threshold).astype(int)
    y_count = y.copy()
    y_count[y_zero == 1] = 0

    # Normalize features if requested
    X_selection, selection_scaler = _normalize(X, normalization=selection_normalization)
    X_count, count_scaler = _normalize(X, normalization=count_normalization)

    # Fit selection model
    selection_model_params = _fit_selection_model(
        X_selection, y_zero,
        model_type=selection_model,
        solver=selection_solver,
        metric=selection_metric,
        penalty=selection_penalty,
        tol=selection_tol,
        max_iter=max_iter,
        random_state=rng
    )

    # Fit count model only on non-zero observations
    if np.sum(y_zero) < len(y):
        count_model_params = _fit_count_model(
            X_count[y_zero == 0],
            y_count[y_zero == 0],
            model_type=count_model,
            solver=count_solver,
            metric=count_metric,
            penalty=count_penalty,
            tol=count_tol,
            max_iter=max_iter,
            random_state=rng
        )
    else:
        count_model_params = None

    # Compute metrics for both models
    selection_metrics = _compute_selection_metrics(
        X_selection, y_zero,
        params=selection_model_params,
        metric=selection_metric
    )

    if count_model_params is not None:
        count_metrics = _compute_count_metrics(
            X_count[y_zero == 0],
            y_count[y_zero == 0],
            params=count_model_params,
            metric=count_metric
        )
    else:
        count_metrics = None

    # Prepare output dictionary
    result = {
        "result": {
            "selection_model_params": selection_model_params,
            "count_model_params": count_model_params,
            "zero_threshold": zero_threshold
        },
        "metrics": {
            "selection_metrics": selection_metrics,
            "count_metrics": count_metrics
        },
        "params_used": {
            "selection_model": selection_model,
            "count_model": count_model,
            "selection_solver": selection_solver.__name__ if selection_solver else None,
            "count_solver": count_solver.__name__ if count_solver else None,
            "selection_metric": selection_metric,
            "count_metric": count_metric,
            "selection_normalization": selection_normalization,
            "count_normalization": count_normalization,
            "selection_penalty": selection_penalty,
            "count_penalty": count_penalty
        },
        "warnings": []
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

def _normalize(X: np.ndarray, normalization: Optional[str] = None) -> tuple:
    """Normalize features."""
    if normalization is None:
        return X, None

    scaler = None
    if normalization == "standard":
        scaler = StandardScaler()
    elif normalization == "minmax":
        scaler = MinMaxScaler()

    if scaler is not None:
        X_normalized = scaler.fit_transform(X)
        return X_normalized, scaler
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _fit_selection_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_type: str = "logistic",
    solver: Optional[Callable] = None,
    metric: str = "logloss",
    penalty: str = "none",
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: np.random.RandomState
) -> Dict:
    """Fit the selection model."""
    if solver is None:
        if model_type == "logistic":
            solver = _default_logistic_solver
        elif model_type == "probit":
            solver = _default_probit_solver
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    params = solver(
        X, y,
        metric=metric,
        penalty=penalty,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state
    )

    return params

def _fit_count_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_type: str = "poisson",
    solver: Optional[Callable] = None,
    metric: str = "poisson_dev",
    penalty: str = "none",
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: np.random.RandomState
) -> Dict:
    """Fit the count model."""
    if solver is None:
        if model_type == "poisson":
            solver = _default_poisson_solver
        elif model_type == "negative_binomial":
            solver = _default_negbin_solver
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    params = solver(
        X, y,
        metric=metric,
        penalty=penalty,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state
    )

    return params

def _compute_selection_metrics(
    X: np.ndarray,
    y: np.ndarray,
    *,
    params: Dict,
    metric: str = "logloss"
) -> Dict:
    """Compute metrics for the selection model."""
    if metric == "logloss":
        return {"logloss": _compute_log_loss(X, y, params)}
    elif metric == "auc":
        return {"auc": _compute_roc_auc(X, y, params)}
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _compute_count_metrics(
    X: np.ndarray,
    y: np.ndarray,
    *,
    params: Dict,
    metric: str = "poisson_dev"
) -> Dict:
    """Compute metrics for the count model."""
    if metric == "poisson_dev":
        return {"poisson_deviance": _compute_poisson_deviance(X, y, params)}
    elif metric == "mse":
        return {"mse": _compute_mse(X, y, params)}
    else:
        raise ValueError(f"Unknown metric: {metric}")

# Default solvers (would be implemented elsewhere in the library)
def _default_logistic_solver(*args, **kwargs):
    pass

def _default_probit_solver(*args, **kwargs):
    pass

def _default_poisson_solver(*args, **kwargs):
    pass

def _default_negbin_solver(*args, **kwargs):
    pass

# Default metric functions (would be implemented elsewhere in the library)
def _compute_log_loss(*args, **kwargs):
    pass

def _compute_roc_auc(*args, **kwargs):
    pass

def _compute_poisson_deviance(*args, **kwargs):
    pass

def _compute_mse(*args, **kwargs):
    pass

# Normalization classes (would be implemented elsewhere in the library)
class StandardScaler:
    def fit_transform(self, X):
        pass

class MinMaxScaler:
    def fit_transform(self, X):
        pass

################################################################################
# quasi_poisson
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def normalize_data(X: np.ndarray, y: np.ndarray,
                  normalization: str = 'standard',
                  custom_normalize: Optional[Callable] = None) -> tuple:
    """Normalize data according to specified method."""
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

def compute_quasi_likelihood(y: np.ndarray,
                           mu: np.ndarray,
                           phi: float = 1.0) -> float:
    """Compute the quasi-likelihood for Poisson regression."""
    if np.any(mu <= 0):
        raise ValueError("Predicted values must be positive")
    if np.any(y < 0):
        raise ValueError("Observed values must be non-negative")
    return np.sum(y * np.log(mu) - mu - y * np.log(y + 1e-10)) / phi

def fit_quasi_poisson(X: np.ndarray,
                     y: np.ndarray,
                     solver: str = 'newton',
                     max_iter: int = 100,
                     tol: float = 1e-6,
                     alpha: float = 0.0,
                     l1_ratio: float = 0.5,
                     custom_solver: Optional[Callable] = None) -> np.ndarray:
    """Fit quasi-Poisson regression model."""
    validate_inputs(X, y)
    X_norm, y_norm = normalize_data(X, y)

    if custom_solver is not None:
        return custom_solver(X_norm, y_norm)

    n_samples, n_features = X_norm.shape
    beta = np.zeros(n_features)
    phi = 1.0

    for _ in range(max_iter):
        mu = np.exp(X_norm @ beta)
        W = np.diag(mu * phi)

        if solver == 'newton':
            XtWX = X_norm.T @ W @ X_norm
            XtWy = X_norm.T @ W @ y_norm
            delta = np.linalg.solve(XtWX, XtWy - X_norm.T @ (mu + alpha * np.sign(beta) * l1_ratio))
            beta -= delta

        elif solver == 'gradient_descent':
            gradient = X_norm.T @ (mu - y_norm) / phi
            beta -= 0.01 * gradient

        if np.linalg.norm(delta) < tol:
            break

    return beta

def compute_metrics(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   metrics: list = ['mse', 'mae'],
                   custom_metric: Optional[Callable] = None) -> Dict[str, float]:
    """Compute specified metrics."""
    results = {}
    if 'mse' in metrics:
        results['mse'] = np.mean((y_true - y_pred) ** 2)
    if 'mae' in metrics:
        results['mae'] = np.mean(np.abs(y_true - y_pred))
    if 'r2' in metrics:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        results['r2'] = 1 - ss_res / ss_tot
    if custom_metric is not None:
        results['custom'] = custom_metric(y_true, y_pred)
    return results

def quasi_poisson_fit(X: np.ndarray,
                     y: np.ndarray,
                     normalization: str = 'standard',
                     solver: str = 'newton',
                     metrics: list = ['mse', 'mae'],
                     max_iter: int = 100,
                     tol: float = 1e-6,
                     alpha: float = 0.0,
                     l1_ratio: float = 0.5,
                     custom_normalize: Optional[Callable] = None,
                     custom_solver: Optional[Callable] = None,
                     custom_metric: Optional[Callable] = None) -> Dict:
    """Fit quasi-Poisson regression model with specified parameters."""
    beta = fit_quasi_poisson(X, y,
                           solver=solver,
                           max_iter=max_iter,
                           tol=tol,
                           alpha=alpha,
                           l1_ratio=l1_ratio,
                           custom_solver=custom_solver)

    mu = np.exp(X @ beta)
    results = {
        'result': {
            'coefficients': beta,
            'predictions': mu
        },
        'metrics': compute_metrics(y, mu, metrics=metrics, custom_metric=custom_metric),
        'params_used': {
            'normalization': normalization,
            'solver': solver if custom_solver is None else 'custom',
            'max_iter': max_iter,
            'tol': tol,
            'alpha': alpha,
            'l1_ratio': l1_ratio
        },
        'warnings': []
    }

    if np.any(mu <= 0):
        results['warnings'].append("Some predicted values are non-positive")

    return results

# Example usage:
"""
X = np.random.rand(100, 5)
y = np.random.poisson(3, 100)

results = quasi_poisson_fit(X, y,
                          normalization='standard',
                          solver='newton',
                          metrics=['mse', 'r2'])
"""

################################################################################
# negative_binomial_regression
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def negative_binomial_regression_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = 'mse',
    solver: str = 'gradient_descent',
    alpha: float = 0.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit a Negative Binomial Regression model.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    normalizer : Optional[Callable], default=None
        Function to normalize features. If None, no normalization is applied.
    metric : Union[str, Callable], default='mse'
        Metric to evaluate the model. Can be 'mse', 'mae', 'r2', or a custom callable.
    solver : str, default='gradient_descent'
        Solver to use. Options: 'gradient_descent', 'newton'.
    alpha : float, default=0.0
        Regularization strength.
    l1_ratio : float, default=0.5
        ElasticNet mixing parameter (0 = L2, 1 = L1).
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for stopping criteria.
    random_state : Optional[int], default=None
        Random seed for reproducibility.
    verbose : bool, default=False
        Whether to print progress information.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': Estimated parameters.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings encountered.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.poisson(2, 100)
    >>> result = negative_binomial_regression_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Initialize parameters
    n_features = X_normalized.shape[1]
    params = np.zeros(n_features + 1)  # Intercept included

    # Choose solver
    if solver == 'gradient_descent':
        params = _gradient_descent_solver(
            X_normalized, y, params,
            alpha=alpha, l1_ratio=l1_ratio,
            max_iter=max_iter, tol=tol,
            random_state=random_state,
            verbose=verbose
        )
    elif solver == 'newton':
        params = _newton_solver(
            X_normalized, y, params,
            max_iter=max_iter, tol=tol
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_normalized, y, params, metric)

    # Prepare output
    result = {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'solver': solver,
            'alpha': alpha,
            'l1_ratio': l1_ratio,
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
        raise ValueError("X contains NaN or Inf values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or Inf values")
    if np.any(y < 0):
        raise ValueError("y must contain non-negative values")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to features."""
    if normalizer is None:
        return X
    try:
        return normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    *,
    alpha: float = 0.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> np.ndarray:
    """Gradient descent solver for Negative Binomial Regression."""
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_features = X.shape
    intercept = params[-1]
    beta = params[:-1]

    for _ in range(max_iter):
        # Compute predictions
        mu = np.exp(X @ beta + intercept)
        theta = 1.0 / (mu - y) if np.any(mu != y) else np.ones_like(y)

        # Compute gradients
        grad_intercept = np.sum((mu - y) / (1 + theta * mu))
        grad_beta = X.T @ ((mu - y) / (1 + theta * mu))

        # Add regularization
        if alpha > 0:
            l1_penalty = l1_ratio * np.sign(beta)
            l2_penalty = (1 - l1_ratio) * beta
            grad_beta += alpha * (l1_penalty + l2_penalty)

        # Update parameters
        old_params = np.concatenate([beta, [intercept]])
        beta -= tol * grad_beta
        intercept -= tol * grad_intercept

        # Check convergence
        new_params = np.concatenate([beta, [intercept]])
        if np.linalg.norm(new_params - old_params) < tol:
            break

    return new_params

def _newton_solver(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    *,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> np.ndarray:
    """Newton's method solver for Negative Binomial Regression."""
    n_samples, n_features = X.shape
    intercept = params[-1]
    beta = params[:-1]

    for _ in range(max_iter):
        # Compute predictions
        mu = np.exp(X @ beta + intercept)
        theta = 1.0 / (mu - y) if np.any(mu != y) else np.ones_like(y)

        # Compute gradients
        grad_intercept = np.sum((mu - y) / (1 + theta * mu))
        grad_beta = X.T @ ((mu - y) / (1 + theta * mu))

        # Compute Hessian
        hess_intercept = np.sum(mu / (1 + theta * mu) ** 2)
        hess_beta = X.T @ np.diag(mu / (1 + theta * mu) ** 2) @ X

        # Update parameters
        old_params = np.concatenate([beta, [intercept]])
        delta = np.linalg.solve(np.vstack([np.hstack([hess_beta, X.T]), np.hstack([X, [[0]]])]),
                               np.concatenate([-grad_beta, [-grad_intercept]]))
        beta += delta[:-1]
        intercept += delta[-1]

        # Check convergence
        new_params = np.concatenate([beta, [intercept]])
        if np.linalg.norm(new_params - old_params) < tol:
            break

    return new_params

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """Compute metrics for the model."""
    beta = params[:-1]
    intercept = params[-1]

    # Compute predictions
    mu = np.exp(X @ beta + intercept)

    metrics_dict = {}

    if isinstance(metric, str):
        if metric == 'mse':
            metrics_dict['mse'] = np.mean((y - mu) ** 2)
        elif metric == 'mae':
            metrics_dict['mae'] = np.mean(np.abs(y - mu))
        elif metric == 'r2':
            ss_res = np.sum((y - mu) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            metrics_dict['r2'] = 1 - ss_res / ss_tot
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        metrics_dict['custom'] = metric(y, mu)
    else:
        raise ValueError("metric must be a string or callable")

    return metrics_dict

################################################################################
# interpretation_coefficients
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, coefs: np.ndarray) -> None:
    """Validate input arrays dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(coefs, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if coefs.ndim != 1:
        raise ValueError("coefs must be a 1D array")
    if X.shape[1] != coefs.shape[0]:
        raise ValueError("X and coefs dimensions are incompatible")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(coefs)) or np.any(np.isinf(coefs)):
        raise ValueError("coefs contains NaN or infinite values")

def _apply_normalization(X: np.ndarray, normalization: str) -> np.ndarray:
    """Apply specified normalization to features."""
    if normalization == "none":
        return X
    elif normalization == "standard":
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalization == "minmax":
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalization == "robust":
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      metrics: Union[str, Callable],
                      custom_metric: Optional[Callable] = None) -> Dict[str, float]:
    """Calculate specified metrics between true and predicted values."""
    result = {}
    if isinstance(metrics, str):
        if metrics == "mse":
            result["mse"] = np.mean((y_true - y_pred) ** 2)
        elif metrics == "mae":
            result["mae"] = np.mean(np.abs(y_true - y_pred))
        elif metrics == "r2":
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            result["r2"] = 1 - (ss_res / ss_tot)
        elif metrics == "logloss":
            result["logloss"] = -np.mean(y_true * np.log(y_pred + 1e-15) +
                                        (1 - y_true) * np.log(1 - y_pred + 1e-15))
        else:
            raise ValueError(f"Unknown metric: {metrics}")
    elif callable(metrics):
        result["custom"] = metrics(y_true, y_pred)
    else:
        raise TypeError("metrics must be either a string or callable")

    if custom_metric is not None:
        result["custom"] = custom_metric(y_true, y_pred)

    return result

def _interpret_coefficients(X: np.ndarray,
                          coefs: np.ndarray,
                          normalization: str = "none",
                          metrics: Union[str, Callable] = "mse",
                          custom_metric: Optional[Callable] = None) -> Dict:
    """Interpret regression coefficients with various options."""
    # Validate inputs
    _validate_inputs(X, coefs)

    # Apply normalization if specified
    X_norm = _apply_normalization(X, normalization)

    # Calculate predicted values (Poisson regression interpretation)
    lambda_pred = np.exp(X_norm @ coefs)

    # Calculate metrics
    metric_results = _calculate_metrics(lambda_pred, lambda_pred,
                                      metrics, custom_metric)

    # Prepare results dictionary
    result = {
        "result": {
            "normalized_coefficients": coefs,
            "interpretation": f"Coefficient interpretation for Poisson regression with {normalization} normalization"
        },
        "metrics": metric_results,
        "params_used": {
            "normalization": normalization,
            "metrics": metrics if not callable(metrics) else "custom function",
            "custom_metric_used": custom_metric is not None
        },
        "warnings": []
    }

    return result

def interpretation_coefficients_fit(X: np.ndarray,
                                  coefs: np.ndarray,
                                  normalization: str = "none",
                                  metrics: Union[str, Callable] = "mse",
                                  custom_metric: Optional[Callable] = None) -> Dict:
    """
    Main function to interpret regression coefficients with various options.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    coefs : np.ndarray
        Regression coefficients of shape (n_features,)
    normalization : str, optional
        Normalization method to apply ("none", "standard", "minmax", "robust")
    metrics : str or callable, optional
        Metric(s) to calculate ("mse", "mae", "r2", "logloss") or custom function
    custom_metric : callable, optional
        Additional custom metric function

    Returns
    -------
    dict
        Dictionary containing interpretation results, metrics, parameters used and warnings

    Example
    -------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> coefs = np.array([0.5, -0.3])
    >>> interpretation_coefficients_fit(X, coefs)
    """
    return _interpret_coefficients(X, coefs, normalization, metrics, custom_metric)

################################################################################
# confidence_intervals
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def confidence_intervals_fit(
    y: np.ndarray,
    X: np.ndarray,
    alpha: float = 0.05,
    method: str = 'wald',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    solver: str = 'newton',
    max_iter: int = 1000,
    tol: float = 1e-6
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Compute confidence intervals for Poisson regression coefficients.

    Parameters
    ----------
    y : np.ndarray
        Response variable (count data).
    X : np.ndarray
        Design matrix.
    alpha : float, optional
        Significance level (default: 0.05).
    method : str, optional
        Method for computing confidence intervals ('wald' or 'profile') (default: 'wald').
    normalizer : Callable, optional
        Function to normalize X (default: None).
    solver : str, optional
        Solver for Poisson regression ('newton' or 'gradient_descent') (default: 'newton').
    max_iter : int, optional
        Maximum iterations for solver (default: 1000).
    tol : float, optional
        Tolerance for convergence (default: 1e-6).

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': array of confidence intervals (lower, upper)
        - 'metrics': dictionary of metrics
        - 'params_used': dictionary of parameters used
        - 'warnings': list of warnings

    Example
    -------
    >>> y = np.array([1, 2, 3])
    >>> X = np.array([[1], [2], [3]])
    >>> result = confidence_intervals_fit(y, X)
    """
    # Validate inputs
    _validate_inputs(y, X)

    # Normalize X if specified
    if normalizer is not None:
        X = normalizer(X)

    # Fit Poisson regression model
    beta, covariance = _fit_poisson_regression(y, X, solver=solver, max_iter=max_iter, tol=tol)

    # Compute confidence intervals
    if method == 'wald':
        intervals = _wald_confidence_intervals(beta, covariance, alpha)
    elif method == 'profile':
        intervals = _profile_confidence_intervals(y, X, beta, alpha)
    else:
        raise ValueError("Method must be 'wald' or 'profile'")

    # Compute metrics
    metrics = _compute_metrics(y, X, beta)

    return {
        'result': intervals,
        'metrics': metrics,
        'params_used': {
            'alpha': alpha,
            'method': method,
            'solver': solver
        },
        'warnings': []
    }

def _validate_inputs(y: np.ndarray, X: np.ndarray) -> None:
    """Validate input arrays."""
    if y.ndim != 1 or X.ndim != 2:
        raise ValueError("y must be 1D and X must be 2D")
    if y.shape[0] != X.shape[0]:
        raise ValueError("y and X must have the same number of samples")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or Inf values")
    if np.any(np.isnan(X).any(axis=1)) or np.any(np.isinf(X).any(axis=1)):
        raise ValueError("X contains NaN or Inf values")

def _fit_poisson_regression(
    y: np.ndarray,
    X: np.ndarray,
    solver: str = 'newton',
    max_iter: int = 1000,
    tol: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Fit Poisson regression model and return coefficients and covariance matrix."""
    if solver == 'newton':
        beta, covariance = _newton_poisson_regression(y, X, max_iter, tol)
    elif solver == 'gradient_descent':
        beta, covariance = _gradient_descent_poisson_regression(y, X, max_iter, tol)
    else:
        raise ValueError("Solver must be 'newton' or 'gradient_descent'")
    return beta, covariance

def _wald_confidence_intervals(
    beta: np.ndarray,
    covariance: np.ndarray,
    alpha: float
) -> np.ndarray:
    """Compute Wald confidence intervals."""
    z_score = np.abs(np.sqrt(1.96 ** 2))  # For 95% confidence
    se = np.sqrt(np.diag(covariance))
    intervals = np.column_stack([
        beta - z_score * se,
        beta + z_score * se
    ])
    return intervals

def _profile_confidence_intervals(
    y: np.ndarray,
    X: np.ndarray,
    beta: np.ndarray,
    alpha: float
) -> np.ndarray:
    """Compute profile likelihood confidence intervals."""
    # Placeholder for actual implementation
    raise NotImplementedError("Profile likelihood method not yet implemented")

def _compute_metrics(
    y: np.ndarray,
    X: np.ndarray,
    beta: np.ndarray
) -> Dict[str, float]:
    """Compute regression metrics."""
    mu = np.exp(X @ beta)
    residuals = y - mu
    sse = np.sum(residuals ** 2)
    n = len(y)
    p = X.shape[1]
    rss = np.sum((y - mu) ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)

    return {
        'mse': sse / n,
        'mae': np.mean(np.abs(residuals)),
        'r2': 1 - rss / tss,
        'loglikelihood': _poisson_loglikelihood(y, mu)
    }

def _newton_poisson_regression(
    y: np.ndarray,
    X: np.ndarray,
    max_iter: int,
    tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Newton-Raphson algorithm for Poisson regression."""
    n, p = X.shape
    beta = np.zeros(p)
    mu = np.exp(X @ beta)

    for _ in range(max_iter):
        residuals = y - mu
        gradient = X.T @ (y - mu)
        hessian = -X.T @ np.diag(mu) @ X
        delta = np.linalg.solve(hessian, gradient)
        beta -= delta
        mu = np.exp(X @ beta)

        if np.linalg.norm(delta) < tol:
            break

    # Compute covariance matrix
    variance = np.diag(mu)
    cov = np.linalg.inv(X.T @ np.diag(1/variance) @ X)

    return beta, cov

def _gradient_descent_poisson_regression(
    y: np.ndarray,
    X: np.ndarray,
    max_iter: int,
    tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient descent algorithm for Poisson regression."""
    n, p = X.shape
    beta = np.zeros(p)
    learning_rate = 0.01

    for _ in range(max_iter):
        mu = np.exp(X @ beta)
        gradient = X.T @ (y - mu)
        beta += learning_rate * gradient

        if np.linalg.norm(gradient) < tol:
            break

    # Compute covariance matrix (approximate)
    mu = np.exp(X @ beta)
    variance = np.diag(mu)
    cov = np.linalg.inv(X.T @ np.diag(1/variance) @ X)

    return beta, cov

def _poisson_loglikelihood(y: np.ndarray, mu: np.ndarray) -> float:
    """Compute Poisson log-likelihood."""
    return np.sum(y * np.log(mu) - mu)

################################################################################
# hypothesis_testing
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def hypothesis_testing_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    test_type: str = 'wald',
    alpha: float = 0.05,
    normalization: Optional[str] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'deviance',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[float, Dict, str]]:
    """
    Perform hypothesis testing for Poisson regression models.

    Parameters
    ----------
    y_true : np.ndarray
        Array of observed counts.
    y_pred : np.ndarray
        Array of predicted counts from the Poisson regression model.
    test_type : str, optional
        Type of hypothesis test to perform ('wald', 'score', or 'likelihood_ratio').
    alpha : float, optional
        Significance level for the test (default is 0.05).
    normalization : str, optional
        Type of normalization to apply ('none', 'standard', or 'robust').
    metric : str or callable, optional
        Metric to use for evaluation ('deviance', 'pearson', or custom callable).
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    **kwargs :
        Additional keyword arguments for specific test types.

    Returns
    -------
    Dict[str, Union[float, Dict, str]]
        Dictionary containing test results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> y_true = np.array([10, 20, 30])
    >>> y_pred = np.array([9, 21, 29])
    >>> result = hypothesis_testing_fit(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Normalize data if specified
    if normalization:
        y_true, y_pred = _apply_normalization(y_true, y_pred, normalization)

    # Calculate test statistic and p-value
    if callable(metric):
        stat, p_value = _calculate_custom_test_statistic(y_true, y_pred, metric)
    else:
        stat, p_value = _calculate_test_statistic(y_true, y_pred, test_type, metric)

    # Calculate metrics
    metrics = _calculate_metrics(y_true, y_pred, metric if not callable(metric) else None)

    # Prepare result dictionary
    result = {
        'result': {
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < alpha
        },
        'metrics': metrics,
        'params_used': {
            'test_type': test_type,
            'alpha': alpha,
            'normalization': normalization,
            'metric': metric if not callable(metric) else 'custom'
        },
        'warnings': []
    }

    return result

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.any(y_true < 0) or np.any(y_pred < 0):
        raise ValueError("Counts must be non-negative.")
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("Inputs must not contain NaN values.")

def _apply_normalization(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalization: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply specified normalization to the input arrays."""
    if normalization == 'standard':
        mean_true = np.mean(y_true)
        std_true = np.std(y_true)
        y_true_normalized = (y_true - mean_true) / std_true
        mean_pred = np.mean(y_pred)
        std_pred = np.std(y_pred)
        y_pred_normalized = (y_pred - mean_pred) / std_pred
    elif normalization == 'robust':
        median_true = np.median(y_true)
        iqr_true = np.percentile(y_true, 75) - np.percentile(y_true, 25)
        y_true_normalized = (y_true - median_true) / iqr_true
        median_pred = np.median(y_pred)
        iqr_pred = np.percentile(y_pred, 75) - np.percentile(y_pred, 25)
        y_pred_normalized = (y_pred - median_pred) / iqr_pred
    else:
        y_true_normalized = y_true.copy()
        y_pred_normalized = y_pred.copy()

    return y_true_normalized, y_pred_normalized

def _calculate_test_statistic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_type: str,
    metric: str
) -> tuple[float, float]:
    """Calculate test statistic and p-value for the specified test type."""
    if metric == 'deviance':
        stat = _calculate_deviance(y_true, y_pred)
    elif metric == 'pearson':
        stat = _calculate_pearson_residuals(y_true, y_pred)
    else:
        raise ValueError("Unsupported metric for built-in test statistics.")

    if test_type == 'wald':
        p_value = _calculate_wald_p_value(stat)
    elif test_type == 'score':
        p_value = _calculate_score_p_value(stat)
    elif test_type == 'likelihood_ratio':
        p_value = _calculate_lr_p_value(stat)
    else:
        raise ValueError("Unsupported test type.")

    return stat, p_value

def _calculate_custom_test_statistic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float]
) -> tuple[float, float]:
    """Calculate test statistic and p-value using a custom metric."""
    stat = metric(y_true, y_pred)
    # For simplicity, assume the p-value is calculated using a normal approximation
    p_value = _calculate_normal_p_value(stat)
    return stat, p_value

def _calculate_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the deviance for Poisson regression."""
    return 2 * np.sum(y_true * np.log(y_true / y_pred) - (y_true - y_pred))

def _calculate_pearson_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Pearson chi-squared statistic."""
    return np.sum(((y_true - y_pred) ** 2) / y_pred)

def _calculate_wald_p_value(stat: float) -> float:
    """Calculate p-value for Wald test using normal approximation."""
    from scipy.stats import norm
    return 2 * (1 - norm.cdf(np.abs(stat)))

def _calculate_score_p_value(stat: float) -> float:
    """Calculate p-value for Score test using chi-squared distribution."""
    from scipy.stats import chi2
    return 1 - chi2.cdf(stat, df=1)

def _calculate_lr_p_value(stat: float) -> float:
    """Calculate p-value for Likelihood Ratio test using chi-squared distribution."""
    from scipy.stats import chi2
    return 1 - chi2.cdf(stat, df=1)

def _calculate_normal_p_value(stat: float) -> float:
    """Calculate p-value using normal approximation."""
    from scipy.stats import norm
    return 2 * (1 - norm.cdf(np.abs(stat)))

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Optional[str]
) -> Dict[str, float]:
    """Calculate metrics for the Poisson regression model."""
    metrics = {}

    if metric == 'deviance' or metric is None:
        metrics['deviance'] = _calculate_deviance(y_true, y_pred)
    if metric == 'pearson' or metric is None:
        metrics['pearson_chi2'] = _calculate_pearson_residuals(y_true, y_pred)

    return metrics

################################################################################
# model_selection
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def model_selection_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Perform model selection for Poisson regression.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]]
        Function to normalize the features. Default is None.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Metric to evaluate the model. Can be 'mse', 'mae', 'r2', or a custom callable.
    solver : str
        Solver to use. Options are 'gradient_descent', 'newton', or 'coordinate_descent'.
    regularization : Optional[str]
        Regularization type. Options are 'l1', 'l2', or None.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function. Overrides the `metric` parameter if provided.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if a normalizer is provided
    X_normalized = _apply_normalization(X, normalizer)

    # Choose the appropriate solver
    if solver == 'gradient_descent':
        params, history = _gradient_descent_solver(X_normalized, y, tol, max_iter, regularization)
    elif solver == 'newton':
        params, history = _newton_solver(X_normalized, y, tol, max_iter, regularization)
    elif solver == 'coordinate_descent':
        params, history = _coordinate_descent_solver(X_normalized, y, tol, max_iter, regularization)
    else:
        raise ValueError("Invalid solver specified.")

    # Calculate metrics
    y_pred = _predict(X_normalized, params)
    metrics = _calculate_metrics(y, y_pred, metric, custom_metric)

    # Prepare the result dictionary
    result = {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric if not custom_metric else custom_metric.__name__,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': _check_warnings(y_pred)
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate the input data."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be a 2D array and y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input data contains NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input data contains infinite values.")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    """Apply normalization to the feature matrix."""
    if normalizer is not None:
        return normalizer(X)
    return X

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int,
    regularization: Optional[str]
) -> tuple[np.ndarray, list]:
    """Gradient descent solver for Poisson regression."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    history = []

    for _ in range(max_iter):
        gradient = _poisson_gradient(X, y, params)
        if regularization == 'l1':
            gradient += np.sign(params)  # L1 regularization
        elif regularization == 'l2':
            gradient += 2 * params  # L2 regularization

        params_new = params - tol * gradient
        history.append(np.linalg.norm(params_new - params))
        if np.linalg.norm(params_new - params) < tol:
            break
        params = params_new

    return params, history

def _newton_solver(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int,
    regularization: Optional[str]
) -> tuple[np.ndarray, list]:
    """Newton's method solver for Poisson regression."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    history = []

    for _ in range(max_iter):
        gradient = _poisson_gradient(X, y, params)
        hessian = _poisson_hessian(X, y, params)

        if regularization == 'l1':
            hessian += np.diag(np.sign(params))  # L1 regularization
        elif regularization == 'l2':
            hessian += 2 * np.eye(n_features)  # L2 regularization

        params_new = params - np.linalg.solve(hessian, gradient)
        history.append(np.linalg.norm(params_new - params))
        if np.linalg.norm(params_new - params) < tol:
            break
        params = params_new

    return params, history

def _coordinate_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int,
    regularization: Optional[str]
) -> tuple[np.ndarray, list]:
    """Coordinate descent solver for Poisson regression."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    history = []

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - np.dot(X, params) + params[j] * X_j

            if regularization == 'l1':
                params[j] = np.sign(np.dot(X_j, residuals)) * np.maximum(
                    0, np.abs(np.dot(X_j, residuals)) - 1
                )
            elif regularization == 'l2':
                params[j] = np.dot(X_j, residuals) / (np.dot(X_j, X_j) + 2)
            else:
                params[j] = np.dot(X_j, residuals) / np.dot(X_j, X_j)

        history.append(np.linalg.norm(params))
        if len(history) > 1 and abs(history[-2] - history[-1]) < tol:
            break

    return params, history

def _poisson_gradient(X: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Compute the gradient for Poisson regression."""
    mu = np.exp(np.dot(X, params))
    return X.T @ (mu - y)

def _poisson_hessian(X: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Compute the Hessian for Poisson regression."""
    mu = np.exp(np.dot(X, params))
    return X.T @ np.diag(mu) @ X

def _predict(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Predict using the Poisson regression model."""
    return np.exp(np.dot(X, params))

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Calculate the metrics for the model."""
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

    return metrics

def _check_warnings(y_pred: np.ndarray) -> list:
    """Check for warnings in the predictions."""
    warnings = []
    if np.any(y_pred <= 0):
        warnings.append("Some predicted values are non-positive.")
    return warnings

################################################################################
# residual_analysis
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def residual_analysis_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    normalization: str = "none",
    metrics: Union[str, list] = ["mse", "mae"],
    custom_metrics: Optional[Dict[str, Callable]] = None,
    residual_type: str = "pearson",
    plot: bool = False
) -> Dict:
    """
    Perform residual analysis for Poisson regression.

    Parameters
    ----------
    y_true : np.ndarray
        Array of observed counts.
    y_pred : np.ndarray
        Array of predicted values (mu).
    normalization : str, optional
        Type of normalization to apply to residuals. Options: "none", "standard", "minmax".
    metrics : Union[str, list], optional
        Metrics to compute. Options: "mse", "mae", "r2", custom callable names.
    custom_metrics : Dict[str, Callable], optional
        Dictionary of custom metrics where keys are names and values are callables.
    residual_type : str, optional
        Type of residuals to compute. Options: "pearson", "deviance".
    plot : bool, optional
        Whether to generate plots (not implemented in this example).

    Returns
    -------
    Dict
        Dictionary containing:
        - "result": Computed residuals and other results
        - "metrics": Dictionary of computed metrics
        - "params_used": Parameters used in the computation
        - "warnings": List of warnings

    Example
    -------
    >>> y_true = np.array([1, 2, 3])
    >>> y_pred = np.array([1.5, 2.5, 2.8])
    >>> result = residual_analysis_fit(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Initialize warnings
    warnings = []

    # Compute residuals based on type
    if residual_type == "pearson":
        residuals = _compute_pearson_residuals(y_true, y_pred)
    elif residual_type == "deviance":
        residuals = _compute_deviance_residuals(y_true, y_pred)
    else:
        raise ValueError(f"Unknown residual type: {residual_type}")

    # Apply normalization if specified
    if normalization != "none":
        residuals = _apply_normalization(residuals, method=normalization)

    # Compute requested metrics
    metrics_dict = _compute_metrics(
        y_true, y_pred, residuals,
        metrics=metrics,
        custom_metrics=custom_metrics
    )

    # Prepare output dictionary
    result = {
        "result": {
            "residuals": residuals,
            "normalization_applied": normalization
        },
        "metrics": metrics_dict,
        "params_used": {
            "normalization": normalization,
            "residual_type": residual_type
        },
        "warnings": warnings
    }

    return result

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if np.any(y_true < 0):
        warnings.warn("Negative values in y_true are not valid for Poisson regression")
    if np.any(y_pred <= 0):
        warnings.warn("Non-positive values in y_pred are not valid for Poisson regression")

def _compute_pearson_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute Pearson residuals."""
    return (y_true - y_pred) / np.sqrt(y_pred)

def _compute_deviance_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute deviance residuals."""
    return np.sign(y_true - y_pred) * np.sqrt(
        2 * (y_true * np.log(y_true / y_pred) - (y_true - y_pred))
    )

def _apply_normalization(residuals: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to residuals."""
    if method == "standard":
        mean = np.mean(residuals)
        std = np.std(residuals)
        if std == 0:
            return residuals - mean
        return (residuals - mean) / std
    elif method == "minmax":
        min_val = np.min(residuals)
        max_val = np.max(residuals)
        if min_val == max_val:
            return residuals - min_val
        return (residuals - min_val) / (max_val - min_val)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    metrics: Union[str, list],
    custom_metrics: Optional[Dict[str, Callable]] = None
) -> Dict:
    """Compute requested metrics."""
    if isinstance(metrics, str):
        metrics = [metrics]

    metric_functions = {
        "mse": lambda yt, yp: np.mean((yt - yp) ** 2),
        "mae": lambda yt, yp: np.mean(np.abs(yt - yp)),
        "r2": lambda yt, yp: 1 - np.sum((yt - yp) ** 2) / np.sum((yt - np.mean(yt)) ** 2),
        "logloss": lambda yt, yp: -np.mean(yt * np.log(yp) - yp),
        "residual_mean": lambda _, res: np.mean(res),
        "residual_std": lambda _, res: np.std(res)
    }

    result = {}

    for metric in metrics:
        if metric in metric_functions:
            if metric in ["residual_mean", "residual_std"]:
                result[metric] = metric_functions[metric](None, residuals)
            else:
                result[metric] = metric_functions[metric](y_true, y_pred)
        elif custom_metrics and metric in custom_metrics:
            result[metric] = custom_metrics[metric](y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return result
