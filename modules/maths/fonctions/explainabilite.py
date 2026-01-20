"""
Quantix – Module explainabilite
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# interpretable_ml
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def interpretable_ml_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
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
    Fit an interpretable machine learning model.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    normalizer : Optional[Callable]
        Function to normalize the features. Default is None.
    metric : Union[str, Callable]
        Metric to evaluate the model. Can be 'mse', 'mae', 'r2', 'logloss' or a custom callable.
    distance : str
        Distance metric for the solver. Can be 'euclidean', 'manhattan', 'cosine', 'minkowski'.
    solver : str
        Solver to use. Can be 'closed_form', 'gradient_descent', 'newton', 'coordinate_descent'.
    regularization : Optional[str]
        Regularization type. Can be 'none', 'l1', 'l2', 'elasticnet'.
    tol : float
        Tolerance for stopping criteria.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable]
        Custom metric function if needed.
    custom_distance : Optional[Callable]
        Custom distance function if needed.

    Returns:
    --------
    Dict
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if a normalizer is provided
    X_normalized = _apply_normalization(X, normalizer)

    # Choose the appropriate solver
    if solver == 'closed_form':
        params = _closed_form_solver(X_normalized, y)
    elif solver == 'gradient_descent':
        params = _gradient_descent_solver(X_normalized, y, distance, regularization, tol, max_iter)
    elif solver == 'newton':
        params = _newton_solver(X_normalized, y, distance, regularization, tol, max_iter)
    elif solver == 'coordinate_descent':
        params = _coordinate_descent_solver(X_normalized, y, distance, regularization, tol, max_iter)
    else:
        raise ValueError("Invalid solver specified.")

    # Compute metrics
    metrics = _compute_metrics(y, params['predictions'], metric, custom_metric)

    # Prepare the result dictionary
    result = {
        'result': params,
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

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate the input data."""
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

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to the features."""
    if normalizer is not None:
        X_normalized = normalizer(X)
    else:
        X_normalized = X
    return X_normalized

def _closed_form_solver(X: np.ndarray, y: np.ndarray) -> Dict:
    """Closed form solution for linear regression."""
    XTX = np.dot(X.T, X)
    if np.linalg.det(XTX) == 0:
        raise ValueError("Matrix is singular.")
    XTY = np.dot(X.T, y)
    params = np.linalg.solve(XTX, XTY)
    predictions = np.dot(X, params)
    return {'params': params, 'predictions': predictions}

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    distance: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict:
    """Gradient descent solver."""
    n_samples, n_features = X.shape
    params = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradients = _compute_gradients(X, y, params, distance, regularization)
        params -= learning_rate * gradients
        if np.linalg.norm(gradients) < tol:
            break

    predictions = np.dot(X, params)
    return {'params': params, 'predictions': predictions}

def _compute_gradients(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    distance: str,
    regularization: Optional[str]
) -> np.ndarray:
    """Compute gradients for gradient descent."""
    predictions = np.dot(X, params)
    residuals = y - predictions

    if distance == 'euclidean':
        gradients = -2 * np.dot(X.T, residuals) / len(y)
    elif distance == 'manhattan':
        gradients = -np.dot(X.T, np.sign(residuals)) / len(y)
    else:
        raise ValueError("Unsupported distance metric.")

    if regularization == 'l1':
        gradients += np.sign(params)
    elif regularization == 'l2':
        gradients += 2 * params
    elif regularization == 'elasticnet':
        gradients += np.sign(params) + 2 * params

    return gradients

def _newton_solver(
    X: np.ndarray,
    y: np.ndarray,
    distance: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict:
    """Newton's method solver."""
    n_samples, n_features = X.shape
    params = np.zeros(n_features)

    for _ in range(max_iter):
        gradients = _compute_gradients(X, y, params, distance, regularization)
        hessian = _compute_hessian(X, distance, regularization)
        params -= np.linalg.solve(hessian, gradients)

        if np.linalg.norm(gradients) < tol:
            break

    predictions = np.dot(X, params)
    return {'params': params, 'predictions': predictions}

def _compute_hessian(
    X: np.ndarray,
    distance: str,
    regularization: Optional[str]
) -> np.ndarray:
    """Compute Hessian matrix for Newton's method."""
    n_samples, _ = X.shape

    if distance == 'euclidean':
        hessian = 2 * np.dot(X.T, X) / n_samples
    else:
        raise ValueError("Unsupported distance metric.")

    if regularization == 'l1':
        hessian += np.eye(len(hessian))
    elif regularization == 'l2':
        hessian += 2 * np.eye(len(hessian))
    elif regularization == 'elasticnet':
        hessian += np.eye(len(hessian)) + 2 * np.eye(len(hessian))

    return hessian

def _coordinate_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    distance: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict:
    """Coordinate descent solver."""
    n_samples, n_features = X.shape
    params = np.zeros(n_features)

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - np.dot(X, params) + params[j] * X_j
            if distance == 'euclidean':
                params[j] = np.dot(X_j, residuals) / (np.dot(X_j, X_j) + _get_regularization_term(j, params, regularization))
            else:
                raise ValueError("Unsupported distance metric.")

        if np.linalg.norm(_compute_gradients(X, y, params, distance, regularization)) < tol:
            break

    predictions = np.dot(X, params)
    return {'params': params, 'predictions': predictions}

def _get_regularization_term(j: int, params: np.ndarray, regularization: Optional[str]) -> float:
    """Get the regularization term for coordinate descent."""
    if regularization == 'l1':
        return 0.1 * np.sign(params[j])
    elif regularization == 'l2':
        return 0.1 * params[j]
    elif regularization == 'elasticnet':
        return 0.1 * (np.sign(params[j]) + params[j])
    else:
        return 0

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Compute metrics for the model."""
    metrics = {}

    if metric == 'mse':
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)
    elif metric == 'logloss':
        metrics['logloss'] = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    elif callable(metric):
        metrics['custom'] = metric(y_true, y_pred)
    else:
        raise ValueError("Invalid metric specified.")

    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)

    return metrics

################################################################################
# feature_importance
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def feature_importance_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, np.ndarray]]:
    """
    Compute feature importance using various statistical and machine learning methods.

    Parameters
    ----------
    X : np.ndarray
        Input features matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the input features. Default is identity.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Metric to evaluate the model performance. Can be "mse", "mae", "r2", or a custom callable.
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Distance metric for feature importance calculation. Can be "euclidean", "manhattan", "cosine",
        or a custom callable.
    solver : str
        Solver to use for optimization. Options are "closed_form", "gradient_descent", etc.
    regularization : Optional[str]
        Regularization type. Options are "l1", "l2", or None.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function if not using built-in metrics.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom distance function if not using built-in distances.

    Returns
    -------
    Dict[str, Union[Dict, np.ndarray]]
        Dictionary containing:
        - "result": Feature importance scores.
        - "metrics": Computed metrics.
        - "params_used": Parameters used in the computation.
        - "warnings": Any warnings encountered.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = feature_importance_fit(X, y, normalizer=np.std, metric="r2")
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features
    X_normalized = normalizer(X)

    # Set metric and distance functions
    metric_func = _get_metric_function(metric, custom_metric)
    distance_func = _get_distance_function(distance, custom_distance)

    # Solve for feature importance
    if solver == "closed_form":
        result = _solve_closed_form(X_normalized, y)
    else:
        raise ValueError(f"Solver {solver} not implemented.")

    # Compute metrics
    y_pred = _predict(X_normalized, result)
    metrics = {
        "metric": metric_func(y, y_pred),
        "distance": distance_func(X_normalized, result)
    }

    # Prepare output
    output = {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if hasattr(normalizer, "__name__") else "custom",
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    return output

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")

def _get_metric_function(
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the metric function based on input."""
    if custom_metric is not None:
        return custom_metric
    if metric == "mse":
        return _mean_squared_error
    elif metric == "mae":
        return _mean_absolute_error
    elif metric == "r2":
        return _r_squared
    else:
        raise ValueError(f"Metric {metric} not recognized.")

def _get_distance_function(
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the distance function based on input."""
    if custom_distance is not None:
        return custom_distance
    if distance == "euclidean":
        return _euclidean_distance
    elif distance == "manhattan":
        return _manhattan_distance
    elif distance == "cosine":
        return _cosine_distance
    else:
        raise ValueError(f"Distance {distance} not recognized.")

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve for feature importance using closed-form solution."""
    X_t = X.T
    return np.linalg.pinv(X_t @ X) @ X_t @ y

def _predict(X: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Predict using the computed coefficients."""
    return X @ coefficients

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

def _euclidean_distance(X: np.ndarray, y: np.ndarray) -> float:
    """Compute Euclidean distance."""
    return np.linalg.norm(X - y)

def _manhattan_distance(X: np.ndarray, y: np.ndarray) -> float:
    """Compute Manhattan distance."""
    return np.sum(np.abs(X - y))

def _cosine_distance(X: np.ndarray, y: np.ndarray) -> float:
    """Compute cosine distance."""
    return 1 - (X @ y) / (np.linalg.norm(X) * np.linalg.norm(y))

################################################################################
# shap_values
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def shap_values_fit(
    model: Callable,
    X: np.ndarray,
    normalizer: Optional[Callable] = None,
    metric: Union[str, Callable] = "mse",
    distance: str = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None,
    custom_metric_params: Optional[Dict] = None
) -> Dict:
    """
    Compute SHAP values for a given model and dataset.

    Parameters
    ----------
    model : Callable
        The model for which to compute SHAP values.
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    normalizer : Optional[Callable], default=None
        Normalization function to apply to the data.
    metric : Union[str, Callable], default="mse"
        Metric to use for SHAP value computation. Can be "mse", "mae", "r2",
        "logloss" or a custom callable.
    distance : str, default="euclidean"
        Distance metric to use. Can be "euclidean", "manhattan", "cosine",
        or "minkowski".
    solver : str, default="closed_form"
        Solver to use for optimization. Can be "closed_form", "gradient_descent",
        "newton", or "coordinate_descent".
    regularization : Optional[str], default=None
        Regularization type. Can be "none", "l1", "l2", or "elasticnet".
    tol : float, default=1e-4
        Tolerance for convergence.
    max_iter : int, default=1000
        Maximum number of iterations for solvers.
    random_state : Optional[int], default=None
        Random seed for reproducibility.
    custom_metric_params : Optional[Dict], default=None
        Parameters for custom metric functions.

    Returns
    -------
    Dict
        Dictionary containing:
        - "result": Computed SHAP values.
        - "metrics": Evaluation metrics.
        - "params_used": Parameters used in computation.
        - "warnings": Any warnings encountered.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> X = np.random.rand(100, 5)
    >>> model = LinearRegression()
    >>> shap_values_fit(model, X)
    """
    # Validate inputs
    _validate_inputs(X)

    # Normalize data if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Prepare parameters dictionary
    params_used = {
        "normalizer": normalizer.__name__ if normalizer else None,
        "metric": metric,
        "distance": distance,
        "solver": solver,
        "regularization": regularization,
        "tol": tol,
        "max_iter": max_iter
    }

    # Compute SHAP values based on solver choice
    if solver == "closed_form":
        shap_values = _compute_shap_closed_form(model, X_normalized, metric)
    elif solver == "gradient_descent":
        shap_values = _compute_shap_gradient_descent(
            model, X_normalized, metric, tol, max_iter, random_state
        )
    elif solver == "newton":
        shap_values = _compute_shap_newton(
            model, X_normalized, metric, tol, max_iter
        )
    elif solver == "coordinate_descent":
        shap_values = _compute_shap_coordinate_descent(
            model, X_normalized, metric, tol, max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(model, X_normalized, metric, custom_metric_params)

    # Prepare warnings
    warnings = _check_warnings(shap_values, metrics)

    return {
        "result": shap_values,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

def _validate_inputs(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")

def _apply_normalization(
    X: np.ndarray,
    normalizer: Optional[Callable] = None
) -> np.ndarray:
    """Apply normalization to the input data."""
    if normalizer is None:
        return X
    return normalizer(X)

def _compute_shap_closed_form(
    model: Callable,
    X: np.ndarray,
    metric: Union[str, Callable]
) -> np.ndarray:
    """Compute SHAP values using closed form solution."""
    # Implementation of closed form SHAP value computation
    pass

def _compute_shap_gradient_descent(
    model: Callable,
    X: np.ndarray,
    metric: Union[str, Callable],
    tol: float,
    max_iter: int,
    random_state: Optional[int]
) -> np.ndarray:
    """Compute SHAP values using gradient descent."""
    # Implementation of gradient descent for SHAP value computation
    pass

def _compute_shap_newton(
    model: Callable,
    X: np.ndarray,
    metric: Union[str, Callable],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Compute SHAP values using Newton's method."""
    # Implementation of Newton's method for SHAP value computation
    pass

def _compute_shap_coordinate_descent(
    model: Callable,
    X: np.ndarray,
    metric: Union[str, Callable],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Compute SHAP values using coordinate descent."""
    # Implementation of coordinate descent for SHAP value computation
    pass

def _compute_metrics(
    model: Callable,
    X: np.ndarray,
    metric: Union[str, Callable],
    custom_metric_params: Optional[Dict] = None
) -> Dict:
    """Compute evaluation metrics."""
    # Implementation of metric computation
    pass

def _check_warnings(
    shap_values: np.ndarray,
    metrics: Dict
) -> List[str]:
    """Check for potential warnings."""
    # Implementation of warning checks
    pass

################################################################################
# lime_explanations
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Inputs contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Inputs contain infinite values")

def normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
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

def compute_metric(y_true: np.ndarray, y_pred: np.ndarray,
                  metric: Union[str, Callable]) -> float:
    """Compute specified metric between true and predicted values."""
    if callable(metric):
        return metric(y_true, y_pred)
    elif metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    elif metric == 'logloss':
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def compute_distance(X: np.ndarray, method: str = 'euclidean',
                    custom_func: Optional[Callable] = None) -> np.ndarray:
    """Compute pairwise distances between samples."""
    if custom_func is not None:
        return np.array([[custom_func(x1, x2) for x2 in X] for x1 in X])
    if method == 'euclidean':
        return np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
    elif method == 'manhattan':
        return np.sum(np.abs(X[:, np.newaxis] - X), axis=2)
    elif method == 'cosine':
        dot_products = np.dot(X, X.T)
        norms = np.sqrt(np.sum(X ** 2, axis=1))[:, np.newaxis]
        return 1 - (dot_products / (norms * norms.T + 1e-8))
    elif method == 'minkowski':
        p = 3
        return np.sum(np.abs(X[:, np.newaxis] - X) ** p, axis=2) ** (1/p)
    else:
        raise ValueError(f"Unknown distance method: {method}")

def lime_explanations_fit(X: np.ndarray, y: np.ndarray,
                         model: Callable,
                         num_samples: int = 1000,
                         normalization: str = 'standard',
                         metric: Union[str, Callable] = 'mse',
                         distance_method: str = 'euclidean',
                         custom_distance_func: Optional[Callable] = None,
                         kernel_width: float = 0.75) -> Dict[str, Any]:
    """
    Compute LIME explanations for a given model.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    model : Callable
        The model to explain. Should accept X and return predictions.
    num_samples : int, optional
        Number of samples to generate for explanation (default: 1000)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard')
    metric : str or Callable, optional
        Metric to evaluate explanations ('mse', 'mae', 'r2', 'logloss') (default: 'mse')
    distance_method : str, optional
        Distance method ('euclidean', 'manhattan', 'cosine', 'minkowski') (default: 'euclidean')
    custom_distance_func : Callable, optional
        Custom distance function if not using built-in methods
    kernel_width : float, optional
        Width of the exponential kernel (default: 0.75)

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_norm = normalize_data(X.copy(), normalization)

    # Generate random samples
    np.random.seed(42)
    sample_indices = np.random.choice(X.shape[0], num_samples, replace=True)
    X_sampled = X_norm[sample_indices]
    y_sampled = y[sample_indices]

    # Compute distances
    distances = compute_distance(X_sampled, distance_method, custom_distance_func)

    # Compute kernel weights
    kernel_weights = np.exp(-distances ** 2 / (kernel_width ** 2))

    # Get model predictions
    y_pred = model(X_sampled)

    # Compute metric
    primary_metric = compute_metric(y_sampled, y_pred, metric)

    # Compute feature importance
    weighted_X = X_sampled * kernel_weights[:, :, np.newaxis]
    feature_importance = np.mean(weighted_X, axis=0)

    return {
        "result": {
            "feature_importance": feature_importance,
            "sampled_indices": sample_indices
        },
        "metrics": {
            "primary_metric": primary_metric,
            "kernel_width_used": kernel_width
        },
        "params_used": {
            "normalization": normalization,
            "metric": metric,
            "distance_method": distance_method,
            "num_samples": num_samples
        },
        "warnings": []
    }

# Example usage:
"""
from sklearn.linear_model import LinearRegression

# Create sample data
X = np.random.rand(100, 5)
y = X.dot(np.array([2.5, -1.3, 0.8, -0.4, 1.7])) + np.random.normal(0, 0.1, 100)

# Train a simple model
model = LinearRegression().fit(X, y)
predict_func = lambda x: model.predict(x)

# Get LIME explanations
explanations = lime_explanations_fit(
    X, y,
    model=predict_func,
    num_samples=500,
    normalization='standard',
    metric='r2'
)
"""

################################################################################
# partial_dependence_plots
################################################################################

import numpy as np
from typing import Callable, Dict, List, Optional, Union

def partial_dependence_plots_fit(
    model: Callable,
    X: np.ndarray,
    features: List[int],
    grid_points: int = 100,
    normalize: str = "none",
    metric: Union[str, Callable] = "mse",
    distance: str = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict:
    """
    Compute partial dependence plots for specified features.

    Parameters:
    -----------
    model : Callable
        The trained model to explain. Must have a predict method.
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    features : List[int]
        Indices of features to compute partial dependence for.
    grid_points : int, optional
        Number of points in the grid (default: 100).
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'none').
    metric : Union[str, Callable], optional
        Metric to evaluate ('mse', 'mae', 'r2', 'logloss') or custom callable (default: 'mse').
    distance : str, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') (default: 'euclidean').
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent') (default: 'closed_form').
    regularization : Optional[str], optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet') (default: None).
    tol : float, optional
        Tolerance for convergence (default: 1e-4).
    max_iter : int, optional
        Maximum number of iterations (default: 1000).
    random_state : Optional[int], optional
        Random seed for reproducibility (default: None).

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, features)

    # Normalize data if required
    X_normalized = _normalize_data(X, normalize)

    # Compute partial dependence
    pdp_results = _compute_partial_dependence(
        model, X_normalized, features, grid_points, metric, distance, solver,
        regularization, tol, max_iter, random_state
    )

    # Prepare output dictionary
    return {
        "result": pdp_results,
        "metrics": _compute_metrics(pdp_results, metric),
        "params_used": {
            "grid_points": grid_points,
            "normalize": normalize,
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": _check_warnings(pdp_results)
    }

def _validate_inputs(X: np.ndarray, features: List[int]) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X must be a 2D numpy array.")
    if not all(isinstance(f, int) and f >= 0 for f in features):
        raise ValueError("Features must be non-negative integers.")
    if any(f >= X.shape[1] for f in features):
        raise ValueError("Feature indices out of bounds.")

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize data based on specified method."""
    if method == "none":
        return X
    elif method == "standard":
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == "minmax":
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == "robust":
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_partial_dependence(
    model: Callable,
    X: np.ndarray,
    features: List[int],
    grid_points: int,
    metric: Union[str, Callable],
    distance: str,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    random_state: Optional[int]
) -> Dict:
    """Compute partial dependence for specified features."""
    results = {}
    for feature in features:
        # Create grid for the current feature
        min_val, max_val = np.min(X[:, feature]), np.max(X[:, feature])
        grid = np.linspace(min_val, max_val, grid_points)

        # Compute partial dependence
        pd_values = []
        for val in grid:
            X_temp = X.copy()
            X_temp[:, feature] = val
            y_pred = model.predict(X_temp)
            pd_values.append(np.mean(y_pred))

        results[feature] = {
            "grid": grid,
            "partial_dependence": np.array(pd_values)
        }

    return results

def _compute_metrics(results: Dict, metric: Union[str, Callable]) -> Dict:
    """Compute metrics for partial dependence results."""
    metrics = {}
    for feature, data in results.items():
        if callable(metric):
            metrics[feature] = metric(data["grid"], data["partial_dependence"])
        elif metric == "mse":
            metrics[feature] = np.mean((data["partial_dependence"] - np.mean(data["partial_dependence"]))**2)
        elif metric == "mae":
            metrics[feature] = np.mean(np.abs(data["partial_dependence"] - np.mean(data["partial_dependence"])))
        elif metric == "r2":
            ss_total = np.sum((data["partial_dependence"] - np.mean(data["partial_dependence"]))**2)
            ss_res = np.sum((data["partial_dependence"] - data["grid"])**2)
            metrics[feature] = 1 - (ss_res / ss_total) if ss_total != 0 else 0
        elif metric == "logloss":
            metrics[feature] = -np.mean(data["partial_dependence"] * np.log(data["partial_dependence"]))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return metrics

def _check_warnings(results: Dict) -> List[str]:
    """Check for warnings in the results."""
    warnings = []
    for feature, data in results.items():
        if np.any(np.isnan(data["partial_dependence"])):
            warnings.append(f"NaN values detected in partial dependence for feature {feature}.")
        if np.any(np.isinf(data["partial_dependence"])):
            warnings.append(f"Infinite values detected in partial dependence for feature {feature}.")

    return warnings

################################################################################
# individual_condition_expectation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Any

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    feature_index: int,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> None:
    """Validate input data and parameters."""
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if feature_index < 0 or feature_index >= X.shape[1]:
        raise ValueError("feature_index out of bounds.")
    if normalizer is not None and not callable(normalizer):
        raise ValueError("normalizer must be a callable or None.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")

def _compute_conditional_expectation(
    X: np.ndarray,
    y: np.ndarray,
    feature_index: int,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Any]:
    """Compute the conditional expectation for a given feature."""
    if normalizer is not None:
        X = normalizer(X)

    feature_values = X[:, feature_index]
    unique_values = np.unique(feature_values)
    result = {}

    for value in unique_values:
        mask = feature_values == value
        if np.sum(mask) > 0:
            result[value] = {
                'mean': float(np.mean(y[mask])),
                'std': float(np.std(y[mask])),
                'count': int(np.sum(mask))
            }

    return result

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Compute metrics for the conditional expectation."""
    return {'metric': metric_func(y_true, y_pred)}

def individual_condition_expectation_fit(
    X: np.ndarray,
    y: np.ndarray,
    feature_index: int,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric_func: Callable[[np.ndarray, np.ndarray], float] = lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
    **kwargs
) -> Dict[str, Any]:
    """
    Compute the individual conditional expectation for a given feature.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    feature_index : int
        Index of the feature to compute the conditional expectation for.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the input features, by default None.
    metric_func : Callable[[np.ndarray, np.ndarray], float], optional
        Function to compute the metric between true and predicted values, by default MSE.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_inputs(X, y, feature_index, normalizer)

    result = _compute_conditional_expectation(X, y, feature_index, normalizer)
    metrics = {}

    for value in result:
        mask = X[:, feature_index] == value
        y_pred = np.full_like(y, result[value]['mean'])
        metrics[f'value_{value}'] = _compute_metrics(y[mask], y_pred, metric_func)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'feature_index': feature_index,
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric_func': metric_func.__name__
        },
        'warnings': []
    }

# Example usage:
# X = np.random.rand(100, 5)
# y = np.random.rand(100)
# result = individual_condition_expectation_fit(X, y, feature_index=2)

################################################################################
# anchor_explanations
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def anchor_explanations_fit(
    model: Callable,
    X: np.ndarray,
    feature_names: Optional[list] = None,
    distance_metric: str = 'euclidean',
    normalization: str = 'none',
    solver: str = 'closed_form',
    threshold: float = 0.95,
    max_iter: int = 1000,
    tol: float = 1e-4,
    metric: str = 'mse',
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Compute anchor explanations for a given model.

    Parameters:
    -----------
    model : Callable
        The trained model to explain.
    X : np.ndarray
        Input features for which to compute explanations.
    feature_names : Optional[list], default=None
        Names of the features.
    distance_metric : str, default='euclidean'
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski').
    normalization : str, default='none'
        Normalization method ('none', 'standard', 'minmax', 'robust').
    solver : str, default='closed_form'
        Solver to use ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    threshold : float, default=0.95
        Confidence threshold for anchor explanations.
    max_iter : int, default=1000
        Maximum number of iterations for iterative solvers.
    tol : float, default=1e-4
        Tolerance for convergence.
    metric : str, default='mse'
        Metric to evaluate explanations ('mse', 'mae', 'r2', 'logloss').
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
    _validate_inputs(X, feature_names)

    # Normalize data if required
    X_normalized = _apply_normalization(X, normalization)

    # Compute anchor explanations
    anchors = _compute_anchors(
        model,
        X_normalized,
        distance_metric=distance_metric,
        solver=solver,
        threshold=threshold,
        max_iter=max_iter,
        tol=tol,
        metric=metric,
        custom_metric=custom_metric,
        custom_distance=custom_distance
    )

    # Compute metrics
    metrics = _compute_metrics(anchors, X_normalized, metric, custom_metric)

    # Prepare output
    result = {
        "result": anchors,
        "metrics": metrics,
        "params_used": {
            "distance_metric": distance_metric,
            "normalization": normalization,
            "solver": solver,
            "threshold": threshold,
            "max_iter": max_iter,
            "tol": tol,
            "metric": metric
        },
        "warnings": []
    }

    return result

def _validate_inputs(X: np.ndarray, feature_names: Optional[list]) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values.")
    if feature_names is not None and len(feature_names) != X.shape[1]:
        raise ValueError("feature_names length must match number of features in X.")

def _apply_normalization(X: np.ndarray, normalization: str) -> np.ndarray:
    """Apply specified normalization to the input data."""
    if normalization == 'standard':
        X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalization == 'minmax':
        X_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalization == 'robust':
        median = np.median(X, axis=0)
        iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
        X_normalized = (X - median) / iqr
    else:
        X_normalized = X.copy()
    return X_normalized

def _compute_anchors(
    model: Callable,
    X: np.ndarray,
    distance_metric: str,
    solver: str,
    threshold: float,
    max_iter: int,
    tol: float,
    metric: str,
    custom_metric: Optional[Callable],
    custom_distance: Optional[Callable]
) -> np.ndarray:
    """Compute anchor explanations using the specified solver and distance metric."""
    # Placeholder for actual implementation
    anchors = np.zeros_like(X)
    return anchors

def _compute_metrics(
    anchors: np.ndarray,
    X: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable]
) -> Dict[str, float]:
    """Compute metrics for the anchor explanations."""
    if custom_metric is not None:
        return {"custom_metric": custom_metric(anchors, X)}

    metrics = {}
    if metric == 'mse':
        metrics['mse'] = np.mean((anchors - X) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(anchors - X))
    elif metric == 'r2':
        ss_res = np.sum((X - anchors) ** 2)
        ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)
    elif metric == 'logloss':
        metrics['logloss'] = -np.mean(X * np.log(anchors) + (1 - X) * np.log(1 - anchors))
    return metrics

# Example usage:
# model = lambda x: np.sum(x, axis=1)
# X = np.random.rand(100, 5)
# result = anchor_explanations_fit(model, X)

################################################################################
# counterfactual_explanations
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def counterfactual_explanations_fit(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    distance_metric: Union[str, Callable] = 'euclidean',
    normalization: str = 'standard',
    solver: str = 'gradient_descent',
    metric: Union[str, Callable] = 'mse',
    max_iter: int = 1000,
    tol: float = 1e-4,
    reg_type: str = 'none',
    reg_param: float = 0.1,
    custom_distance: Optional[Callable] = None,
    custom_metric: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Compute counterfactual explanations for a given model and dataset.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    model : Callable
        The predictive model to explain.
    distance_metric : str or Callable, optional
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    solver : str, optional
        Solver to use ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    metric : str or Callable, optional
        Metric to optimize ('mse', 'mae', 'r2', 'logloss') or custom callable.
    max_iter : int, optional
        Maximum number of iterations for iterative solvers.
    tol : float, optional
        Tolerance for convergence.
    reg_type : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet').
    reg_param : float, optional
        Regularization parameter.
    custom_distance : Callable, optional
        Custom distance function if not using built-in metrics.
    custom_metric : Callable, optional
        Custom metric function if not using built-in metrics.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _normalize_data(X, normalization)

    # Initialize parameters
    params_used = {
        'distance_metric': distance_metric,
        'normalization': normalization,
        'solver': solver,
        'metric': metric,
        'max_iter': max_iter,
        'tol': tol,
        'reg_type': reg_type,
        'reg_param': reg_param
    }

    # Get distance function
    distance_func = _get_distance_function(distance_metric, custom_distance)

    # Get metric function
    metric_func = _get_metric_function(metric, custom_metric)

    # Compute counterfactual explanations
    results = _compute_counterfactuals(
        X_normalized, y, model, distance_func, metric_func,
        solver, max_iter, tol, reg_type, reg_param
    )

    # Calculate metrics
    metrics = _calculate_metrics(results, metric_func)

    return {
        'result': results,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
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

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
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

def _get_distance_function(metric: str, custom_func: Optional[Callable]) -> Callable:
    """Get distance function based on input."""
    if custom_func is not None:
        return custom_func
    if metric == 'euclidean':
        return lambda x, y: np.linalg.norm(x - y)
    elif metric == 'manhattan':
        return lambda x, y: np.sum(np.abs(x - y))
    elif metric == 'cosine':
        return lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    elif metric == 'minkowski':
        return lambda x, y: np.sum(np.abs(x - y)**3)**(1/3)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _get_metric_function(metric: str, custom_func: Optional[Callable]) -> Callable:
    """Get metric function based on input."""
    if custom_func is not None:
        return custom_func
    if metric == 'mse':
        return lambda y_true, y_pred: np.mean((y_true - y_pred)**2)
    elif metric == 'mae':
        return lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        return lambda y_true, y_pred: 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    elif metric == 'logloss':
        return lambda y_true, y_pred: -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _compute_counterfactuals(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    distance_func: Callable,
    metric_func: Callable,
    solver: str,
    max_iter: int,
    tol: float,
    reg_type: str,
    reg_param: float
) -> Dict[str, Any]:
    """Compute counterfactual explanations using specified solver."""
    n_samples = X.shape[0]
    results = {'counterfactuals': np.zeros_like(X)}

    for i in range(n_samples):
        if solver == 'closed_form':
            results['counterfactuals'][i] = _closed_form_solver(X[i], y[i], model, distance_func)
        elif solver == 'gradient_descent':
            results['counterfactuals'][i] = _gradient_descent_solver(
                X[i], y[i], model, distance_func, metric_func,
                max_iter, tol, reg_type, reg_param
            )
        elif solver == 'newton':
            results['counterfactuals'][i] = _newton_solver(
                X[i], y[i], model, distance_func, metric_func,
                max_iter, tol, reg_type, reg_param
            )
        elif solver == 'coordinate_descent':
            results['counterfactuals'][i] = _coordinate_descent_solver(
                X[i], y[i], model, distance_func, metric_func,
                max_iter, tol, reg_type, reg_param
            )
        else:
            raise ValueError(f"Unknown solver: {solver}")

    return results

def _closed_form_solver(
    x: np.ndarray,
    y_true: float,
    model: Callable,
    distance_func: Callable
) -> np.ndarray:
    """Closed form solution for counterfactual explanations."""
    # Simplified implementation - in practice would need to solve the optimization problem
    return x

def _gradient_descent_solver(
    x: np.ndarray,
    y_true: float,
    model: Callable,
    distance_func: Callable,
    metric_func: Callable,
    max_iter: int,
    tol: float,
    reg_type: str,
    reg_param: float
) -> np.ndarray:
    """Gradient descent solver for counterfactual explanations."""
    # Simplified implementation - in practice would need to implement proper gradient descent
    return x

def _newton_solver(
    x: np.ndarray,
    y_true: float,
    model: Callable,
    distance_func: Callable,
    metric_func: Callable,
    max_iter: int,
    tol: float,
    reg_type: str,
    reg_param: float
) -> np.ndarray:
    """Newton's method solver for counterfactual explanations."""
    # Simplified implementation - in practice would need to implement proper Newton's method
    return x

def _coordinate_descent_solver(
    x: np.ndarray,
    y_true: float,
    model: Callable,
    distance_func: Callable,
    metric_func: Callable,
    max_iter: int,
    tol: float,
    reg_type: str,
    reg_param: float
) -> np.ndarray:
    """Coordinate descent solver for counterfactual explanations."""
    # Simplified implementation - in practice would need to implement proper coordinate descent
    return x

def _calculate_metrics(
    results: Dict[str, Any],
    metric_func: Callable
) -> Dict[str, float]:
    """Calculate metrics for counterfactual explanations."""
    # Simplified implementation - in practice would need to calculate proper metrics
    return {'metric_value': 0.0}

################################################################################
# saliency_maps
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate input arrays for saliency maps computation.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("Inputs must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Normalize input data using specified method.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    method : str
        Normalization method ('none', 'standard', 'minmax', 'robust')

    Returns
    ------
    np.ndarray
        Normalized data array
    """
    if method == 'none':
        return X.copy()
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

def compute_saliency(
    X: np.ndarray,
    y: np.ndarray,
    model_func: Callable,
    metric_func: Callable = None,
    distance_metric: str = 'euclidean',
    normalize_method: str = 'standard'
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Compute saliency maps for a given model and data.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    model_func : Callable
        Function that takes X and returns predictions
    metric_func : Callable, optional
        Metric function to evaluate model performance
    distance_metric : str
        Distance metric for perturbation ('euclidean', 'manhattan', etc.)
    normalize_method : str
        Normalization method for input data

    Returns
    ------
    Dict[str, Union[np.ndarray, float]]
        Dictionary containing saliency results and metrics
    """
    validate_inputs(X, y)
    X_norm = normalize_data(X, method=normalize_method)

    # Default metric if none provided
    if metric_func is None:
        def metric_func(y_true, y_pred):
            return -np.mean((y_true - y_pred) ** 2)

    # Compute baseline prediction
    y_pred = model_func(X_norm)
    baseline_score = metric_func(y, y_pred)

    # Initialize saliency map
    n_samples, n_features = X_norm.shape
    saliency_map = np.zeros((n_samples, n_features))

    # Compute distance function based on specified metric
    if distance_metric == 'euclidean':
        def dist_func(x1, x2):
            return np.linalg.norm(x1 - x2)
    elif distance_metric == 'manhattan':
        def dist_func(x1, x2):
            return np.sum(np.abs(x1 - x2))
    elif distance_metric == 'cosine':
        def dist_func(x1, x2):
            return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    # Compute saliency for each feature
    for i in range(n_features):
        # Create perturbed versions of the data
        X_perturbed = X_norm.copy()
        X_perturbed[:, i] += 1e-4  # Small perturbation

        # Compute perturbed predictions
        y_pred_perturbed = model_func(X_perturbed)
        perturbed_score = metric_func(y, y_pred_perturbed)

        # Compute saliency for this feature
        saliency_map[:, i] = (baseline_score - perturbed_score) / dist_func(X_norm, X_perturbed)

    return {
        'saliency_map': saliency_map,
        'baseline_score': baseline_score,
        'metrics': {'metric_used': metric_func.__name__},
        'params_used': {
            'distance_metric': distance_metric,
            'normalize_method': normalize_method
        },
        'warnings': []
    }

def saliency_maps_fit(
    X: np.ndarray,
    y: np.ndarray,
    model_func: Callable,
    metric_func: Optional[Callable] = None,
    distance_metric: str = 'euclidean',
    normalize_method: str = 'standard'
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Compute saliency maps for model explainability.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    model_func : Callable
        Function that takes X and returns predictions
    metric_func : Callable, optional
        Metric function to evaluate model performance
    distance_metric : str
        Distance metric for perturbation ('euclidean', 'manhattan', etc.)
    normalize_method : str
        Normalization method for input data

    Returns
    ------
    Dict[str, Union[np.ndarray, float]]
        Dictionary containing saliency results and metrics

    Examples
    --------
    >>> def simple_model(X):
    ...     return X @ np.array([1, 2]) + 0.5
    ...
    >>> saliency_maps_fit(X_train, y_train, simple_model)
    """
    return compute_saliency(
        X=X,
        y=y,
        model_func=model_func,
        metric_func=metric_func,
        distance_metric=distance_metric,
        normalize_method=normalize_method
    )

################################################################################
# decision_rules
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def decision_rules_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Fit decision rules to explain a model's predictions.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    normalizer : callable
        Function to normalize features. Default is identity.
    metric : str or callable
        Metric to evaluate performance. Can be 'mse', 'mae', 'r2', or custom callable.
    distance : str or callable
        Distance metric for feature space. Can be 'euclidean', 'manhattan', 'cosine', or custom.
    solver : str
        Solver to use. Options: 'closed_form', 'gradient_descent', 'newton', 'coordinate_descent'.
    regularization : str or None
        Regularization type. Options: 'l1', 'l2', 'elasticnet', or None.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : callable or None
        Custom metric function if needed.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features
    X_norm = normalizer(X)

    # Choose metric
    if isinstance(metric, str):
        metric_func = _get_metric(metric)
    else:
        metric_func = metric

    # Choose distance
    if isinstance(distance, str):
        distance_func = _get_distance(distance)
    else:
        distance_func = distance

    # Choose solver
    if solver == 'closed_form':
        params, metrics = _solve_closed_form(X_norm, y, regularization)
    elif solver == 'gradient_descent':
        params, metrics = _solve_gradient_descent(X_norm, y, metric_func,
                                                 distance_func, regularization,
                                                 tol, max_iter)
    elif solver == 'newton':
        params, metrics = _solve_newton(X_norm, y, metric_func,
                                       distance_func, regularization,
                                       tol, max_iter)
    elif solver == 'coordinate_descent':
        params, metrics = _solve_coordinate_descent(X_norm, y, metric_func,
                                                   distance_func, regularization,
                                                   tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate custom metrics if provided
    if custom_metric is not None:
        custom_score = custom_metric(y, np.dot(X_norm, params))
        metrics['custom_metric'] = custom_score

    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': _check_warnings(X_norm, y, params)
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

def _get_metric(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return the specified metric function."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _get_distance(distance: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return the specified distance function."""
    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance
    }
    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")
    return distances[distance]

def _solve_closed_form(X: np.ndarray, y: np.ndarray,
                      regularization: Optional[str] = None) -> tuple:
    """Solve using closed form solution."""
    if regularization is None:
        params = np.linalg.pinv(X) @ y
    elif regularization == 'l2':
        params = _ridge_regression(X, y)
    else:
        raise ValueError(f"Regularization {regularization} not supported for closed form")
    metrics = {'mse': _mean_squared_error(y, X @ params)}
    return params, metrics

def _solve_gradient_descent(X: np.ndarray, y: np.ndarray,
                           metric_func: Callable[[np.ndarray, np.ndarray], float],
                           distance_func: Callable[[np.ndarray, np.ndarray], float],
                           regularization: Optional[str] = None,
                           tol: float = 1e-4, max_iter: int = 1000) -> tuple:
    """Solve using gradient descent."""
    params = np.zeros(X.shape[1])
    for _ in range(max_iter):
        gradient = _compute_gradient(X, y, params, metric_func,
                                   distance_func, regularization)
        params -= gradient
        if np.linalg.norm(gradient) < tol:
            break
    metrics = {'mse': _mean_squared_error(y, X @ params)}
    return params, metrics

def _solve_newton(X: np.ndarray, y: np.ndarray,
                 metric_func: Callable[[np.ndarray, np.ndarray], float],
                 distance_func: Callable[[np.ndarray, np.ndarray], float],
                 regularization: Optional[str] = None,
                 tol: float = 1e-4, max_iter: int = 1000) -> tuple:
    """Solve using Newton's method."""
    params = np.zeros(X.shape[1])
    for _ in range(max_iter):
        gradient = _compute_gradient(X, y, params, metric_func,
                                   distance_func, regularization)
        hessian = _compute_hessian(X, y, params, metric_func,
                                  distance_func, regularization)
        params -= np.linalg.pinv(hessian) @ gradient
        if np.linalg.norm(gradient) < tol:
            break
    metrics = {'mse': _mean_squared_error(y, X @ params)}
    return params, metrics

def _solve_coordinate_descent(X: np.ndarray, y: np.ndarray,
                             metric_func: Callable[[np.ndarray, np.ndarray], float],
                             distance_func: Callable[[np.ndarray, np.ndarray], float],
                             regularization: Optional[str] = None,
                             tol: float = 1e-4, max_iter: int = 1000) -> tuple:
    """Solve using coordinate descent."""
    params = np.zeros(X.shape[1])
    for _ in range(max_iter):
        for i in range(X.shape[1]):
            X_i = X[:, i]
            params[i] = _coordinate_descent_step(X_i, y,
                                               params, i, metric_func,
                                               distance_func, regularization)
        if np.linalg.norm(_compute_gradient(X, y, params,
                                          metric_func, distance_func,
                                          regularization)) < tol:
            break
    metrics = {'mse': _mean_squared_error(y, X @ params)}
    return params, metrics

def _compute_gradient(X: np.ndarray, y: np.ndarray,
                     params: np.ndarray,
                     metric_func: Callable[[np.ndarray, np.ndarray], float],
                     distance_func: Callable[[np.ndarray, np.ndarray], float],
                     regularization: Optional[str] = None) -> np.ndarray:
    """Compute gradient for optimization."""
    predictions = X @ params
    error = y - predictions
    if metric_func == _mean_squared_error:
        gradient = -2 * X.T @ error / len(y)
    elif metric_func == _mean_absolute_error:
        gradient = -np.sign(error) @ X / len(y)
    else:
        raise ValueError("Unsupported metric for gradient computation")

    if regularization == 'l2':
        gradient += 2 * params
    return gradient

def _compute_hessian(X: np.ndarray, y: np.ndarray,
                    params: np.ndarray,
                    metric_func: Callable[[np.ndarray, np.ndarray], float],
                    distance_func: Callable[[np.ndarray, np.ndarray], float],
                    regularization: Optional[str] = None) -> np.ndarray:
    """Compute Hessian matrix for optimization."""
    if metric_func != _mean_squared_error:
        raise ValueError("Hessian computation only supported for MSE")
    hessian = 2 * X.T @ X / len(y)
    if regularization == 'l2':
        hessian += 2 * np.eye(X.shape[1])
    return hessian

def _coordinate_descent_step(X_i: np.ndarray, y: np.ndarray,
                            params: np.ndarray, i: int,
                            metric_func: Callable[[np.ndarray, np.ndarray], float],
                            distance_func: Callable[[np.ndarray, np.ndarray], float],
                            regularization: Optional[str] = None) -> float:
    """Perform a single coordinate descent step."""
    X_i_neg = params.copy()
    X_i_pos = params.copy()

    X_i_neg[i] = 0
    X_i_pos[i] = params[i]

    if metric_func == _mean_squared_error:
        a_neg = X_i.T @ (y - X @ X_i_neg)
        b_neg = X_i.T @ X_i
    else:
        raise ValueError("Unsupported metric for coordinate descent")

    if regularization == 'l2':
        b_neg += 2

    return a_neg / b_neg if b_neg != 0 else 0

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

def _euclidean_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Euclidean distance."""
    return np.linalg.norm(X - Y)

def _manhattan_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Manhattan distance."""
    return np.sum(np.abs(X - Y))

def _cosine_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute cosine distance."""
    return 1 - np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y))

def _ridge_regression(X: np.ndarray, y: np.ndarray,
                     alpha: float = 1.0) -> np.ndarray:
    """Compute Ridge regression coefficients."""
    I = np.eye(X.shape[1])
    return np.linalg.pinv(X.T @ X + alpha * I) @ X.T @ y

def _check_warnings(X: np.ndarray, y: np.ndarray,
                   params: np.ndarray) -> list:
    """Check for potential warnings."""
    warnings = []
    if np.any(np.isnan(params)):
        warnings.append("Parameters contain NaN values")
    if np.any(np.isinf(params)):
        warnings.append("Parameters contain infinite values")
    return warnings

################################################################################
# model_distillation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> None:
    """
    Validate input data and normalizer function.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    normalizer : Optional[Callable]
        Normalization function to validate

    Raises
    ------
    ValueError
        If inputs are invalid or normalizer fails
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")

    if normalizer is not None:
        try:
            _ = normalizer(X)
        except Exception as e:
            raise ValueError(f"Normalizer function failed: {str(e)}")

def compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "mse",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> float:
    """
    Compute specified metric between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
    metric : str
        Metric name ("mse", "mae", "r2", "logloss")
    custom_metric : Optional[Callable]
        Custom metric function

    Returns
    ------
    float
        Computed metric value
    """
    if custom_metric is not None:
        return custom_metric(y_true, y_pred)

    if metric == "mse":
        return np.mean((y_true - y_pred) ** 2)
    elif metric == "mae":
        return np.mean(np.abs(y_true - y_pred))
    elif metric == "r2":
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    elif metric == "logloss":
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def estimate_parameters(
    X: np.ndarray,
    y: np.ndarray,
    solver: str = "closed_form",
    distance: str = "euclidean",
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **solver_params
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Estimate model parameters using specified solver.

    Parameters
    ----------
    X : np.ndarray
        Input features array
    y : np.ndarray
        Target values array
    solver : str
        Solver method ("closed_form", "gradient_descent", etc.)
    distance : str
        Distance metric for solver ("euclidean", "manhattan", etc.)
    custom_distance : Optional[Callable]
        Custom distance function
    **solver_params
        Additional solver parameters

    Returns
    ------
    Dict[str, Union[np.ndarray, float]]
        Dictionary containing estimated parameters and solver info
    """
    if custom_distance is not None:
        distance_func = custom_distance
    else:
        if distance == "euclidean":
            distance_func = lambda a, b: np.linalg.norm(a - b)
        elif distance == "manhattan":
            distance_func = lambda a, b: np.sum(np.abs(a - b))
        elif distance == "cosine":
            distance_func = lambda a, b: 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        else:
            raise ValueError(f"Unknown distance: {distance}")

    if solver == "closed_form":
        X_tx = np.dot(X.T, X)
        params = np.linalg.solve(X_tx + 1e-8 * np.eye(X.shape[1]), np.dot(X.T, y))
        return {"params": params, "solver": solver}
    elif solver == "gradient_descent":
        learning_rate = solver_params.get("learning_rate", 0.01)
        n_iter = solver_params.get("n_iter", 1000)
        tol = solver_params.get("tol", 1e-4)

        n_features = X.shape[1]
        params = np.zeros(n_features)
        prev_loss = float('inf')

        for _ in range(n_iter):
            grad = -2 * np.dot(X.T, y - np.dot(X, params)) / X.shape[0]
            params -= learning_rate * grad
            current_loss = np.mean((y - np.dot(X, params)) ** 2)

            if abs(prev_loss - current_loss) < tol:
                break
            prev_loss = current_loss

        return {"params": params, "solver": solver}
    else:
        raise ValueError(f"Unknown solver: {solver}")

def model_distillation_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = "mse",
    solver: str = "closed_form",
    distance: str = "euclidean",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **solver_params
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit a distilled model to the input data.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    normalizer : Optional[Callable]
        Normalization function
    metric : str
        Metric name ("mse", "mae", "r2", "logloss")
    solver : str
        Solver method ("closed_form", "gradient_descent", etc.)
    distance : str
        Distance metric for solver ("euclidean", "manhattan", etc.)
    custom_metric : Optional[Callable]
        Custom metric function
    custom_distance : Optional[Callable]
        Custom distance function
    **solver_params
        Additional solver parameters

    Returns
    ------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - "result": estimated parameters
        - "metrics": computed metrics
        - "params_used": solver and distance information
        - "warnings": any warnings generated

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = model_distillation_fit(X, y, solver="gradient_descent", learning_rate=0.1)
    """
    # Validate inputs
    validate_inputs(X, y, normalizer)

    # Apply normalization if specified
    X_normalized = X if normalizer is None else normalizer(X)

    # Estimate parameters
    params_result = estimate_parameters(
        X_normalized, y,
        solver=solver,
        distance=distance,
        custom_distance=custom_distance,
        **solver_params
    )

    # Compute predictions and metrics
    y_pred = np.dot(X_normalized, params_result["params"])
    main_metric = compute_metric(y, y_pred, metric=metric, custom_metric=custom_metric)

    # Compute additional metrics
    metrics = {
        "main": main_metric,
        "mse": compute_metric(y, y_pred, metric="mse"),
        "mae": compute_metric(y, y_pred, metric="mae")
    }

    return {
        "result": params_result["params"],
        "metrics": metrics,
        "params_used": {
            "solver": params_result["solver"],
            "distance": distance if custom_distance is None else "custom",
            **solver_params
        },
        "warnings": []
    }

################################################################################
# local_interpretable_model_agnostic_explanations
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[list] = None,
    sample_weight: Optional[np.ndarray] = None
) -> None:
    """
    Validate input data and parameters.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    feature_names : Optional[list]
        List of feature names
    sample_weight : Optional[np.ndarray]
        Sample weights

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if sample_weight is not None:
        if len(sample_weight) != X.shape[0]:
            raise ValueError("sample_weight must have the same length as X")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative values")

def _normalize_data(
    X: np.ndarray,
    y: np.ndarray,
    normalization: str = "standard",
    feature_names: Optional[list] = None
) -> tuple:
    """
    Normalize the input data.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    feature_names : Optional[list]
        List of feature names

    Returns
    -------
    tuple
        Normalized X, y and normalization parameters
    """
    if normalization == "none":
        return X, y, None

    n_samples, n_features = X.shape
    params = {}

    if normalization == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1.0
        X_normalized = (X - mean) / std
        params["mean"] = mean
        params["std"] = std

    elif normalization == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0
        X_normalized = (X - min_val) / range_val
        params["min"] = min_val
        params["max"] = max_val

    elif normalization == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        iqr[iqr == 0] = 1.0
        X_normalized = (X - median) / iqr
        params["median"] = median
        params["iqr"] = iqr

    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    return X_normalized, y, params

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    sample_weight: Optional[np.ndarray] = None
) -> float:
    """
    Compute the specified metric.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted target values
    metric : Union[str, Callable]
        Metric name or callable function
    sample_weight : Optional[np.ndarray]
        Sample weights

    Returns
    -------
    float
        Computed metric value
    """
    if callable(metric):
        return metric(y_true, y_pred, sample_weight=sample_weight)

    if metric == "mse":
        return np.mean((y_true - y_pred) ** 2)
    elif metric == "mae":
        return np.mean(np.abs(y_true - y_pred))
    elif metric == "r2":
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    elif metric == "logloss":
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _solve_lime(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    distance_metric: str = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    lambda_reg: float = 1.0,
    sample_weight: Optional[np.ndarray] = None
) -> tuple:
    """
    Solve the LIME problem.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    metric : Union[str, Callable]
        Metric to optimize
    distance_metric : str
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski')
    solver : str
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    regularization : Optional[str]
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    lambda_reg : float
        Regularization strength
    sample_weight : Optional[np.ndarray]
        Sample weights

    Returns
    -------
    tuple
        Coefficients and intercept of the local model
    """
    n_samples, n_features = X.shape

    if solver == "closed_form":
        # Closed form solution for weighted least squares
        if sample_weight is None:
            W = np.eye(n_samples)
        else:
            W = np.diag(sample_weight)

        if regularization == "l2":
            reg_matrix = lambda_reg * np.eye(n_features)
            XtWX = X.T @ W @ X
            XtWX += reg_matrix
        else:
            XtWX = X.T @ W @ X

        XtWy = X.T @ W @ y
        coef = np.linalg.solve(XtWX, XtWy)
        intercept = y.mean() if sample_weight is None else np.average(y, weights=sample_weight)

    elif solver == "gradient_descent":
        # Gradient descent implementation
        learning_rate = 0.01
        n_iterations = 1000
        tolerance = 1e-4

        coef = np.zeros(n_features)
        intercept = y.mean() if sample_weight is None else np.average(y, weights=sample_weight)

        for _ in range(n_iterations):
            y_pred = X @ coef + intercept
            gradient_coef = -2 * (X.T @ ((y - y_pred) * sample_weight)) / n_samples
            gradient_intercept = -2 * np.sum((y - y_pred) * sample_weight) / n_samples

            coef -= learning_rate * gradient_coef
            intercept -= learning_rate * gradient_intercept

    else:
        raise ValueError(f"Unknown solver: {solver}")

    return coef, intercept

def local_interpretable_model_agnostic_explanations_fit(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[list] = None,
    sample_weight: Optional[np.ndarray] = None,
    normalization: str = "standard",
    metric: Union[str, Callable] = "mse",
    distance_metric: str = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    lambda_reg: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> Dict[str, Any]:
    """
    Fit a local interpretable model-agnostic explanation.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    feature_names : Optional[list]
        List of feature names
    sample_weight : Optional[np.ndarray]
        Sample weights
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : Union[str, Callable]
        Metric to optimize
    distance_metric : str
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski')
    solver : str
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    regularization : Optional[str]
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    lambda_reg : float
        Regularization strength
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for stopping criterion

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - "result": dict with coefficients and intercept
        - "metrics": dict with computed metrics
        - "params_used": dict with parameters used
        - "warnings": list of warnings

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = local_interpretable_model_agnostic_explanations_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y, feature_names, sample_weight)

    # Normalize data
    X_norm, y_norm, norm_params = _normalize_data(X, y, normalization, feature_names)

    # Solve the LIME problem
    coef, intercept = _solve_lime(
        X_norm,
        y_norm,
        metric,
        distance_metric,
        solver,
        regularization,
        lambda_reg,
        sample_weight
    )

    # Compute predictions and metrics
    y_pred = X_norm @ coef + intercept
    main_metric = _compute_metric(y_norm, y_pred, metric, sample_weight)

    # Prepare output
    result = {
        "coef": coef,
        "intercept": intercept,
        "feature_names": feature_names
    }

    metrics = {
        "main_metric": main_metric,
        "metric_name": metric if not callable(metric) else "custom"
    }

    params_used = {
        "normalization": normalization,
        "metric": metric if not callable(metric) else "custom",
        "distance_metric": distance_metric,
        "solver": solver,
        "regularization": regularization,
        "lambda_reg": lambda_reg
    }

    warnings = []

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

################################################################################
# global_surrogate_models
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
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

def _normalize_data(X: np.ndarray, y: np.ndarray,
                   normalization: str = 'standard',
                   custom_normalization: Optional[Callable] = None) -> tuple:
    """Normalize input data based on specified method."""
    if custom_normalization is not None:
        X_norm = custom_normalization(X)
        y_norm = custom_normalization(y.reshape(-1, 1)).flatten()
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

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metrics: Union[str, list] = 'mse',
                    custom_metric: Optional[Callable] = None) -> Dict[str, float]:
    """Compute specified metrics between true and predicted values."""
    metric_results = {}

    if isinstance(metrics, str):
        metrics = [metrics]

    for metric in metrics:
        if metric == 'mse':
            metric_results['mse'] = np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            metric_results['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metric_results['r2'] = 1 - (ss_res / ss_tot)
        elif metric == 'logloss':
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            metric_results['logloss'] = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        elif custom_metric is not None and metric == 'custom':
            metric_results['custom'] = custom_metric(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return metric_results

def _fit_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit model using closed form solution (linear regression)."""
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
    coefficients = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
    return coefficients

def _fit_gradient_descent(X: np.ndarray, y: np.ndarray,
                         learning_rate: float = 0.01,
                         n_iterations: int = 1000,
                         tol: float = 1e-4) -> np.ndarray:
    """Fit model using gradient descent."""
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features + 1)
    X_with_intercept = np.column_stack([np.ones(n_samples), X])

    for _ in range(n_iterations):
        gradient = 2 * (X_with_intercept @ coefficients - y) @ X_with_intercept
        new_coefficients = coefficients - learning_rate * gradient

        if np.linalg.norm(new_coefficients - coefficients) < tol:
            break

        coefficients = new_coefficients

    return coefficients

def _fit_newton(X: np.ndarray, y: np.ndarray,
               tol: float = 1e-4,
               max_iterations: int = 100) -> np.ndarray:
    """Fit model using Newton's method."""
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features + 1)
    X_with_intercept = np.column_stack([np.ones(n_samples), X])

    for _ in range(max_iterations):
        residuals = X_with_intercept @ coefficients - y
        gradient = 2 * (X_with_intercept.T @ residuals)
        hessian = 2 * X_with_intercept.T @ X_with_intercept
        delta = np.linalg.solve(hessian, gradient)
        coefficients -= delta

        if np.linalg.norm(delta) < tol:
            break

    return coefficients

def _fit_coordinate_descent(X: np.ndarray, y: np.ndarray,
                           n_iterations: int = 1000,
                           tol: float = 1e-4) -> np.ndarray:
    """Fit model using coordinate descent."""
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features + 1)
    X_with_intercept = np.column_stack([np.ones(n_samples), X])

    for _ in range(n_iterations):
        old_coefficients = coefficients.copy()

        for i in range(coefficients.shape[0]):
            X_i = X_with_intercept[:, i]
            residuals = y - (X_with_intercept @ coefficients)
            numerator = X_i.T @ residuals
            denominator = X_i.T @ X_i
            coefficients[i] += numerator / denominator

        if np.linalg.norm(coefficients - old_coefficients) < tol:
            break

    return coefficients

def global_surrogate_models_fit(X: np.ndarray,
                              y: np.ndarray,
                              normalization: str = 'standard',
                              metrics: Union[str, list] = 'mse',
                              solver: str = 'closed_form',
                              learning_rate: float = 0.01,
                              n_iterations: int = 1000,
                              tol: float = 1e-4,
                              custom_normalization: Optional[Callable] = None,
                              custom_metric: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Fit global surrogate models to explain complex black-box models.

    Parameters:
    -----------
    X : np.ndarray
        Input features (n_samples, n_features)
    y : np.ndarray
        Target values (n_samples,)
    normalization : str or callable, optional
        Normalization method ('standard', 'minmax', 'robust') or custom function
    metrics : str, list of str, or callable, optional
        Metrics to compute ('mse', 'mae', 'r2', 'logloss') or custom function
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    learning_rate : float, optional
        Learning rate for gradient descent (default: 0.01)
    n_iterations : int, optional
        Maximum number of iterations (default: 1000)
    tol : float, optional
        Tolerance for convergence (default: 1e-4)
    custom_normalization : callable, optional
        Custom normalization function
    custom_metric : callable, optional
        Custom metric function

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': fitted model coefficients
        - 'metrics': computed metrics
        - 'params_used': parameters used for fitting
        - 'warnings': any warnings during execution

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = global_surrogate_models_fit(X, y,
    ...                                     normalization='standard',
    ...                                     metrics=['mse', 'r2'],
    ...                                     solver='gradient_descent')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = _normalize_data(X, y, normalization, custom_normalization)

    # Fit model based on solver choice
    if solver == 'closed_form':
        coefficients = _fit_closed_form(X_norm, y_norm)
    elif solver == 'gradient_descent':
        coefficients = _fit_gradient_descent(X_norm, y_norm,
                                           learning_rate=learning_rate,
                                           n_iterations=n_iterations,
                                           tol=tol)
    elif solver == 'newton':
        coefficients = _fit_newton(X_norm, y_norm,
                                 tol=tol)
    elif solver == 'coordinate_descent':
        coefficients = _fit_coordinate_descent(X_norm, y_norm,
                                             n_iterations=n_iterations,
                                             tol=tol)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute predictions
    X_with_intercept = np.column_stack([np.ones(X_norm.shape[0]), X_norm])
    y_pred = X_with_intercept @ coefficients

    # Compute metrics
    metric_results = _compute_metrics(y_norm, y_pred,
                                    metrics=metrics,
                                    custom_metric=custom_metric)

    # Prepare output
    result = {
        'result': coefficients,
        'metrics': metric_results,
        'params_used': {
            'normalization': normalization if custom_normalization is None else 'custom',
            'metrics': metrics,
            'solver': solver,
            'learning_rate': learning_rate if solver == 'gradient_descent' else None,
            'n_iterations': n_iterations,
            'tol': tol
        },
        'warnings': []
    }

    return result

################################################################################
# decision_boundary_visualization
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def decision_boundary_visualization_fit(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    normalization: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Visualize decision boundaries for a given model.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    model : Callable
        The model to fit and visualize.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or Callable, optional
        Metric to evaluate the model ('mse', 'mae', 'r2', 'logloss').
    distance : str, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski').
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet').
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : Callable, optional
        Custom metric function.
    custom_distance : Callable, optional
        Custom distance function.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _apply_normalization(X, normalization)

    # Initialize model parameters
    params = _initialize_params(model, X_normalized.shape[1])

    # Fit model using specified solver
    fitted_model = _fit_model(
        X_normalized, y, params, model,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter
    )

    # Compute metrics
    metrics = _compute_metrics(
        X_normalized, y, fitted_model,
        metric=metric,
        distance=distance,
        custom_metric=custom_metric
    )

    # Prepare output
    result = {
        "result": fitted_model,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": _check_warnings(X_normalized, y)
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
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

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to the data."""
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

def _initialize_params(model: Callable, n_features: int) -> Dict[str, Any]:
    """Initialize model parameters."""
    return model.initialize(n_features)

def _fit_model(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any],
    model: Callable,
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> Dict[str, Any]:
    """Fit the model using specified solver."""
    if solver == 'closed_form':
        return _fit_closed_form(X, y, params, model)
    elif solver == 'gradient_descent':
        return _fit_gradient_descent(X, y, params, model, tol, max_iter, regularization)
    elif solver == 'newton':
        return _fit_newton(X, y, params, model, tol, max_iter)
    elif solver == 'coordinate_descent':
        return _fit_coordinate_descent(X, y, params, model, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _fit_closed_form(X: np.ndarray, y: np.ndarray, params: Dict[str, Any], model: Callable) -> Dict[str, Any]:
    """Fit model using closed-form solution."""
    return model.fit_closed_form(X, y, params)

def _fit_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any],
    model: Callable,
    tol: float = 1e-4,
    max_iter: int = 1000,
    regularization: Optional[str] = None
) -> Dict[str, Any]:
    """Fit model using gradient descent."""
    return model.fit_gradient_descent(X, y, params, tol, max_iter, regularization)

def _fit_newton(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any],
    model: Callable,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> Dict[str, Any]:
    """Fit model using Newton's method."""
    return model.fit_newton(X, y, params, tol, max_iter)

def _fit_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any],
    model: Callable,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> Dict[str, Any]:
    """Fit model using coordinate descent."""
    return model.fit_coordinate_descent(X, y, params, tol, max_iter)

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    model: Dict[str, Any],
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """Compute specified metrics."""
    if custom_metric is not None:
        return {"custom_metric": custom_metric(X, y, model)}

    if metric == 'mse':
        return {"mse": _compute_mse(X, y, model)}
    elif metric == 'mae':
        return {"mae": _compute_mae(X, y, model)}
    elif metric == 'r2':
        return {"r2": _compute_r2(X, y, model)}
    elif metric == 'logloss':
        return {"logloss": _compute_logloss(X, y, model)}
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _compute_mse(X: np.ndarray, y: np.ndarray, model: Dict[str, Any]) -> float:
    """Compute Mean Squared Error."""
    y_pred = model.predict(X)
    return np.mean((y - y_pred) ** 2)

def _compute_mae(X: np.ndarray, y: np.ndarray, model: Dict[str, Any]) -> float:
    """Compute Mean Absolute Error."""
    y_pred = model.predict(X)
    return np.mean(np.abs(y - y_pred))

def _compute_r2(X: np.ndarray, y: np.ndarray, model: Dict[str, Any]) -> float:
    """Compute R-squared."""
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

def _compute_logloss(X: np.ndarray, y: np.ndarray, model: Dict[str, Any]) -> float:
    """Compute Log Loss."""
    y_pred = model.predict_proba(X)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def _check_warnings(X: np.ndarray, y: np.ndarray) -> list:
    """Check for potential warnings."""
    warnings = []
    if np.any(np.std(X, axis=0) == 0):
        warnings.append("Some features have zero variance.")
    return warnings

################################################################################
# attention_mechanisms
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    attention_weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate input arrays and return validated versions.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,) or (n_samples, n_outputs)
    attention_weights : np.ndarray, optional
        Precomputed attention weights of shape (n_samples, n_features)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Validated X and y arrays

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim not in (1, 2):
        raise ValueError("y must be a 1D or 2D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")

    if attention_weights is not None:
        if attention_weights.shape != X.shape:
            raise ValueError("attention_weights must have the same shape as X")
        if np.any(np.isnan(attention_weights)) or np.any(np.isinf(attention_weights)):
            raise ValueError("attention_weights contains NaN or Inf values")

    return X, y

def _normalize_data(
    data: np.ndarray,
    method: str = 'standard',
    custom_func: Optional[Callable] = None
) -> np.ndarray:
    """
    Normalize data using specified method or custom function.

    Parameters
    ----------
    data : np.ndarray
        Input data to normalize
    method : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    custom_func : Callable, optional
        Custom normalization function

    Returns
    -------
    np.ndarray
        Normalized data
    """
    if custom_func is not None:
        return custom_func(data)

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
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_attention_scores(
    X: np.ndarray,
    y: np.ndarray,
    metric: str = 'mse',
    custom_metric: Optional[Callable] = None
) -> np.ndarray:
    """
    Compute attention scores based on specified metric.

    Parameters
    ----------
    X : np.ndarray
        Input features array
    y : np.ndarray
        Target values array
    metric : str, optional
        Metric to use ('mse', 'mae', 'r2', 'logloss')
    custom_metric : Callable, optional
        Custom metric function

    Returns
    -------
    np.ndarray
        Attention scores array
    """
    if custom_metric is not None:
        return custom_metric(X, y)

    n_samples = X.shape[0]
    attention_scores = np.zeros((n_samples, X.shape[1]))

    for i in range(n_samples):
        if metric == 'mse':
            residuals = y - X[i]
            attention_scores[i] = np.mean(residuals**2, axis=0)
        elif metric == 'mae':
            residuals = np.abs(y - X[i])
            attention_scores[i] = np.mean(residuals, axis=0)
        elif metric == 'r2':
            residuals = y - X[i]
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            attention_scores[i] = 1 - (ss_res / (ss_tot + 1e-8))
        elif metric == 'logloss':
            # Assuming binary classification for log loss
            y_pred = X[i]
            y_pred = 1 / (1 + np.exp(-y_pred))
            attention_scores[i] = -np.mean(y * np.log(y_pred + 1e-8) +
                                          (1 - y) * np.log(1 - y_pred + 1e-8))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return attention_scores

def _normalize_attention_weights(
    weights: np.ndarray,
    method: str = 'softmax'
) -> np.ndarray:
    """
    Normalize attention weights using specified method.

    Parameters
    ----------
    weights : np.ndarray
        Attention weights to normalize
    method : str, optional
        Normalization method ('softmax', 'l1', 'l2')

    Returns
    -------
    np.ndarray
        Normalized attention weights
    """
    if method == 'softmax':
        exp_weights = np.exp(weights - np.max(weights, axis=1, keepdims=True))
        return exp_weights / (np.sum(exp_weights, axis=1, keepdims=True) + 1e-8)
    elif method == 'l1':
        return weights / (np.sum(np.abs(weights), axis=1, keepdims=True) + 1e-8)
    elif method == 'l2':
        return weights / (np.sqrt(np.sum(weights**2, axis=1, keepdims=True)) + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def attention_mechanisms_fit(
    X: np.ndarray,
    y: np.ndarray,
    attention_weights: Optional[np.ndarray] = None,
    normalize_X: str = 'standard',
    normalize_y: str = 'none',
    metric: str = 'mse',
    weight_normalization: str = 'softmax',
    custom_metric: Optional[Callable] = None,
    custom_normalize_X: Optional[Callable] = None,
    custom_normalize_y: Optional[Callable] = None
) -> Dict:
    """
    Fit attention mechanisms to data.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,) or (n_samples, n_outputs)
    attention_weights : np.ndarray, optional
        Precomputed attention weights of shape (n_samples, n_features)
    normalize_X : str, optional
        Normalization method for X ('none', 'standard', 'minmax', 'robust')
    normalize_y : str, optional
        Normalization method for y ('none', 'standard', 'minmax', 'robust')
    metric : str, optional
        Metric to use for attention scores ('mse', 'mae', 'r2', 'logloss')
    weight_normalization : str, optional
        Normalization method for attention weights ('softmax', 'l1', 'l2')
    custom_metric : Callable, optional
        Custom metric function
    custom_normalize_X : Callable, optional
        Custom normalization function for X
    custom_normalize_y : Callable, optional
        Custom normalization function for y

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Computed attention weights
        - 'metrics': Dictionary of computed metrics
        - 'params_used': Parameters used in the computation
        - 'warnings': List of warnings encountered

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = attention_mechanisms_fit(X, y)
    """
    # Validate inputs
    X_valid, y_valid = _validate_inputs(X, y, attention_weights)

    # Normalize data
    X_norm = _normalize_data(X_valid, normalize_X, custom_normalize_X)
    y_norm = _normalize_data(y_valid, normalize_y, custom_normalize_y)

    warnings = []

    # Compute attention scores
    if attention_weights is None:
        attention_scores = _compute_attention_scores(X_norm, y_norm, metric, custom_metric)
    else:
        attention_scores = attention_weights

    # Normalize attention weights
    attention_weights_final = _normalize_attention_weights(attention_scores, weight_normalization)

    # Compute metrics
    metrics = {}
    if metric == 'mse':
        residuals = y_norm - np.dot(X_norm, attention_weights_final.T)
        metrics['mse'] = np.mean(residuals**2)
    elif metric == 'mae':
        residuals = np.abs(y_norm - np.dot(X_norm, attention_weights_final.T))
        metrics['mae'] = np.mean(residuals)
    elif metric == 'r2':
        ss_res = np.sum((y_norm - np.dot(X_norm, attention_weights_final.T))**2)
        ss_tot = np.sum((y_norm - np.mean(y_norm))**2)
        metrics['r2'] = 1 - (ss_res / (ss_tot + 1e-8))
    elif metric == 'logloss':
        y_pred = np.dot(X_norm, attention_weights_final.T)
        y_pred = 1 / (1 + np.exp(-y_pred))
        metrics['logloss'] = -np.mean(y_norm * np.log(y_pred + 1e-8) +
                                     (1 - y_norm) * np.log(1 - y_pred + 1e-8))

    # Check for NaN/Inf in results
    if np.any(np.isnan(attention_weights_final)) or np.any(np.isinf(attention_weights_final)):
        warnings.append("Attention weights contain NaN or Inf values")

    return {
        'result': attention_weights_final,
        'metrics': metrics,
        'params_used': {
            'normalize_X': normalize_X if custom_normalize_X is None else 'custom',
            'normalize_y': normalize_y if custom_normalize_y is None else 'custom',
            'metric': metric if custom_metric is None else 'custom',
            'weight_normalization': weight_normalization
        },
        'warnings': warnings
    }

################################################################################
# gradient_based_explanations
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate input arrays for gradient-based explanations.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("Inputs must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Inputs contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Inputs contain infinite values")

def normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Normalize feature matrix using specified method.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix to normalize
    method : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')

    Returns
    ------
    np.ndarray
        Normalized feature matrix
    """
    if method == 'none':
        return X

    X_norm = X.copy()
    n_samples, n_features = X.shape

    if method == 'standard':
        means = np.mean(X_norm, axis=0)
        stds = np.std(X_norm, axis=0)
        X_norm = (X_norm - means) / stds

    elif method == 'minmax':
        mins = np.min(X_norm, axis=0)
        maxs = np.max(X_norm, axis=0)
        X_norm = (X_norm - mins) / (maxs - mins)

    elif method == 'robust':
        medians = np.median(X_norm, axis=0)
        iqrs = np.subtract(*np.percentile(X_norm, [75, 25], axis=0))
        X_norm = (X_norm - medians) / iqrs

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_norm

def compute_gradient(X: np.ndarray, y: np.ndarray,
                    loss_func: Callable = None) -> np.ndarray:
    """
    Compute gradient of the specified loss function.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    loss_func : Callable, optional
        Loss function to use. If None, uses MSE by default.

    Returns
    ------
    np.ndarray
        Gradient of the loss function with respect to model parameters
    """
    if loss_func is None:
        def mse_loss(y_true, y_pred):
            return 0.5 * np.sum((y_true - y_pred) ** 2)

        def mse_gradient(y_true, y_pred):
            return (y_pred - y_true) @ X

        loss_func = mse_loss
        gradient_func = mse_gradient
    else:
        # For custom loss functions, we need to compute the gradient numerically
        def numerical_gradient(y_true, y_pred):
            epsilon = 1e-8
            grad = np.zeros(X.shape[1])
            for i in range(X.shape[1]):
                theta_plus = np.zeros_like(y_pred)
                theta_minus = np.zeros_like(y_pred)

                # Perturb each parameter slightly
                X_plus = X.copy()
                X_minus = X.copy()
                X_plus[:, i] += epsilon
                X_minus[:, i] -= epsilon

                # Compute loss for both perturbations
                y_plus = np.dot(X_plus, theta_plus)
                y_minus = np.dot(X_minus, theta_minus)

                # Numerical gradient
                grad[i] = (loss_func(y_true, y_plus) - loss_func(y_true, y_minus)) / (2 * epsilon)
            return grad

        gradient_func = numerical_gradient

    # Initialize parameters
    theta = np.zeros(X.shape[1])
    y_pred = np.dot(X, theta)

    return gradient_func(y, y_pred)

def optimize_parameters(X: np.ndarray, y: np.ndarray,
                      solver: str = 'gradient_descent',
                      loss_func: Callable = None,
                      max_iter: int = 1000,
                      tol: float = 1e-4,
                      learning_rate: float = 0.01) -> np.ndarray:
    """
    Optimize model parameters using specified solver.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    solver : str, optional
        Optimization method ('gradient_descent', 'newton')
    loss_func : Callable, optional
        Loss function to use. If None, uses MSE by default.
    max_iter : int, optional
        Maximum number of iterations
    tol : float, optional
        Tolerance for convergence
    learning_rate : float, optional
        Learning rate for gradient descent

    Returns
    ------
    np.ndarray
        Optimized model parameters
    """
    if solver == 'gradient_descent':
        theta = np.zeros(X.shape[1])
        prev_loss = float('inf')

        for _ in range(max_iter):
            gradient = compute_gradient(X, y, loss_func)
            theta -= learning_rate * gradient
            current_loss = np.mean(loss_func(y, np.dot(X, theta)))

            if abs(prev_loss - current_loss) < tol:
                break
            prev_loss = current_loss

    elif solver == 'newton':
        # Newton's method implementation would go here
        raise NotImplementedError("Newton solver not yet implemented")

    else:
        raise ValueError(f"Unknown solver: {solver}")

    return theta

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   metric_funcs: Dict[str, Callable] = None) -> Dict[str, float]:
    """
    Compute specified metrics between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
    metric_funcs : Dict[str, Callable], optional
        Dictionary of metric functions to compute

    Returns
    ------
    Dict[str, float]
        Computed metrics
    """
    if metric_funcs is None:
        metric_funcs = {
            'mse': lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
            'mae': lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)),
            'r2': lambda y_true, y_pred: 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        }

    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(y_true, y_pred)
        except Exception as e:
            metrics[name] = np.nan
            print(f"Warning: Could not compute metric {name}: {str(e)}")

    return metrics

def gradient_based_explanations_fit(X: np.ndarray, y: np.ndarray,
                                 normalization: str = 'standard',
                                 solver: str = 'gradient_descent',
                                 loss_func: Callable = None,
                                 metric_funcs: Dict[str, Callable] = None,
                                 max_iter: int = 1000,
                                 tol: float = 1e-4,
                                 learning_rate: float = 0.01) -> Dict:
    """
    Fit gradient-based explanation model to data.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    solver : str, optional
        Optimization method ('gradient_descent', 'newton')
    loss_func : Callable, optional
        Loss function to use. If None, uses MSE by default.
    metric_funcs : Dict[str, Callable], optional
        Dictionary of metric functions to compute
    max_iter : int, optional
        Maximum number of iterations for optimization
    tol : float, optional
        Tolerance for convergence
    learning_rate : float, optional
        Learning rate for gradient descent

    Returns
    ------
    Dict
        Dictionary containing:
        - 'result': Optimized parameters
        - 'metrics': Computed metrics
        - 'params_used': Parameters used in the fitting process
        - 'warnings': Any warnings generated during fitting

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = gradient_based_explanations_fit(X, y,
    ...                                         normalization='standard',
    ...                                         solver='gradient_descent')
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_norm = normalize_data(X, normalization)

    # Optimize parameters
    theta = optimize_parameters(X_norm, y,
                              solver=solver,
                              loss_func=loss_func,
                              max_iter=max_iter,
                              tol=tol,
                              learning_rate=learning_rate)

    # Compute predictions
    y_pred = np.dot(X_norm, theta)

    # Compute metrics
    metrics = compute_metrics(y, y_pred, metric_funcs)

    return {
        'result': theta,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'solver': solver,
            'max_iter': max_iter,
            'tol': tol,
            'learning_rate': learning_rate
        },
        'warnings': []
    }
