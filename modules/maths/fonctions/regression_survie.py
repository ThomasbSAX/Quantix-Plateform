"""
Quantix – Module regression_survie
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# censorship
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def censorship_fit(
    times: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: str = "mse",
    distance: str = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, str]]:
    """
    Fit a censorship model for survival analysis.

    Parameters
    ----------
    times : np.ndarray
        Array of observed times.
    events : np.ndarray
        Binary array indicating whether the event occurred (1) or was censored (0).
    covariates : np.ndarray
        Matrix of covariates.
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the covariates. Default is identity.
    metric : str
        Metric to evaluate model performance: 'mse', 'mae', 'r2', or custom.
    distance : str
        Distance metric: 'euclidean', 'manhattan', 'cosine', or custom.
    solver : str
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'.
    regularization : Optional[str]
        Regularization type: None, 'l1', 'l2', or 'elasticnet'.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom distance function.

    Returns
    -------
    Dict[str, Union[Dict, float, str]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> times = np.array([1.2, 2.3, 3.4])
    >>> events = np.array([1, 0, 1])
    >>> covariates = np.random.rand(3, 2)
    >>> result = censorship_fit(times, events, covariates)
    """
    # Validate inputs
    _validate_inputs(times, events, covariates)

    # Normalize covariates
    X = normalizer(covariates)
    y = times.copy()
    e = events.copy()

    # Choose metric
    if custom_metric is not None:
        metric_func = custom_metric
    else:
        metric_func = _get_metric(metric)

    # Choose distance
    if custom_distance is not None:
        distance_func = custom_distance
    else:
        distance_func = _get_distance(distance)

    # Choose solver
    if solver == "closed_form":
        params = _solve_closed_form(X, y, e)
    elif solver == "gradient_descent":
        params = _solve_gradient_descent(X, y, e, tol, max_iter)
    elif solver == "newton":
        params = _solve_newton(X, y, e, tol, max_iter)
    elif solver == "coordinate_descent":
        params = _solve_coordinate_descent(X, y, e, tol, max_iter)
    else:
        raise ValueError("Invalid solver specified.")

    # Apply regularization if needed
    if regularization is not None:
        params = _apply_regularization(params, X, y, e, regularization)

    # Calculate metrics
    predictions = _predict(X, params)
    metrics = {
        "metric": metric_func(y[e == 1], predictions[e == 1]),
        "distance": distance_func(y, predictions)
    }

    # Prepare output
    result = {
        "result": params,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if hasattr(normalizer, '__name__') else "custom",
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularization": regularization
        },
        "warnings": []
    }

    return result

def _validate_inputs(times: np.ndarray, events: np.ndarray, covariates: np.ndarray) -> None:
    """Validate input arrays."""
    if times.shape[0] != events.shape[0]:
        raise ValueError("times and events must have the same length.")
    if covariates.shape[0] != times.shape[0]:
        raise ValueError("covariates must have the same number of rows as times.")
    if np.any(np.isnan(times)) or np.any(np.isinf(times)):
        raise ValueError("times contains NaN or inf values.")
    if np.any(np.isnan(events)) or np.any(np.isinf(events)):
        raise ValueError("events contains NaN or inf values.")
    if np.any(np.isnan(covariates)) or np.any(np.isinf(covariates)):
        raise ValueError("covariates contains NaN or inf values.")

def _get_metric(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the specified metric function."""
    metrics = {
        "mse": _mean_squared_error,
        "mae": _mean_absolute_error,
        "r2": _r_squared
    }
    if metric not in metrics:
        raise ValueError(f"Invalid metric: {metric}")
    return metrics[metric]

def _get_distance(distance: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the specified distance function."""
    distances = {
        "euclidean": _euclidean_distance,
        "manhattan": _manhattan_distance,
        "cosine": _cosine_distance
    }
    if distance not in distances:
        raise ValueError(f"Invalid distance: {distance}")
    return distances[distance]

def _solve_closed_form(X: np.ndarray, y: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Solve using closed-form solution."""
    # Placeholder for actual implementation
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    e: np.ndarray,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve using gradient descent."""
    # Placeholder for actual implementation
    return np.zeros(X.shape[1])

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    e: np.ndarray,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve using Newton's method."""
    # Placeholder for actual implementation
    return np.zeros(X.shape[1])

def _solve_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    e: np.ndarray,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve using coordinate descent."""
    # Placeholder for actual implementation
    return np.zeros(X.shape[1])

def _apply_regularization(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    e: np.ndarray,
    regularization: str
) -> np.ndarray:
    """Apply specified regularization."""
    # Placeholder for actual implementation
    return params

def _predict(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Make predictions."""
    return X @ params

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

def _euclidean_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Euclidean distance."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def _manhattan_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Manhattan distance."""
    return np.mean(np.abs(y_true - y_pred))

def _cosine_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate cosine distance."""
    return 1 - (np.dot(y_true, y_pred) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred)))

################################################################################
# hazard_function
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def hazard_function_fit(
    time: np.ndarray,
    event: np.ndarray,
    features: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = "mse",
    distance: str = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, np.ndarray]]:
    """
    Fit a hazard function model for survival analysis.

    Parameters
    ----------
    time : np.ndarray
        Array of observed times.
    event : np.ndarray
        Binary array indicating if the event occurred (1) or was censored (0).
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize features. Default is None.
    metric : str, optional
        Metric to evaluate the model. Options: "mse", "mae", "r2", "logloss". Default is "mse".
    distance : str, optional
        Distance metric for the solver. Options: "euclidean", "manhattan", "cosine", "minkowski". Default is "euclidean".
    solver : str, optional
        Solver to use. Options: "closed_form", "gradient_descent", "newton", "coordinate_descent". Default is "closed_form".
    regularization : Optional[str], optional
        Regularization type. Options: None, "l1", "l2", "elasticnet". Default is None.
    tol : float, optional
        Tolerance for convergence. Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function. Default is None.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom distance function. Default is None.

    Returns
    -------
    Dict[str, Union[Dict, np.ndarray]]
        Dictionary containing the result, metrics, parameters used, and warnings.

    Examples
    --------
    >>> time = np.array([1.0, 2.0, 3.0])
    >>> event = np.array([1, 0, 1])
    >>> features = np.random.rand(3, 2)
    >>> result = hazard_function_fit(time, event, features)
    """
    # Validate inputs
    _validate_inputs(time, event, features)

    # Normalize features if a normalizer is provided
    if normalizer is not None:
        features = normalizer(features)

    # Choose the appropriate solver
    if solver == "closed_form":
        params = _solve_closed_form(time, event, features)
    elif solver == "gradient_descent":
        params = _solve_gradient_descent(time, event, features, distance, tol, max_iter)
    elif solver == "newton":
        params = _solve_newton(time, event, features, tol, max_iter)
    elif solver == "coordinate_descent":
        params = _solve_coordinate_descent(time, event, features, tol, max_iter)
    else:
        raise ValueError("Invalid solver specified.")

    # Apply regularization if specified
    if regularization is not None:
        params = _apply_regularization(params, features, regularization)

    # Compute metrics
    metrics = _compute_metrics(time, event, features, params, metric, custom_metric)

    # Prepare the result dictionary
    result = {
        "result": params,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    return result

def _validate_inputs(time: np.ndarray, event: np.ndarray, features: np.ndarray) -> None:
    """Validate the input arrays."""
    if time.shape[0] != event.shape[0]:
        raise ValueError("Time and event arrays must have the same length.")
    if features.shape[0] != time.shape[0]:
        raise ValueError("Number of samples in features must match the number of time points.")
    if np.any(np.isnan(time)) or np.any(np.isinf(time)):
        raise ValueError("Time array contains NaN or infinite values.")
    if np.any(np.isnan(event)) or np.any(np.isinf(event)):
        raise ValueError("Event array contains NaN or infinite values.")
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        raise ValueError("Features array contains NaN or infinite values.")

def _solve_closed_form(time: np.ndarray, event: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Solve the hazard function using closed-form solution."""
    # Implement closed-form solution logic here
    return np.zeros(features.shape[1])

def _solve_gradient_descent(
    time: np.ndarray,
    event: np.ndarray,
    features: np.ndarray,
    distance: str,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve the hazard function using gradient descent."""
    # Implement gradient descent logic here
    return np.zeros(features.shape[1])

def _solve_newton(
    time: np.ndarray,
    event: np.ndarray,
    features: np.ndarray,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve the hazard function using Newton's method."""
    # Implement Newton's method logic here
    return np.zeros(features.shape[1])

def _solve_coordinate_descent(
    time: np.ndarray,
    event: np.ndarray,
    features: np.ndarray,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve the hazard function using coordinate descent."""
    # Implement coordinate descent logic here
    return np.zeros(features.shape[1])

def _apply_regularization(
    params: np.ndarray,
    features: np.ndarray,
    regularization: str
) -> np.ndarray:
    """Apply regularization to the parameters."""
    # Implement regularization logic here
    return params

def _compute_metrics(
    time: np.ndarray,
    event: np.ndarray,
    features: np.ndarray,
    params: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute the metrics for the hazard function."""
    # Implement metric computation logic here
    return {"metric": 0.0}

################################################################################
# survival_function
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def survival_function_fit(
    times: np.ndarray,
    events: np.ndarray,
    predictors: Optional[np.ndarray] = None,
    normalizer: str = "standard",
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
    Fit a survival function model to the given data.

    Parameters:
    -----------
    times : np.ndarray
        Array of survival times.
    events : np.ndarray
        Binary array indicating whether the event occurred (1) or was censored (0).
    predictors : Optional[np.ndarray]
        Array of predictor variables. If None, intercept-only model is fit.
    normalizer : str
        Normalization method: "none", "standard", "minmax", or "robust".
    metric : Union[str, Callable]
        Metric to optimize: "mse", "mae", "r2", or custom callable.
    distance : str
        Distance metric: "euclidean", "manhattan", "cosine", or custom callable.
    solver : str
        Solver method: "closed_form", "gradient_descent", "newton", or "coordinate_descent".
    regularization : Optional[str]
        Regularization type: "none", "l1", "l2", or "elasticnet".
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable]
        Custom metric function if not using built-in metrics.
    custom_distance : Optional[Callable]
        Custom distance function if not using built-in distances.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(times, events, predictors)

    # Normalize data if needed
    normalized_predictors = _normalize(predictors, normalizer) if predictors is not None else None

    # Choose metric
    metric_func = _get_metric(metric, custom_metric)

    # Choose distance
    distance_func = _get_distance(distance, custom_distance)

    # Fit model based on solver choice
    if solver == "closed_form":
        params = _fit_closed_form(normalized_predictors, times, events)
    elif solver == "gradient_descent":
        params = _fit_gradient_descent(
            normalized_predictors, times, events,
            metric_func, distance_func, regularization,
            tol, max_iter
        )
    elif solver == "newton":
        params = _fit_newton(
            normalized_predictors, times, events,
            metric_func, distance_func, regularization,
            tol, max_iter
        )
    elif solver == "coordinate_descent":
        params = _fit_coordinate_descent(
            normalized_predictors, times, events,
            metric_func, distance_func, regularization,
            tol, max_iter
        )
    else:
        raise ValueError("Invalid solver specified")

    # Calculate metrics
    metrics = _calculate_metrics(params, normalized_predictors, times, events, metric_func)

    return {
        "result": params,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer,
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularization": regularization
        },
        "warnings": _check_warnings(params, metrics)
    }

def _validate_inputs(times: np.ndarray, events: np.ndarray, predictors: Optional[np.ndarray]) -> None:
    """Validate input arrays."""
    if times.ndim != 1 or events.ndim != 1:
        raise ValueError("times and events must be 1-dimensional arrays")
    if len(times) != len(events):
        raise ValueError("times and events must have the same length")
    if predictors is not None:
        if predictors.ndim != 2 or predictors.shape[0] != len(times):
            raise ValueError("predictors must be 2D array with n_samples rows")
        if np.any(np.isnan(times)) or np.any(np.isinf(times)):
            raise ValueError("times contains NaN or inf values")
        if np.any(np.isnan(events)) or np.any(np.isinf(events)):
            raise ValueError("events contains NaN or inf values")
        if np.any(np.isnan(predictors)) or np.any(np.isinf(predictors)):
            raise ValueError("predictors contains NaN or inf values")

def _normalize(predictors: np.ndarray, method: str) -> np.ndarray:
    """Normalize predictor variables."""
    if method == "none":
        return predictors
    elif method == "standard":
        mean = np.mean(predictors, axis=0)
        std = np.std(predictors, axis=0)
        return (predictors - mean) / std
    elif method == "minmax":
        min_val = np.min(predictors, axis=0)
        max_val = np.max(predictors, axis=0)
        return (predictors - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(predictors, axis=0)
        iqr = np.subtract(*np.percentile(predictors, [75, 25], axis=0))
        return (predictors - median) / (iqr + 1e-8)
    else:
        raise ValueError("Invalid normalization method")

def _get_metric(metric: Union[str, Callable], custom_metric: Optional[Callable]) -> Callable:
    """Get metric function."""
    if callable(metric):
        return metric
    elif custom_metric is not None:
        return custom_metric
    elif metric == "mse":
        return _mean_squared_error
    elif metric == "mae":
        return _mean_absolute_error
    elif metric == "r2":
        return _r_squared
    else:
        raise ValueError("Invalid metric specified")

def _get_distance(distance: str, custom_distance: Optional[Callable]) -> Callable:
    """Get distance function."""
    if callable(distance):
        return distance
    elif custom_distance is not None:
        return custom_distance
    elif distance == "euclidean":
        return _euclidean_distance
    elif distance == "manhattan":
        return _manhattan_distance
    elif distance == "cosine":
        return _cosine_distance
    else:
        raise ValueError("Invalid distance specified")

def _fit_closed_form(predictors: Optional[np.ndarray], times: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Fit model using closed-form solution."""
    if predictors is None:
        return np.array([np.mean(events)])
    # Implement closed-form solution for survival regression
    # This is a placeholder - actual implementation would depend on specific model
    return np.zeros(predictors.shape[1])

def _fit_gradient_descent(
    predictors: Optional[np.ndarray],
    times: np.ndarray,
    events: np.ndarray,
    metric_func: Callable,
    distance_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Fit model using gradient descent."""
    if predictors is None:
        # Handle intercept-only case
        return np.array([np.mean(events)])
    # Implement gradient descent algorithm
    # This is a placeholder - actual implementation would depend on specific model
    return np.zeros(predictors.shape[1])

def _fit_newton(
    predictors: Optional[np.ndarray],
    times: np.ndarray,
    events: np.ndarray,
    metric_func: Callable,
    distance_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Fit model using Newton's method."""
    if predictors is None:
        # Handle intercept-only case
        return np.array([np.mean(events)])
    # Implement Newton's method algorithm
    # This is a placeholder - actual implementation would depend on specific model
    return np.zeros(predictors.shape[1])

def _fit_coordinate_descent(
    predictors: Optional[np.ndarray],
    times: np.ndarray,
    events: np.ndarray,
    metric_func: Callable,
    distance_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Fit model using coordinate descent."""
    if predictors is None:
        # Handle intercept-only case
        return np.array([np.mean(events)])
    # Implement coordinate descent algorithm
    # This is a placeholder - actual implementation would depend on specific model
    return np.zeros(predictors.shape[1])

def _calculate_metrics(
    params: np.ndarray,
    predictors: Optional[np.ndarray],
    times: np.ndarray,
    events: np.ndarray,
    metric_func: Callable
) -> Dict:
    """Calculate evaluation metrics."""
    if predictors is None:
        predictions = np.full_like(times, params[0])
    else:
        predictions = _predict(predictors, params)

    return {
        "metric": metric_func(events, predictions),
        "mean_time": np.mean(times),
        "event_rate": np.mean(events)
    }

def _predict(predictors: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Make predictions using fitted parameters."""
    return predictors @ params

def _check_warnings(params: np.ndarray, metrics: Dict) -> list:
    """Check for potential warnings."""
    warnings = []
    if np.any(np.isnan(params)):
        warnings.append("Parameters contain NaN values")
    if metrics["metric"] > 1e6:
        warnings.append("High metric value - potential convergence issue")
    return warnings

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

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine distance."""
    return 1 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

################################################################################
# kaplan_meier
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def kaplan_meier_fit(
    times: np.ndarray,
    events: np.ndarray,
    *,
    metric_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    normalize: str = "none",
    weights: Optional[np.ndarray] = None
) -> Dict[str, Union[Dict, float]]:
    """
    Estimate survival probabilities using the Kaplan-Meier estimator.

    Parameters
    ----------
    times : np.ndarray
        Array of survival times.
    events : np.ndarray
        Binary array indicating whether the event occurred (1) or was censored (0).
    metric_func : Optional[Callable]
        Custom metric function to evaluate the fit. If None, no custom metric is computed.
    normalize : str
        Normalization method for the survival probabilities. Options: "none", "standard".
    weights : Optional[np.ndarray]
        Weights for the observations. If None, uniform weights are used.

    Returns
    -------
    Dict[str, Union[Dict, float]]
        Dictionary containing:
        - "result": Estimated survival probabilities.
        - "metrics": Computed metrics (if applicable).
        - "params_used": Parameters used in the computation.
        - "warnings": Any warnings generated during computation.

    Examples
    --------
    >>> times = np.array([5, 6, 6, 7, 8])
    >>> events = np.array([1, 0, 1, 1, 0])
    >>> result = kaplan_meier_fit(times, events)
    """
    # Validate inputs
    _validate_inputs(times, events, weights)

    # Compute Kaplan-Meier estimator
    survival_probabilities = _compute_kaplan_meier(times, events, weights)

    # Normalize if required
    if normalize == "standard":
        survival_probabilities = _normalize_survival_probs(survival_probabilities)

    # Compute metrics if a custom metric function is provided
    metrics = {}
    if metric_func is not None:
        # Example: Compute metric on some reference data
        metrics["custom_metric"] = metric_func(survival_probabilities, times)

    # Prepare output
    output = {
        "result": {"survival_probabilities": survival_probabilities},
        "metrics": metrics,
        "params_used": {
            "normalize": normalize,
            "weights_provided": weights is not None
        },
        "warnings": []
    }

    return output

def _validate_inputs(
    times: np.ndarray,
    events: np.ndarray,
    weights: Optional[np.ndarray]
) -> None:
    """Validate input arrays."""
    if times.ndim != 1 or events.ndim != 1:
        raise ValueError("Input arrays must be 1-dimensional.")
    if len(times) != len(events):
        raise ValueError("Times and events arrays must have the same length.")
    if weights is not None:
        if len(weights) != len(times):
            raise ValueError("Weights array must have the same length as times and events.")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")

def _compute_kaplan_meier(
    times: np.ndarray,
    events: np.ndarray,
    weights: Optional[np.ndarray]
) -> np.ndarray:
    """Compute Kaplan-Meier survival probabilities."""
    unique_times = np.unique(times)
    survival_probabilities = np.ones_like(unique_times, dtype=float)

    if weights is None:
        weights = np.ones_like(times)

    for i in range(1, len(unique_times)):
        mask = times <= unique_times[i]
        n_at_risk = np.sum(weights[mask])
        n_events = np.sum(events[mask] * weights[mask])

        if n_at_risk > 0:
            survival_probabilities[i] = survival_probabilities[i-1] * (1 - n_events / n_at_risk)

    return survival_probabilities

def _normalize_survival_probs(
    survival_probabilities: np.ndarray
) -> np.ndarray:
    """Normalize survival probabilities."""
    return (survival_probabilities - np.min(survival_probabilities)) / (
        np.max(survival_probabilities) - np.min(survival_probabilities)
    )

################################################################################
# cox_model
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def cox_model_fit(
    time: np.ndarray,
    event: np.ndarray,
    features: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    solver: str = 'newton',
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 1000,
    metric: str = 'logloss',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    verbose: bool = False
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit a Cox proportional hazards model.

    Parameters
    ----------
    time : np.ndarray
        Array of survival times.
    event : np.ndarray
        Binary array indicating if the event occurred (1) or was censored (0).
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize features, by default None.
    solver : str, optional
        Solver to use ('newton', 'gradient_descent'), by default 'newton'.
    regularization : Optional[str], optional
        Type of regularization ('l1', 'l2'), by default None.
    alpha : float, optional
        Regularization strength, by default 1.0.
    tol : float, optional
        Tolerance for convergence, by default 1e-6.
    max_iter : int, optional
        Maximum number of iterations, by default 1000.
    metric : str, optional
        Metric to compute ('logloss'), by default 'logloss'.
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
    >>> time = np.array([1.0, 2.0, 3.0])
    >>> event = np.array([1, 0, 1])
    >>> features = np.random.rand(3, 2)
    >>> result = cox_model_fit(time, event, features)
    """
    # Validate inputs
    _validate_inputs(time, event, features)

    # Normalize features if specified
    if normalizer is not None:
        features = normalizer(features)

    # Initialize parameters
    n_features = features.shape[1]
    coefs = np.zeros(n_features)

    # Choose solver
    if solver == 'newton':
        coefs = _newton_solver(time, event, features, coefs, tol, max_iter, verbose)
    elif solver == 'gradient_descent':
        coefs = _gradient_descent_solver(time, event, features, coefs, tol, max_iter, verbose)
    else:
        raise ValueError("Solver must be 'newton' or 'gradient_descent'")

    # Apply regularization if specified
    if regularization == 'l1':
        coefs = _apply_l1_regularization(coefs, alpha)
    elif regularization == 'l2':
        coefs = _apply_l2_regularization(coefs, alpha)

    # Compute metrics
    metrics = _compute_metrics(time, event, features, coefs, metric, custom_metric)

    # Prepare output
    result = {
        'result': coefs,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'solver': solver,
            'regularization': regularization,
            'alpha': alpha,
            'tol': tol,
            'max_iter': max_iter,
            'metric': metric
        },
        'warnings': []
    }

    return result

def _validate_inputs(time: np.ndarray, event: np.ndarray, features: np.ndarray) -> None:
    """Validate input arrays."""
    if time.shape[0] != event.shape[0]:
        raise ValueError("Time and event arrays must have the same length")
    if time.shape[0] != features.shape[0]:
        raise ValueError("Number of samples in time/event and features must match")
    if np.any(np.isnan(time)) or np.any(np.isinf(time)):
        raise ValueError("Time array contains NaN or inf values")
    if np.any(np.isnan(event)) or np.any(np.isinf(event)):
        raise ValueError("Event array contains NaN or inf values")
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        raise ValueError("Features array contains NaN or inf values")

def _newton_solver(
    time: np.ndarray,
    event: np.ndarray,
    features: np.ndarray,
    coefs: np.ndarray,
    tol: float,
    max_iter: int,
    verbose: bool
) -> np.ndarray:
    """Newton-Raphson solver for Cox model."""
    for _ in range(max_iter):
        # Compute gradient and Hessian
        grad, hess = _compute_grad_hess(time, event, features, coefs)

        # Update coefficients
        delta = np.linalg.solve(hess, grad)
        coefs -= delta

        if verbose:
            print(f"Iteration: {_}, Coefficients: {coefs}")

        if np.linalg.norm(delta) < tol:
            break
    return coefs

def _gradient_descent_solver(
    time: np.ndarray,
    event: np.ndarray,
    features: np.ndarray,
    coefs: np.ndarray,
    tol: float,
    max_iter: int,
    verbose: bool
) -> np.ndarray:
    """Gradient descent solver for Cox model."""
    learning_rate = 0.01
    for _ in range(max_iter):
        # Compute gradient
        grad = _compute_grad(time, event, features, coefs)

        # Update coefficients
        delta = learning_rate * grad
        coefs -= delta

        if verbose:
            print(f"Iteration: {_}, Coefficients: {coefs}")

        if np.linalg.norm(delta) < tol:
            break
    return coefs

def _compute_grad(
    time: np.ndarray,
    event: np.ndarray,
    features: np.ndarray,
    coefs: np.ndarray
) -> np.ndarray:
    """Compute gradient of the partial likelihood."""
    risk = np.exp(np.dot(features, coefs))
    sum_risk = np.sum(risk, axis=1)
    grad = -np.dot(features.T, event * (risk / sum_risk[:, np.newaxis]))
    return grad

def _compute_grad_hess(
    time: np.ndarray,
    event: np.ndarray,
    features: np.ndarray,
    coefs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute gradient and Hessian of the partial likelihood."""
    risk = np.exp(np.dot(features, coefs))
    sum_risk = np.sum(risk, axis=1)
    grad = -np.dot(features.T, event * (risk / sum_risk[:, np.newaxis]))

    # Compute Hessian
    hess = np.zeros((features.shape[1], features.shape[1]))
    for i in range(features.shape[0]):
        if event[i] == 1:
            risk_i = risk[i, :]
            sum_risk_i = sum_risk[i]
            hess += np.outer(features[i, :], features[i, :]) * (risk_i / sum_risk_i) * (
                1 - risk_i / sum_risk_i
            )
    return grad, hess

def _apply_l1_regularization(coefs: np.ndarray, alpha: float) -> np.ndarray:
    """Apply L1 regularization."""
    return np.sign(coefs) * np.maximum(np.abs(coefs) - alpha, 0)

def _apply_l2_regularization(coefs: np.ndarray, alpha: float) -> np.ndarray:
    """Apply L2 regularization."""
    return coefs / (1 + alpha * np.linalg.norm(coefs))

def _compute_metrics(
    time: np.ndarray,
    event: np.ndarray,
    features: np.ndarray,
    coefs: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute metrics for the Cox model."""
    risk = np.exp(np.dot(features, coefs))
    sum_risk = np.sum(risk, axis=1)
    partial_likelihood = -np.sum(np.log(sum_risk) * event)

    metrics = {'logloss': partial_likelihood}

    if custom_metric is not None:
        metrics['custom'] = custom_metric(time, event)

    return metrics

################################################################################
# log_rank_test
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(time: np.ndarray, status: np.ndarray) -> None:
    """
    Validate input arrays for log-rank test.

    Parameters
    ----------
    time : np.ndarray
        Array of survival times.
    status : np.ndarray
        Array of event indicators (1 for event, 0 for censored).

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if len(time) != len(status):
        raise ValueError("Time and status arrays must have the same length.")
    if np.any(time <= 0):
        raise ValueError("All survival times must be positive.")
    if np.any((status != 0) & (status != 1)):
        raise ValueError("Status must be either 0 or 1.")

def _compute_observed_expected(time: np.ndarray, status: np.ndarray,
                              group_indices: np.ndarray) -> Dict[str, float]:
    """
    Compute observed and expected events for log-rank test.

    Parameters
    ----------
    time : np.ndarray
        Array of survival times.
    status : np.ndarray
        Array of event indicators (1 for event, 0 for censored).
    group_indices : np.ndarray
        Array indicating group membership.

    Returns
    -------
    Dict[str, float]
        Dictionary containing observed and expected events.
    """
    unique_times = np.unique(time)
    result = {
        'observed': 0.0,
        'expected': 0.0
    }

    for t in unique_times:
        mask = time == t
        n_at_risk_group1 = np.sum((group_indices == 0) & (time >= t))
        n_at_risk_group2 = np.sum((group_indices == 1) & (time >= t))
        n_events_group1 = np.sum((group_indices == 0) & mask & (status == 1))
        n_events_group2 = np.sum((group_indices == 1) & mask & (status == 1))

        total_at_risk = n_at_risk_group1 + n_at_risk_group2
        if total_at_risk == 0:
            continue

        prob_event_group1 = n_at_risk_group1 / total_at_risk
        expected_events_group1 = prob_event_group1 * (n_events_group1 + n_events_group2)

        result['observed'] += n_events_group1 - n_events_group2
        result['expected'] += expected_events_group1 - (n_events_group2 * prob_event_group1)

    return result

def log_rank_test_fit(time: np.ndarray, status: np.ndarray,
                     group_indices: np.ndarray) -> Dict[str, Union[float, Dict]]:
    """
    Perform log-rank test for survival analysis.

    Parameters
    ----------
    time : np.ndarray
        Array of survival times.
    status : np.ndarray
        Array of event indicators (1 for event, 0 for censored).
    group_indices : np.ndarray
        Array indicating group membership (0 or 1).

    Returns
    -------
    Dict[str, Union[float, Dict]]
        Dictionary containing test results, metrics, and parameters used.
    """
    _validate_inputs(time, status)

    if len(np.unique(group_indices)) != 2:
        raise ValueError("Exactly two groups are required for log-rank test.")

    result = _compute_observed_expected(time, status, group_indices)

    # Calculate test statistic
    chi_square = (result['observed'] ** 2) / result['expected']
    p_value = 1 - _chi_square_cdf(chi_square, df=1)

    metrics = {
        'observed': result['observed'],
        'expected': result['expected'],
        'chi_square': chi_square,
        'p_value': p_value
    }

    return {
        'result': chi_square,
        'metrics': metrics,
        'params_used': {},
        'warnings': []
    }

def _chi_square_cdf(x: float, df: int) -> float:
    """
    Compute the CDF of the chi-square distribution.

    Parameters
    ----------
    x : float
        Value at which to evaluate the CDF.
    df : int
        Degrees of freedom.

    Returns
    -------
    float
        CDF value.
    """
    # This is a placeholder implementation. In practice, use scipy.stats.chi2.cdf
    if x < 0:
        return 0.0
    # Simplified approximation for demonstration purposes
    return 1 - np.exp(-x / 2)

################################################################################
# weibull_regression
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def weibull_regression_fit(
    times: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = "mse",
    solver: str = "gradient_descent",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Fit a Weibull regression model to survival data.

    Parameters
    ----------
    times : np.ndarray
        Array of observed times.
    events : np.ndarray
        Binary array indicating whether the event occurred (1) or was censored (0).
    covariates : np.ndarray
        Matrix of covariates (n_samples, n_features).
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize covariates. Default is None.
    metric : str, optional
        Metric to optimize: 'mse', 'mae', 'r2', or custom callable. Default is "mse".
    solver : str, optional
        Solver method: 'gradient_descent', 'newton', or 'coordinate_descent'. Default is "gradient_descent".
    regularization : Optional[str], optional
        Regularization type: 'l1', 'l2', or None. Default is None.
    tol : float, optional
        Tolerance for convergence. Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function. Default is None.
    **kwargs : dict
        Additional solver-specific parameters.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], str]]
        Dictionary containing:
        - "result": Estimated parameters.
        - "metrics": Computed metrics.
        - "params_used": Parameters used in the fitting process.
        - "warnings": Any warnings generated during fitting.

    Example
    -------
    >>> times = np.array([1.2, 2.3, 3.4])
    >>> events = np.array([1, 0, 1])
    >>> covariates = np.random.rand(3, 2)
    >>> result = weibull_regression_fit(times, events, covariates)
    """
    # Validate inputs
    _validate_inputs(times, events, covariates)

    # Normalize covariates if specified
    if normalizer is not None:
        covariates = normalizer(covariates)

    # Initialize parameters
    params = _initialize_parameters(covariates.shape[1])

    # Choose solver and fit model
    if solver == "gradient_descent":
        params = _gradient_descent(
            times, events, covariates, params,
            metric=metric, regularization=regularization,
            tol=tol, max_iter=max_iter, custom_metric=custom_metric,
            **kwargs
        )
    elif solver == "newton":
        params = _newton_method(
            times, events, covariates, params,
            metric=metric, regularization=regularization,
            tol=tol, max_iter=max_iter, custom_metric=custom_metric,
            **kwargs
        )
    elif solver == "coordinate_descent":
        params = _coordinate_descent(
            times, events, covariates, params,
            metric=metric, regularization=regularization,
            tol=tol, max_iter=max_iter, custom_metric=custom_metric,
            **kwargs
        )
    else:
        raise ValueError("Unsupported solver method.")

    # Compute metrics
    metrics = _compute_metrics(times, events, covariates, params, metric, custom_metric)

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

def _validate_inputs(times: np.ndarray, events: np.ndarray, covariates: np.ndarray) -> None:
    """Validate input arrays."""
    if times.shape[0] != events.shape[0]:
        raise ValueError("times and events must have the same length.")
    if times.shape[0] != covariates.shape[0]:
        raise ValueError("times and covariates must have the same number of samples.")
    if np.any(times <= 0):
        raise ValueError("times must be positive.")
    if np.any((events != 0) & (events != 1)):
        raise ValueError("events must be binary (0 or 1).")
    if np.any(np.isnan(times)) or np.any(np.isinf(times)):
        raise ValueError("times contains NaN or inf values.")
    if np.any(np.isnan(covariates)) or np.any(np.isinf(covariates)):
        raise ValueError("covariates contains NaN or inf values.")

def _initialize_parameters(n_features: int) -> np.ndarray:
    """Initialize parameters for Weibull regression."""
    return np.zeros(n_features + 1)  # Coefficients + shape parameter

def _gradient_descent(
    times: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    params: np.ndarray,
    metric: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]],
    **kwargs
) -> np.ndarray:
    """Gradient descent solver for Weibull regression."""
    learning_rate = kwargs.get("learning_rate", 0.01)
    for _ in range(max_iter):
        # Compute gradients
        grad = _compute_gradient(times, events, covariates, params)
        # Apply regularization
        if regularization == "l1":
            grad[1:] += np.sign(params[1:]) * kwargs.get("alpha", 0.1)
        elif regularization == "l2":
            grad[1:] += kwargs.get("alpha", 0.1) * params[1:]
        # Update parameters
        params -= learning_rate * grad
    return params

def _newton_method(
    times: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    params: np.ndarray,
    metric: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]],
    **kwargs
) -> np.ndarray:
    """Newton method solver for Weibull regression."""
    for _ in range(max_iter):
        # Compute gradient and Hessian
        grad = _compute_gradient(times, events, covariates, params)
        hessian = _compute_hessian(times, events, covariates, params)
        # Update parameters
        params -= np.linalg.solve(hessian, grad)
    return params

def _coordinate_descent(
    times: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    params: np.ndarray,
    metric: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]],
    **kwargs
) -> np.ndarray:
    """Coordinate descent solver for Weibull regression."""
    alpha = kwargs.get("alpha", 0.1)
    for _ in range(max_iter):
        for i in range(params.shape[0]):
            # Save current parameter
            old_param = params[i]
            # Compute gradient for this parameter
            grad = _compute_gradient(times, events, covariates, params)[i]
            # Update parameter
            if regularization == "l1":
                if grad < -alpha / 2:
                    params[i] = (old_param - grad + alpha / 2)
                elif grad > alpha / 2:
                    params[i] = (old_param - grad - alpha / 2)
                else:
                    params[i] = 0
            elif regularization == "l2":
                params[i] -= grad / (2 * alpha + 1)
            else:
                params[i] -= grad
    return params

def _compute_gradient(
    times: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    params: np.ndarray
) -> np.ndarray:
    """Compute gradient of the loss function."""
    # Placeholder for actual gradient computation
    return np.zeros_like(params)

def _compute_hessian(
    times: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    params: np.ndarray
) -> np.ndarray:
    """Compute Hessian matrix of the loss function."""
    # Placeholder for actual Hessian computation
    return np.eye(params.shape[0])

def _compute_metrics(
    times: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    params: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute metrics for the fitted model."""
    if custom_metric is not None:
        return {"custom": custom_metric(times, events)}
    elif metric == "mse":
        return {"mse": _mean_squared_error(times, events, covariates, params)}
    elif metric == "mae":
        return {"mae": _mean_absolute_error(times, events, covariates, params)}
    elif metric == "r2":
        return {"r2": _r_squared(times, events, covariates, params)}
    else:
        raise ValueError("Unsupported metric.")

def _mean_squared_error(
    times: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    params: np.ndarray
) -> float:
    """Compute mean squared error."""
    # Placeholder for actual MSE computation
    return 0.0

def _mean_absolute_error(
    times: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    params: np.ndarray
) -> float:
    """Compute mean absolute error."""
    # Placeholder for actual MAE computation
    return 0.0

def _r_squared(
    times: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    params: np.ndarray
) -> float:
    """Compute R-squared."""
    # Placeholder for actual R-squared computation
    return 0.0

################################################################################
# parametric_models
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def parametric_models_fit(
    X: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    model_type: str = 'weibull',
    solver: str = 'lbfgs',
    normalization: Optional[str] = None,
    metric: Union[str, Callable] = 'concordance',
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Fit parametric survival models to data.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Response variable
    time : np.ndarray
        Survival times
    event : np.ndarray
        Event indicators (1 if event occurred, 0 otherwise)
    model_type : str
        Type of parametric model ('weibull', 'exponential', 'lognormal')
    solver : str
        Optimization solver ('lbfgs', 'newton', 'coordinate_descent')
    normalization : Optional[str]
        Normalization method ('standard', 'minmax', None)
    metric : Union[str, Callable]
        Evaluation metric ('concordance', 'brier_score')
    regularization : Optional[str]
        Regularization type ('l1', 'l2', 'elasticnet')
    alpha : float
        Regularization strength
    l1_ratio : float
        Elastic net mixing parameter (0 <= l1_ratio <= 1)
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for optimization convergence
    random_state : Optional[int]
        Random seed for reproducibility
    custom_metric : Optional[Callable]
        Custom metric function

    Returns:
    --------
    Dict containing:
        - 'result': Fitted model parameters
        - 'metrics': Computed metrics
        - 'params_used': Parameters used in fitting
        - 'warnings': Any warnings generated

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> time = np.random.exponential(size=100)
    >>> event = np.random.randint(0, 2, size=100)
    >>> result = parametric_models_fit(X, y, time, event)
    """
    # Input validation
    _validate_inputs(X, y, time, event)

    # Normalization
    X_normalized = _apply_normalization(X, normalization)

    # Model selection and fitting
    if model_type == 'weibull':
        params, metrics = _fit_weibull(
            X_normalized, y, time, event,
            solver=solver,
            regularization=regularization,
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tol=tol
        )
    elif model_type == 'exponential':
        params, metrics = _fit_exponential(
            X_normalized, y, time, event,
            solver=solver,
            regularization=regularization,
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tol=tol
        )
    elif model_type == 'lognormal':
        params, metrics = _fit_lognormal(
            X_normalized, y, time, event,
            solver=solver,
            regularization=regularization,
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tol=tol
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Compute custom metrics if provided
    if custom_metric is not None:
        custom_value = custom_metric(params, X_normalized, y, time, event)
        metrics['custom'] = custom_value

    # Compute requested metric
    if isinstance(metric, str):
        if metric == 'concordance':
            metrics['concordance'] = _compute_concordance(params, X_normalized, time, event)
        elif metric == 'brier_score':
            metrics['brier_score'] = _compute_brier_score(params, X_normalized, time, event)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        metrics['custom_metric'] = metric(params, X_normalized, y, time, event)

    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'model_type': model_type,
            'solver': solver,
            'normalization': normalization,
            'metric': metric,
            'regularization': regularization,
            'alpha': alpha,
            'l1_ratio': l1_ratio
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray, time: np.ndarray, event: np.ndarray) -> None:
    """Validate input arrays."""
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if X.shape[0] != time.shape[0]:
        raise ValueError("X and time must have the same number of samples")
    if X.shape[0] != event.shape[0]:
        raise ValueError("X and event must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")
    if np.any(np.isnan(time)) or np.any(np.isinf(time)):
        raise ValueError("time contains NaN or infinite values")
    if np.any(event < 0) or np.any(event > 1):
        raise ValueError("event must be binary (0 or 1)")

def _apply_normalization(X: np.ndarray, method: Optional[str]) -> np.ndarray:
    """Apply normalization to features."""
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
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _fit_weibull(
    X: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    solver: str = 'lbfgs',
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> tuple:
    """Fit Weibull survival model."""
    # Implementation of Weibull model fitting
    params = _optimize_weibull(
        X, y, time, event,
        solver=solver,
        regularization=regularization,
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        tol=tol
    )
    metrics = _compute_metrics_weibull(params, X, y, time, event)
    return params, metrics

def _fit_exponential(
    X: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    solver: str = 'lbfgs',
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> tuple:
    """Fit Exponential survival model."""
    # Implementation of Exponential model fitting
    params = _optimize_exponential(
        X, y, time, event,
        solver=solver,
        regularization=regularization,
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        tol=tol
    )
    metrics = _compute_metrics_exponential(params, X, y, time, event)
    return params, metrics

def _fit_lognormal(
    X: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    solver: str = 'lbfgs',
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> tuple:
    """Fit Lognormal survival model."""
    # Implementation of Lognormal model fitting
    params = _optimize_lognormal(
        X, y, time, event,
        solver=solver,
        regularization=regularization,
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        tol=tol
    )
    metrics = _compute_metrics_lognormal(params, X, y, time, event)
    return params, metrics

def _optimize_weibull(
    X: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    solver: str = 'lbfgs',
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> Dict:
    """Optimize Weibull model parameters."""
    # Implementation of optimization for Weibull model
    return {}

def _optimize_exponential(
    X: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    solver: str = 'lbfgs',
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> Dict:
    """Optimize Exponential model parameters."""
    # Implementation of optimization for Exponential model
    return {}

def _optimize_lognormal(
    X: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    solver: str = 'lbfgs',
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> Dict:
    """Optimize Lognormal model parameters."""
    # Implementation of optimization for Lognormal model
    return {}

def _compute_metrics_weibull(
    params: Dict,
    X: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    event: np.ndarray
) -> Dict:
    """Compute metrics for Weibull model."""
    return {}

def _compute_metrics_exponential(
    params: Dict,
    X: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    event: np.ndarray
) -> Dict:
    """Compute metrics for Exponential model."""
    return {}

def _compute_metrics_lognormal(
    params: Dict,
    X: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    event: np.ndarray
) -> Dict:
    """Compute metrics for Lognormal model."""
    return {}

def _compute_concordance(
    params: Dict,
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray
) -> float:
    """Compute concordance index."""
    return 0.0

def _compute_brier_score(
    params: Dict,
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray
) -> float:
    """Compute Brier score."""
    return 0.0

################################################################################
# semi_parametric_models
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    event_observed: np.ndarray
) -> None:
    """
    Validate input data for semi-parametric survival models.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Survival times of shape (n_samples,)
    event_observed : np.ndarray
        Event indicators of shape (n_samples,) where 1 indicates event occurred

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1 or len(y) != X.shape[0]:
        raise ValueError("y must be a 1D array with length equal to X rows")
    if event_observed.ndim != 1 or len(event_observed) != X.shape[0]:
        raise ValueError("event_observed must be a 1D array with length equal to X rows")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")
    if np.any(np.isnan(event_observed)) or np.any((event_observed != 0) & (event_observed != 1)):
        raise ValueError("event_observed must contain only 0 or 1 values")

def normalize_data(
    X: np.ndarray,
    method: str = 'standard'
) -> tuple[np.ndarray, dict]:
    """
    Normalize feature matrix using specified method.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix to normalize
    method : str, optional
        Normalization method (none, standard, minmax, robust)

    Returns
    -------
    tuple[np.ndarray, dict]
        Normalized X and normalization parameters
    """
    params = {}
    if method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_norm = (X - mean) / (std + 1e-8)
        params['mean'] = mean
        params['std'] = std
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
        params['min'] = min_val
        params['max'] = max_val
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - median) / (iqr + 1e-8)
        params['median'] = median
        params['iqr'] = iqr
    else:
        X_norm = X.copy()
    return X_norm, params

def compute_risk_scores(
    X: np.ndarray,
    baseline_hazard: Callable,
    coefficients: np.ndarray
) -> np.ndarray:
    """
    Compute risk scores for each sample.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    baseline_hazard : Callable
        Baseline hazard function
    coefficients : np.ndarray
        Model coefficients of shape (n_features,)

    Returns
    -------
    np.ndarray
        Risk scores for each sample of shape (n_samples,)
    """
    linear_predictor = X @ coefficients
    return baseline_hazard(linear_predictor)

def fit_cox_proportional_hazards(
    X: np.ndarray,
    y: np.ndarray,
    event_observed: np.ndarray,
    solver: str = 'newton',
    max_iter: int = 100,
    tol: float = 1e-6
) -> tuple[np.ndarray, dict]:
    """
    Fit Cox proportional hazards model.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Survival times of shape (n_samples,)
    event_observed : np.ndarray
        Event indicators of shape (n_samples,)
    solver : str, optional
        Solver method (newton, gradient_descent)
    max_iter : int, optional
        Maximum number of iterations
    tol : float, optional
        Tolerance for convergence

    Returns
    -------
    tuple[np.ndarray, dict]
        Model coefficients and fit information
    """
    n_samples, n_features = X.shape

    # Initialize coefficients
    beta = np.zeros(n_features)
    prev_beta = beta.copy()
    converged = False

    for _ in range(max_iter):
        # Compute risk scores
        exp_beta_x = np.exp(X @ beta)

        # Compute partial likelihood gradient and hessian
        grad = np.zeros(n_features)
        hess = np.zeros((n_features, n_features))

        for i in range(n_samples):
            if event_observed[i]:
                # Compute gradient contribution
                grad += X[i] - (X @ exp_beta_x) * X[i] / np.sum(exp_beta_x)

                # Compute hessian contribution
                hess -= (X[i].reshape(-1, 1) @ X[i].reshape(1, -1)) * np.sum(exp_beta_x) / (np.sum(exp_beta_x)**2)

        # Update coefficients based on solver
        if solver == 'newton':
            beta = beta - np.linalg.inv(hess) @ grad
        else:  # gradient_descent
            beta = beta - 0.01 * grad

        # Check convergence
        if np.linalg.norm(beta - prev_beta) < tol:
            converged = True
            break

        prev_beta = beta.copy()

    fit_info = {
        'converged': converged,
        'iterations': _ + 1 if not converged else _,
        'tol': tol
    }

    return beta, fit_info

def compute_metrics(
    y_true: np.ndarray,
    event_observed: np.ndarray,
    risk_scores: np.ndarray,
    metrics: list = ['concordance_index']
) -> dict:
    """
    Compute evaluation metrics for survival models.

    Parameters
    ----------
    y_true : np.ndarray
        True survival times of shape (n_samples,)
    event_observed : np.ndarray
        Event indicators of shape (n_samples,)
    risk_scores : np.ndarray
        Predicted risk scores of shape (n_samples,)
    metrics : list, optional
        List of metric names to compute

    Returns
    -------
    dict
        Dictionary of computed metrics
    """
    results = {}

    if 'concordance_index' in metrics:
        # Compute concordance index
        n_pairs = 0
        concordant = 0

        for i in range(len(y_true)):
            if not event_observed[i]:
                continue

            for j in range(i+1, len(y_true)):
                if not event_observed[j]:
                    continue

                n_pairs += 1
                if (y_true[i] < y_true[j] and risk_scores[i] < risk_scores[j]) or \
                   (y_true[i] > y_true[j] and risk_scores[i] > risk_scores[j]):
                    concordant += 1

        results['concordance_index'] = concordant / n_pairs if n_pairs > 0 else np.nan

    return results

def semi_parametric_models_fit(
    X: np.ndarray,
    y: np.ndarray,
    event_observed: np.ndarray,
    normalization: str = 'standard',
    solver: str = 'newton',
    metrics: list = ['concordance_index'],
    baseline_hazard: Optional[Callable] = None,
    max_iter: int = 100,
    tol: float = 1e-6
) -> dict:
    """
    Fit semi-parametric survival model.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Survival times of shape (n_samples,)
    event_observed : np.ndarray
        Event indicators of shape (n_samples,)
    normalization : str, optional
        Normalization method (none, standard, minmax, robust)
    solver : str, optional
        Solver method (newton, gradient_descent)
    metrics : list, optional
        List of metric names to compute
    baseline_hazard : Callable, optional
        Custom baseline hazard function
    max_iter : int, optional
        Maximum number of iterations
    tol : float, optional
        Tolerance for convergence

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': Model coefficients
        - 'metrics': Computed metrics
        - 'params_used': Parameters used in fitting
        - 'warnings': Any warnings generated

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.exponential(size=100)
    >>> event_observed = np.ones(100, dtype=int)
    >>> result = semi_parametric_models_fit(X, y, event_observed)
    """
    # Validate inputs
    validate_inputs(X, y, event_observed)

    # Normalize data
    X_norm, norm_params = normalize_data(X, normalization)

    # Fit model
    coefficients, fit_info = fit_cox_proportional_hazards(
        X_norm,
        y,
        event_observed,
        solver=solver,
        max_iter=max_iter,
        tol=tol
    )

    # Compute risk scores
    if baseline_hazard is None:
        def default_baseline(x):
            return np.exp(x)
        risk_scores = compute_risk_scores(X_norm, default_baseline, coefficients)
    else:
        risk_scores = compute_risk_scores(X_norm, baseline_hazard, coefficients)

    # Compute metrics
    metrics_results = compute_metrics(y, event_observed, risk_scores, metrics)

    # Prepare output
    result = {
        'result': coefficients,
        'metrics': metrics_results,
        'params_used': {
            'normalization': normalization,
            'solver': solver,
            'max_iter': max_iter,
            'tol': tol
        },
        'warnings': []
    }

    if not fit_info['converged']:
        result['warnings'].append(f"Model did not converge after {fit_info['iterations']} iterations")

    return result

################################################################################
# non_parametric_models
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def non_parametric_models_fit(
    X: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    distance_metric: Union[str, Callable] = 'euclidean',
    normalization: str = 'standard',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    max_iter: int = 1000,
    tol: float = 1e-4,
    learning_rate: float = 0.01,
    alpha: float = 1.0,
    metric: Union[str, Callable] = 'mse',
    custom_metric: Optional[Callable] = None,
    verbose: bool = False
) -> Dict:
    """
    Fit non-parametric survival regression models.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Response variable.
    time : np.ndarray
        Survival times.
    event : np.ndarray
        Event indicators (1 if event occurred, 0 otherwise).
    distance_metric : str or callable
        Distance metric to use. Options: 'euclidean', 'manhattan', 'cosine',
        'minkowski', or custom callable.
    normalization : str
        Normalization method. Options: 'none', 'standard', 'minmax', 'robust'.
    solver : str
        Solver to use. Options: 'gradient_descent', 'newton', 'coordinate_descent'.
    regularization : str or None
        Regularization method. Options: 'none', 'l1', 'l2', 'elasticnet'.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for stopping criteria.
    learning_rate : float
        Learning rate for gradient descent.
    alpha : float
        Regularization strength.
    metric : str or callable
        Metric to evaluate model performance. Options: 'mse', 'mae', 'r2',
        'logloss', or custom callable.
    custom_metric : callable or None
        Custom metric function if needed.
    verbose : bool
        Whether to print progress information.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y, time, event)

    # Normalize data
    X_normalized = _normalize_data(X, normalization)

    # Choose distance metric
    distance_func = _get_distance_function(distance_metric)

    # Initialize parameters
    params = _initialize_parameters(X_normalized.shape[1])

    # Choose solver and fit model
    if solver == 'gradient_descent':
        params = _gradient_descent(
            X_normalized, y, time, event, distance_func,
            params, max_iter, tol, learning_rate, alpha, regularization
        )
    elif solver == 'newton':
        params = _newton_method(
            X_normalized, y, time, event, distance_func,
            params, max_iter, tol, alpha, regularization
        )
    elif solver == 'coordinate_descent':
        params = _coordinate_descent(
            X_normalized, y, time, event, distance_func,
            params, max_iter, tol, alpha, regularization
        )
    else:
        raise ValueError("Unsupported solver")

    # Compute metrics
    metrics = _compute_metrics(
        X_normalized, y, time, event, params,
        metric if custom_metric is None else custom_metric
    )

    # Prepare output
    result = {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'distance_metric': distance_metric,
            'normalization': normalization,
            'solver': solver,
            'regularization': regularization,
            'max_iter': max_iter,
            'tol': tol,
            'learning_rate': learning_rate,
            'alpha': alpha
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray, time: np.ndarray, event: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if X.shape[0] != y.shape[0] or X.shape[0] != time.shape[0] or X.shape[0] != event.shape[0]:
        raise ValueError("All input arrays must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input data contains NaN or infinite values")
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError("Feature matrix must contain numerical values")

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
        raise ValueError("Unsupported normalization method")

def _get_distance_function(metric: Union[str, Callable]) -> Callable:
    """Get the distance function based on the metric name or return custom callable."""
    if metric == 'euclidean':
        return lambda a, b: np.linalg.norm(a - b)
    elif metric == 'manhattan':
        return lambda a, b: np.sum(np.abs(a - b))
    elif metric == 'cosine':
        return lambda a, b: 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    elif metric == 'minkowski':
        return lambda a, b: np.sum(np.abs(a - b)**3)**(1/3)
    elif callable(metric):
        return metric
    else:
        raise ValueError("Unsupported distance metric")

def _initialize_parameters(n_features: int) -> np.ndarray:
    """Initialize model parameters."""
    return np.zeros(n_features)

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    distance_func: Callable,
    params: np.ndarray,
    max_iter: int,
    tol: float,
    learning_rate: float,
    alpha: float,
    regularization: Optional[str]
) -> np.ndarray:
    """Gradient descent solver for non-parametric survival regression."""
    for _ in range(max_iter):
        # Compute gradients and update parameters
        pass  # Implementation of gradient descent logic
    return params

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    distance_func: Callable,
    params: np.ndarray,
    max_iter: int,
    tol: float,
    alpha: float,
    regularization: Optional[str]
) -> np.ndarray:
    """Newton method solver for non-parametric survival regression."""
    for _ in range(max_iter):
        # Compute Hessian and update parameters
        pass  # Implementation of Newton method logic
    return params

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    distance_func: Callable,
    params: np.ndarray,
    max_iter: int,
    tol: float,
    alpha: float,
    regularization: Optional[str]
) -> np.ndarray:
    """Coordinate descent solver for non-parametric survival regression."""
    for _ in range(max_iter):
        # Update one coordinate at a time
        pass  # Implementation of coordinate descent logic
    return params

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    params: np.ndarray,
    metric_func: Callable
) -> Dict:
    """Compute evaluation metrics."""
    if metric_func == 'mse':
        return {'mse': np.mean((y - X @ params)**2)}
    elif metric_func == 'mae':
        return {'mae': np.mean(np.abs(y - X @ params))}
    elif metric_func == 'r2':
        ss_res = np.sum((y - X @ params)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return {'r2': 1 - ss_res / (ss_tot + 1e-8)}
    elif metric_func == 'logloss':
        # Implement log loss for survival analysis
        pass  # Implementation of log loss logic
    elif callable(metric_func):
        return {'custom_metric': metric_func(y, X @ params)}
    else:
        raise ValueError("Unsupported metric")

################################################################################
# time_dependent_covariates
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def time_dependent_covariates_fit(
    times: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = 'mse',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **solver_kwargs
) -> Dict:
    """
    Fit a time-dependent covariates survival regression model.

    Parameters:
    -----------
    times : np.ndarray
        Array of event/censoring times.
    events : np.ndarray
        Binary array indicating if the event occurred (1) or was censored (0).
    covariates : np.ndarray
        2D array of shape (n_samples, n_features) containing time-dependent covariates.
    normalizer : Optional[Callable]
        Function to normalize the covariates. Default is None (no normalization).
    metric : Union[str, Callable]
        Metric to evaluate the model. Can be 'mse', 'mae', 'r2', or a custom callable.
    solver : str
        Solver to use. Options: 'gradient_descent', 'newton', 'coordinate_descent'.
    regularization : Optional[str]
        Regularization type. Options: None, 'l1', 'l2', 'elasticnet'.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable]
        Custom metric function if not using built-in metrics.
    **solver_kwargs
        Additional keyword arguments for the solver.

    Returns:
    --------
    Dict containing:
        - 'result': Fitted model parameters.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings encountered.

    Example:
    --------
    >>> times = np.array([1.0, 2.0, 3.0])
    >>> events = np.array([1, 0, 1])
    >>> covariates = np.random.rand(3, 2)
    >>> result = time_dependent_covariates_fit(times, events, covariates)
    """
    # Validate inputs
    _validate_inputs(times, events, covariates)

    # Normalize covariates if specified
    if normalizer is not None:
        covariates = normalizer(covariates)

    # Initialize parameters
    n_features = covariates.shape[1]
    params = np.zeros(n_features)

    # Choose solver
    if solver == 'gradient_descent':
        params = _gradient_descent(
            times, events, covariates, params,
            metric=metric, regularization=regularization,
            tol=tol, max_iter=max_iter, **solver_kwargs
        )
    elif solver == 'newton':
        params = _newton_method(
            times, events, covariates, params,
            metric=metric, regularization=regularization,
            tol=tol, max_iter=max_iter, **solver_kwargs
        )
    elif solver == 'coordinate_descent':
        params = _coordinate_descent(
            times, events, covariates, params,
            metric=metric, regularization=regularization,
            tol=tol, max_iter=max_iter, **solver_kwargs
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(times, events, covariates, params, metric, custom_metric)

    return {
        'result': params,
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

def _validate_inputs(times: np.ndarray, events: np.ndarray, covariates: np.ndarray) -> None:
    """Validate input arrays."""
    if len(times) != len(events):
        raise ValueError("times and events must have the same length")
    if len(times) != covariates.shape[0]:
        raise ValueError("number of samples in times/events must match covariates")
    if np.any(np.isnan(times)) or np.any(np.isinf(times)):
        raise ValueError("times contains NaN or inf values")
    if np.any(np.isnan(events)) or np.any(np.isinf(events)):
        raise ValueError("events contains NaN or inf values")
    if np.any(np.isnan(covariates)) or np.any(np.isinf(covariates)):
        raise ValueError("covariates contains NaN or inf values")

def _gradient_descent(
    times: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    **kwargs
) -> np.ndarray:
    """Gradient descent solver for time-dependent covariates."""
    learning_rate = kwargs.get('learning_rate', 0.01)
    for _ in range(max_iter):
        gradient = _compute_gradient(times, events, covariates, params, metric)
        if regularization == 'l1':
            gradient += np.sign(params) * kwargs.get('alpha', 0.1)
        elif regularization == 'l2':
            gradient += 2 * kwargs.get('alpha', 0.1) * params
        params -= learning_rate * gradient
        if np.linalg.norm(gradient) < tol:
            break
    return params

def _newton_method(
    times: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    **kwargs
) -> np.ndarray:
    """Newton's method solver for time-dependent covariates."""
    for _ in range(max_iter):
        gradient = _compute_gradient(times, events, covariates, params, metric)
        hessian = _compute_hessian(times, events, covariates, params)
        if regularization == 'l2':
            hessian += 2 * kwargs.get('alpha', 0.1) * np.eye(len(params))
        params -= np.linalg.solve(hessian, gradient)
        if np.linalg.norm(gradient) < tol:
            break
    return params

def _coordinate_descent(
    times: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    **kwargs
) -> np.ndarray:
    """Coordinate descent solver for time-dependent covariates."""
    alpha = kwargs.get('alpha', 0.1)
    for _ in range(max_iter):
        for i in range(len(params)):
            # Compute gradient for feature i
            grad_i = _compute_gradient_feature(
                times, events, covariates, params, i, metric
            )
            if regularization == 'l1':
                grad_i += np.sign(params[i]) * alpha
            elif regularization == 'l2':
                grad_i += 2 * alpha * params[i]
            # Update parameter
            params[i] -= grad_i / _compute_hessian_feature(
                times, events, covariates, params, i
            )
        if np.linalg.norm(_compute_gradient(times, events, covariates, params, metric)) < tol:
            break
    return params

def _compute_gradient(
    times: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable]
) -> np.ndarray:
    """Compute gradient of the loss function."""
    if metric == 'mse':
        residuals = _compute_residuals(times, events, covariates @ params)
        return -2 * (covariates.T @ residuals) / len(times)
    elif metric == 'mae':
        residuals = _compute_residuals(times, events, covariates @ params)
        return -np.sign(residuals).T @ covariates / len(times)
    elif callable(metric):
        return metric(covariates, times, events, params)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _compute_hessian(
    times: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    params: np.ndarray
) -> np.ndarray:
    """Compute Hessian matrix of the loss function."""
    return 2 * (covariates.T @ covariates) / len(times)

def _compute_hessian_feature(
    times: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    params: np.ndarray,
    feature_idx: int
) -> float:
    """Compute Hessian for a single feature."""
    return 2 * (covariates[:, feature_idx] ** 2).sum() / len(times)

def _compute_gradient_feature(
    times: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    params: np.ndarray,
    feature_idx: int,
    metric: Union[str, Callable]
) -> float:
    """Compute gradient for a single feature."""
    if metric == 'mse':
        residuals = _compute_residuals(times, events, covariates @ params)
        return -2 * (covariates[:, feature_idx] @ residuals) / len(times)
    elif metric == 'mae':
        residuals = _compute_residuals(times, events, covariates @ params)
        return -np.sign(residuals) @ covariates[:, feature_idx] / len(times)
    elif callable(metric):
        return metric(covariates, times, events, params)[feature_idx]
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _compute_residuals(
    times: np.ndarray,
    events: np.ndarray,
    predictions: np.ndarray
) -> np.ndarray:
    """Compute residuals for survival regression."""
    return events * (times - predictions)

def _compute_metrics(
    times: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Compute evaluation metrics."""
    predictions = covariates @ params
    metrics_dict = {}

    if metric == 'mse':
        residuals = _compute_residuals(times, events, predictions)
        metrics_dict['mse'] = np.mean(residuals ** 2)
    elif metric == 'mae':
        residuals = _compute_residuals(times, events, predictions)
        metrics_dict['mae'] = np.mean(np.abs(residuals))
    elif metric == 'r2':
        residuals = _compute_residuals(times, events, predictions)
        ss_total = np.sum((times - np.mean(times)) ** 2)
        metrics_dict['r2'] = 1 - (np.sum(residuals ** 2) / ss_total)
    elif callable(metric):
        metrics_dict['custom'] = metric(times, events, predictions)

    if custom_metric is not None:
        metrics_dict['custom'] = custom_metric(times, events, predictions)

    return metrics_dict

################################################################################
# risk_set
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def risk_set_fit(
    time: np.ndarray,
    event: np.ndarray,
    features: np.ndarray,
    *,
    normalization: str = "standard",
    metric: Union[str, Callable] = "mse",
    distance: str = "euclidean",
    solver: str = "gradient_descent",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fit a risk set model for survival analysis.

    Parameters:
    -----------
    time : np.ndarray
        Array of observed times.
    event : np.ndarray
        Binary array indicating if the event occurred (1) or was censored (0).
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    normalization : str, optional
        Normalization method: "none", "standard", "minmax", or "robust".
    metric : str or callable, optional
        Metric to optimize: "mse", "mae", "r2", or custom callable.
    distance : str, optional
        Distance metric: "euclidean", "manhattan", "cosine", or custom callable.
    solver : str, optional
        Solver method: "closed_form", "gradient_descent", "newton", or "coordinate_descent".
    regularization : str, optional
        Regularization type: "none", "l1", "l2", or "elasticnet".
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    custom_distance : callable, optional
        Custom distance function if not using built-in distances.

    Returns:
    --------
    Dict containing:
        - "result": Fitted model parameters.
        - "metrics": Computed metrics.
        - "params_used": Parameters used during fitting.
        - "warnings": Any warnings encountered.

    Example:
    --------
    >>> time = np.array([1, 2, 3])
    >>> event = np.array([1, 0, 1])
    >>> features = np.random.rand(3, 2)
    >>> result = risk_set_fit(time, event, features)
    """
    # Validate inputs
    _validate_inputs(time, event, features)

    # Normalize features
    normalized_features = _normalize(features, method=normalization)

    # Prepare risk set data
    risk_set_data = _prepare_risk_set(time, event)

    # Choose metric and distance functions
    metric_func = _get_metric(metric, custom_metric)
    distance_func = _get_distance(distance, custom_distance)

    # Choose solver
    if solver == "closed_form":
        params = _solve_closed_form(normalized_features, risk_set_data)
    elif solver == "gradient_descent":
        params = _solve_gradient_descent(
            normalized_features, risk_set_data,
            metric_func=metric_func,
            tol=tol,
            max_iter=max_iter
        )
    elif solver == "newton":
        params = _solve_newton(
            normalized_features, risk_set_data,
            metric_func=metric_func,
            tol=tol,
            max_iter=max_iter
        )
    elif solver == "coordinate_descent":
        params = _solve_coordinate_descent(
            normalized_features, risk_set_data,
            metric_func=metric_func,
            tol=tol,
            max_iter=max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if specified
    if regularization is not None:
        params = _apply_regularization(params, normalized_features, risk_set_data, regularization)

    # Compute metrics
    metrics = _compute_metrics(params, normalized_features, risk_set_data, metric_func)

    return {
        "result": params,
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
        "warnings": []
    }

def _validate_inputs(time: np.ndarray, event: np.ndarray, features: np.ndarray) -> None:
    """Validate input arrays."""
    if time.shape[0] != event.shape[0]:
        raise ValueError("time and event must have the same length")
    if features.shape[0] != time.shape[0]:
        raise ValueError("features must have same number of samples as time/event")
    if np.any(np.isnan(time)) or np.any(np.isinf(time)):
        raise ValueError("time contains NaN or inf values")
    if np.any(np.isnan(event)) or np.any(np.isinf(event)):
        raise ValueError("event contains NaN or inf values")
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        raise ValueError("features contains NaN or inf values")

def _normalize(features: np.ndarray, method: str = "standard") -> np.ndarray:
    """Normalize features."""
    if method == "none":
        return features
    elif method == "standard":
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        return (features - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(features, axis=0)
        max_val = np.max(features, axis=0)
        return (features - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(features, axis=0)
        iqr = np.subtract(*np.percentile(features, [75, 25], axis=0))
        return (features - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _prepare_risk_set(time: np.ndarray, event: np.ndarray) -> Dict:
    """Prepare risk set data structure."""
    # Sort by time
    sort_idx = np.argsort(time)
    sorted_time = time[sort_idx]
    sorted_event = event[sort_idx]

    # Create risk set structure
    risk_set = []
    current_risk_set = np.ones_like(time, dtype=bool)

    for t, e in zip(sorted_time, sorted_event):
        # Get current risk set
        current_indices = np.where(current_risk_set)[0]
        risk_set.append({
            "time": t,
            "event": e,
            "indices": current_indices
        })
        # Update risk set (remove observed events)
        if e:
            current_risk_set[current_indices] = False

    return {
        "sorted_time": sorted_time,
        "sorted_event": sorted_event,
        "risk_set": risk_set
    }

def _get_metric(metric: Union[str, Callable], custom_metric: Optional[Callable] = None) -> Callable:
    """Get metric function."""
    if callable(metric):
        return metric
    elif custom_metric is not None:
        return custom_metric
    elif metric == "mse":
        return _mean_squared_error
    elif metric == "mae":
        return _mean_absolute_error
    elif metric == "r2":
        return _r_squared
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _get_distance(distance: str, custom_distance: Optional[Callable] = None) -> Callable:
    """Get distance function."""
    if custom_distance is not None:
        return custom_distance
    elif distance == "euclidean":
        return _euclidean_distance
    elif distance == "manhattan":
        return _manhattan_distance
    elif distance == "cosine":
        return _cosine_distance
    else:
        raise ValueError(f"Unknown distance: {distance}")

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance."""
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def _solve_closed_form(features: np.ndarray, risk_set_data: Dict) -> np.ndarray:
    """Closed form solution."""
    # This is a placeholder - actual implementation would depend on the specific risk set model
    X = features[risk_set_data["risk_set"][0]["indices"]]
    y = risk_set_data["sorted_event"][risk_set_data["risk_set"][0]["indices"]]
    XTX = np.dot(X.T, X)
    XTy = np.dot(X.T, y)
    return np.linalg.solve(XTX + 1e-8 * np.eye(X.shape[1]), XTy)

def _solve_gradient_descent(
    features: np.ndarray,
    risk_set_data: Dict,
    *,
    metric_func: Callable,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Gradient descent solver."""
    # Initialize parameters
    params = np.zeros(features.shape[1])
    prev_params = None

    for _ in range(max_iter):
        # Compute gradient
        grad = _compute_gradient(features, risk_set_data, params)

        # Update parameters
        prev_params = params.copy()
        params -= 0.01 * grad

        # Check convergence
        if prev_params is not None and np.linalg.norm(params - prev_params) < tol:
            break

    return params

def _solve_newton(
    features: np.ndarray,
    risk_set_data: Dict,
    *,
    metric_func: Callable,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Newton's method solver."""
    # Initialize parameters
    params = np.zeros(features.shape[1])
    prev_params = None

    for _ in range(max_iter):
        # Compute gradient and hessian
        grad = _compute_gradient(features, risk_set_data, params)
        hessian = _compute_hessian(features, risk_set_data, params)

        # Update parameters
        prev_params = params.copy()
        params -= np.linalg.solve(hessian + 1e-8 * np.eye(features.shape[1]), grad)

        # Check convergence
        if prev_params is not None and np.linalg.norm(params - prev_params) < tol:
            break

    return params

def _solve_coordinate_descent(
    features: np.ndarray,
    risk_set_data: Dict,
    *,
    metric_func: Callable,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Coordinate descent solver."""
    # Initialize parameters
    params = np.zeros(features.shape[1])
    prev_params = None

    for _ in range(max_iter):
        for i in range(features.shape[1]):
            # Save current parameter
            old_param = params[i]

            # Compute gradient for this feature
            grad = _compute_gradient(features, risk_set_data, params)[i]

            # Update parameter
            if grad != 0:
                params[i] -= old_param / (grad + 1e-8)

        # Check convergence
        if prev_params is not None and np.linalg.norm(params - prev_params) < tol:
            break

    return params

def _compute_gradient(features: np.ndarray, risk_set_data: Dict, params: np.ndarray) -> np.ndarray:
    """Compute gradient of the loss function."""
    # This is a placeholder - actual implementation would depend on the specific risk set model
    return np.zeros(features.shape[1])

def _compute_hessian(features: np.ndarray, risk_set_data: Dict, params: np.ndarray) -> np.ndarray:
    """Compute hessian of the loss function."""
    # This is a placeholder - actual implementation would depend on the specific risk set model
    return np.eye(features.shape[1])

def _apply_regularization(
    params: np.ndarray,
    features: np.ndarray,
    risk_set_data: Dict,
    regularization: str
) -> np.ndarray:
    """Apply regularization to parameters."""
    if regularization == "l1":
        # Lasso - add L1 penalty
        return params * (1 + 0.1)  # Placeholder - actual implementation would depend on penalty strength
    elif regularization == "l2":
        # Ridge - add L2 penalty
        return params * (1 + 0.1)  # Placeholder - actual implementation would depend on penalty strength
    elif regularization == "elasticnet":
        # Elastic net - combination of L1 and L2
        return params * (1 + 0.1)  # Placeholder - actual implementation would depend on penalty strengths
    else:
        return params

def _compute_metrics(
    params: np.ndarray,
    features: np.ndarray,
    risk_set_data: Dict,
    metric_func: Callable
) -> Dict:
    """Compute metrics for the fitted model."""
    # This is a placeholder - actual implementation would depend on the specific risk set model
    return {
        "metric": metric_func(np.zeros(features.shape[0]), np.zeros(features.shape[0])),  # Placeholder
        "c_index": 0.5,  # Placeholder for concordance index
        "brier_score": 0.1  # Placeholder for Brier score
    }

################################################################################
# baseline_hazard
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def baseline_hazard_fit(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    normalization: str = 'none',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Compute the baseline hazard function for survival analysis.

    Parameters:
    -----------
    event_times : np.ndarray
        Array of observed event times.
    event_indicators : np.ndarray
        Binary array indicating whether an event occurred (1) or was censored (0).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to optimize ('mse', 'mae', 'r2', 'logloss') or custom callable.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    **kwargs :
        Additional solver-specific parameters.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, used parameters, and warnings.

    Example:
    --------
    >>> event_times = np.array([1.2, 3.4, 5.6])
    >>> event_indicators = np.array([1, 0, 1])
    >>> result = baseline_hazard_fit(event_times, event_indicators)
    """
    # Validate inputs
    _validate_inputs(event_times, event_indicators)

    # Choose normalization
    normalized_event_times = _apply_normalization(event_times, normalization)

    # Select metric
    if callable(metric):
        metric_func = metric
    else:
        metric_func = _get_metric(metric)

    # Select solver
    if solver == 'closed_form':
        baseline_hazard = _closed_form_solver(normalized_event_times, event_indicators)
    elif solver == 'gradient_descent':
        baseline_hazard = _gradient_descent_solver(
            normalized_event_times, event_indicators,
            metric_func, tol, max_iter, **kwargs
        )
    elif solver == 'newton':
        baseline_hazard = _newton_solver(
            normalized_event_times, event_indicators,
            metric_func, tol, max_iter, **kwargs
        )
    elif solver == 'coordinate_descent':
        baseline_hazard = _coordinate_descent_solver(
            normalized_event_times, event_indicators,
            metric_func, tol, max_iter, **kwargs
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(baseline_hazard, normalized_event_times, event_indicators, metric_func)

    # Prepare output
    result = {
        'result': baseline_hazard,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'solver': solver,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return result

def _validate_inputs(event_times: np.ndarray, event_indicators: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(event_times, np.ndarray) or not isinstance(event_indicators, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if event_times.ndim != 1 or event_indicators.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional")
    if len(event_times) != len(event_indicators):
        raise ValueError("event_times and event_indicators must have the same length")
    if np.any(np.isnan(event_times)) or np.any(np.isinf(event_times)):
        raise ValueError("event_times contains NaN or infinite values")
    if np.any(event_indicators < 0) or np.any(event_indicators > 1):
        raise ValueError("event_indicators must be binary (0 or 1)")

def _apply_normalization(event_times: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to event times."""
    if method == 'none':
        return event_times
    elif method == 'standard':
        mean = np.mean(event_times)
        std = np.std(event_times)
        return (event_times - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(event_times)
        max_val = np.max(event_times)
        return (event_times - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(event_times)
        iqr = np.percentile(event_times, 75) - np.percentile(event_times, 25)
        return (event_times - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _get_metric(metric_name: str) -> Callable:
    """Get metric function by name."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared,
        'logloss': _log_loss
    }
    if metric_name not in metrics:
        raise ValueError(f"Unknown metric: {metric_name}")
    return metrics[metric_name]

def _closed_form_solver(event_times: np.ndarray, event_indicators: np.ndarray) -> np.ndarray:
    """Closed form solution for baseline hazard."""
    # Implement closed form calculation
    unique_times = np.unique(event_times)
    n_at_risk = np.zeros_like(unique_times, dtype=float)
    n_events = np.zeros_like(unique_times, dtype=float)

    for i, t in enumerate(unique_times):
        n_at_risk[i] = np.sum(event_times >= t)
        n_events[i] = np.sum((event_times == t) & (event_indicators == 1))

    baseline_hazard = n_events / (n_at_risk + 1e-8)
    return baseline_hazard

def _gradient_descent_solver(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    metric_func: Callable,
    tol: float,
    max_iter: int,
    **kwargs
) -> np.ndarray:
    """Gradient descent solver for baseline hazard."""
    # Implement gradient descent algorithm
    unique_times = np.unique(event_times)
    baseline_hazard = np.ones_like(unique_times) * 0.5
    prev_metric = float('inf')

    for _ in range(max_iter):
        # Compute gradient and update
        gradient = _compute_gradient(baseline_hazard, event_times, event_indicators)
        baseline_hazard -= kwargs.get('learning_rate', 0.01) * gradient

        # Check convergence
        current_metric = metric_func(baseline_hazard, event_times, event_indicators)
        if abs(prev_metric - current_metric) < tol:
            break
        prev_metric = current_metric

    return baseline_hazard

def _compute_gradient(
    baseline_hazard: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray
) -> np.ndarray:
    """Compute gradient for baseline hazard."""
    # Implement gradient computation
    unique_times = np.unique(event_times)
    n_at_risk = np.zeros_like(unique_times, dtype=float)

    for i, t in enumerate(unique_times):
        n_at_risk[i] = np.sum(event_times >= t)

    # Simple gradient approximation (to be replaced with actual implementation)
    return np.random.randn(len(unique_times)) * 0.1

def _newton_solver(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    metric_func: Callable,
    tol: float,
    max_iter: int,
    **kwargs
) -> np.ndarray:
    """Newton's method solver for baseline hazard."""
    # Implement Newton's method algorithm
    unique_times = np.unique(event_times)
    baseline_hazard = np.ones_like(unique_times) * 0.5
    prev_metric = float('inf')

    for _ in range(max_iter):
        # Compute gradient and hessian
        gradient = _compute_gradient(baseline_hazard, event_times, event_indicators)
        hessian = _compute_hessian(baseline_hazard, event_times, event_indicators)

        # Update parameters
        baseline_hazard -= np.linalg.solve(hessian, gradient)

        # Check convergence
        current_metric = metric_func(baseline_hazard, event_times, event_indicators)
        if abs(prev_metric - current_metric) < tol:
            break
        prev_metric = current_metric

    return baseline_hazard

def _compute_hessian(
    baseline_hazard: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray
) -> np.ndarray:
    """Compute hessian for baseline hazard."""
    # Implement hessian computation
    unique_times = np.unique(event_times)
    n_at_risk = np.zeros_like(unique_times, dtype=float)

    for i, t in enumerate(unique_times):
        n_at_risk[i] = np.sum(event_times >= t)

    # Simple hessian approximation (to be replaced with actual implementation)
    return np.diag(np.random.rand(len(unique_times)) * 0.1)

def _coordinate_descent_solver(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    metric_func: Callable,
    tol: float,
    max_iter: int,
    **kwargs
) -> np.ndarray:
    """Coordinate descent solver for baseline hazard."""
    # Implement coordinate descent algorithm
    unique_times = np.unique(event_times)
    baseline_hazard = np.ones_like(unique_times) * 0.5
    prev_metric = float('inf')

    for _ in range(max_iter):
        for i in range(len(unique_times)):
            # Update one parameter at a time
            old_value = baseline_hazard[i]
            baseline_hazard[i] = _optimize_single_parameter(
                i, baseline_hazard, event_times, event_indicators
            )

        # Check convergence
        current_metric = metric_func(baseline_hazard, event_times, event_indicators)
        if abs(prev_metric - current_metric) < tol:
            break
        prev_metric = current_metric

    return baseline_hazard

def _optimize_single_parameter(
    param_idx: int,
    baseline_hazard: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray
) -> float:
    """Optimize a single parameter in baseline hazard."""
    # Implement optimization for one parameter
    unique_times = np.unique(event_times)
    t = unique_times[param_idx]

    # Simple optimization (to be replaced with actual implementation)
    return np.random.rand()

def _compute_metrics(
    baseline_hazard: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    metric_func: Callable
) -> Dict:
    """Compute metrics for baseline hazard."""
    metrics = {
        'primary_metric': metric_func(baseline_hazard, event_times, event_indicators),
        'mse': _mean_squared_error(baseline_hazard, event_times, event_indicators),
        'mae': _mean_absolute_error(baseline_hazard, event_times, event_indicators),
        'r2': _r_squared(baseline_hazard, event_times, event_indicators)
    }
    return metrics

def _mean_squared_error(
    baseline_hazard: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray
) -> float:
    """Compute mean squared error."""
    # Implement MSE calculation
    return np.mean((baseline_hazard - event_times) ** 2)

def _mean_absolute_error(
    baseline_hazard: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray
) -> float:
    """Compute mean absolute error."""
    # Implement MAE calculation
    return np.mean(np.abs(baseline_hazard - event_times))

def _r_squared(
    baseline_hazard: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray
) -> float:
    """Compute R-squared."""
    # Implement R-squared calculation
    return 1.0 - np.sum((baseline_hazard - event_times) ** 2) / np.sum((event_times - np.mean(event_times)) ** 2)

def _log_loss(
    baseline_hazard: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray
) -> float:
    """Compute log loss."""
    # Implement log loss calculation
    return -np.mean(event_indicators * np.log(baseline_hazard + 1e-8) +
                    (1 - event_indicators) * np.log(1 - baseline_hazard + 1e-8))

################################################################################
# proportional_hazards
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def proportional_hazards_fit(
    durations: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = "logloss",
    solver: str = "gradient_descent",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **solver_kwargs
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit a proportional hazards model to survival data.

    Parameters
    ----------
    durations : np.ndarray
        Array of observed durations (time to event or censoring).
    events : np.ndarray
        Binary array indicating whether the event occurred (1) or was censored (0).
    covariates : np.ndarray
        2D array of shape (n_samples, n_features) containing the covariates.
    normalizer : Optional[Callable]
        Function to normalize the covariates. If None, no normalization is applied.
    metric : str
        Metric to evaluate the model. Options: "logloss", "mse", "mae".
    solver : str
        Solver to use for optimization. Options: "gradient_descent", "newton".
    regularization : Optional[str]
        Regularization type. Options: None, "l1", "l2".
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable]
        Custom metric function. If provided, overrides the `metric` parameter.
    **solver_kwargs
        Additional keyword arguments for the solver.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing the results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> durations = np.array([1.0, 2.0, 3.0])
    >>> events = np.array([1, 0, 1])
    >>> covariates = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> result = proportional_hazards_fit(durations, events, covariates)
    """
    # Validate inputs
    _validate_inputs(durations, events, covariates)

    # Normalize covariates if a normalizer is provided
    if normalizer is not None:
        covariates = normalizer(covariates)

    # Initialize parameters
    n_features = covariates.shape[1]
    coefs = np.zeros(n_features)

    # Choose solver
    if solver == "gradient_descent":
        coefs = _gradient_descent(
            durations, events, covariates, coefs,
            metric=metric, regularization=regularization,
            tol=tol, max_iter=max_iter, custom_metric=custom_metric,
            **solver_kwargs
        )
    elif solver == "newton":
        coefs = _newton_method(
            durations, events, covariates, coefs,
            metric=metric, regularization=regularization,
            tol=tol, max_iter=max_iter, custom_metric=custom_metric,
            **solver_kwargs
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(durations, events, covariates, coefs, metric, custom_metric)

    # Prepare output
    result = {
        "result": {"coefficients": coefs},
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

def _validate_inputs(durations: np.ndarray, events: np.ndarray, covariates: np.ndarray) -> None:
    """Validate the input data."""
    if durations.shape[0] != events.shape[0]:
        raise ValueError("durations and events must have the same length")
    if covariates.shape[0] != durations.shape[0]:
        raise ValueError("covariates must have the same number of samples as durations and events")
    if not np.all(np.isfinite(durations)) or not np.all(np.isfinite(events)) or not np.all(np.isfinite(covariates)):
        raise ValueError("Input data must contain only finite values")
    if not np.all((events == 0) | (events == 1)):
        raise ValueError("events must be binary (0 or 1)")

def _gradient_descent(
    durations: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    coefs: np.ndarray,
    metric: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]],
    **kwargs
) -> np.ndarray:
    """Gradient descent solver for proportional hazards model."""
    learning_rate = kwargs.get("learning_rate", 0.01)
    for _ in range(max_iter):
        # Compute gradient
        gradient = _compute_gradient(durations, events, covariates, coefs)
        # Apply regularization
        if regularization == "l1":
            gradient += np.sign(coefs)
        elif regularization == "l2":
            gradient += 2 * coefs
        # Update coefficients
        coefs -= learning_rate * gradient
        # Check convergence
        if np.linalg.norm(gradient) < tol:
            break
    return coefs

def _newton_method(
    durations: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    coefs: np.ndarray,
    metric: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]],
    **kwargs
) -> np.ndarray:
    """Newton's method solver for proportional hazards model."""
    for _ in range(max_iter):
        # Compute gradient and hessian
        gradient = _compute_gradient(durations, events, covariates, coefs)
        hessian = _compute_hessian(durations, events, covariates, coefs)
        # Apply regularization
        if regularization == "l1":
            hessian += np.eye(len(coefs))
        elif regularization == "l2":
            hessian += 2 * np.eye(len(coefs))
        # Update coefficients
        coefs -= np.linalg.solve(hessian, gradient)
        # Check convergence
        if np.linalg.norm(gradient) < tol:
            break
    return coefs

def _compute_gradient(
    durations: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    coefs: np.ndarray
) -> np.ndarray:
    """Compute the gradient of the negative log-likelihood."""
    risk = np.exp(np.dot(covariates, coefs))
    sum_risk = np.sum(risk)
    gradient = -np.dot(covariates.T, events * (covariates - np.outer(risk / sum_risk, covariates) @ (events / risk)))
    return gradient

def _compute_hessian(
    durations: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    coefs: np.ndarray
) -> np.ndarray:
    """Compute the Hessian matrix of the negative log-likelihood."""
    risk = np.exp(np.dot(covariates, coefs))
    sum_risk = np.sum(risk)
    hessian = -np.dot(covariates.T, events * (covariates - np.outer(risk / sum_risk, covariates) @ (events / risk)))
    return hessian

def _compute_metrics(
    durations: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    coefs: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute the metrics for the model."""
    if custom_metric is not None:
        return {"custom_metric": custom_metric(durations, events)}

    metrics = {}
    if metric == "logloss":
        risk = np.exp(np.dot(covariates, coefs))
        sum_risk = np.sum(risk)
        loglikelihood = -np.sum(events * (np.log(sum_risk) + np.dot(covariates, coefs)) - risk / sum_risk)
        metrics["logloss"] = loglikelihood
    elif metric == "mse":
        # Placeholder for MSE calculation
        pass
    elif metric == "mae":
        # Placeholder for MAE calculation
        pass
    return metrics

def _standard_normalizer(data: np.ndarray) -> np.ndarray:
    """Standard normalizer for covariates."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / (std + 1e-8)

def _minmax_normalizer(data: np.ndarray) -> np.ndarray:
    """Min-max normalizer for covariates."""
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    return (data - min_val) / (max_val - min_val + 1e-8)

def _robust_normalizer(data: np.ndarray) -> np.ndarray:
    """Robust normalizer for covariates."""
    median = np.median(data, axis=0)
    iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
    return (data - median) / (iqr + 1e-8)

################################################################################
# martingale_residuals
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(
    time: np.ndarray,
    event: np.ndarray,
    predictions: np.ndarray,
    normalizer: Optional[Callable] = None
) -> None:
    """
    Validate input data for martingale residuals calculation.

    Parameters
    ----------
    time : np.ndarray
        Array of event times.
    event : np.ndarray
        Array indicating if the event occurred (1) or was censored (0).
    predictions : np.ndarray
        Predicted survival probabilities.
    normalizer : Optional[Callable]
        Function to normalize the residuals.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if len(time) != len(event):
        raise ValueError("Time and event arrays must have the same length")
    if len(time) != len(predictions):
        raise ValueError("Time and predictions arrays must have the same length")
    if np.any((time <= 0) | (event < 0) | (event > 1)):
        raise ValueError("Invalid values in time or event arrays")
    if np.any(predictions < 0) or np.any(predictions > 1):
        raise ValueError("Predictions must be between 0 and 1")

def compute_martingale_residuals(
    time: np.ndarray,
    event: np.ndarray,
    predictions: np.ndarray
) -> np.ndarray:
    """
    Compute martingale residuals.

    Parameters
    ----------
    time : np.ndarray
        Array of event times.
    event : np.ndarray
        Array indicating if the event occurred (1) or was censored (0).
    predictions : np.ndarray
        Predicted survival probabilities.

    Returns
    ------
    np.ndarray
        Martingale residuals.
    """
    return event - predictions

def normalize_residuals(
    residuals: np.ndarray,
    normalizer: Optional[Callable] = None
) -> np.ndarray:
    """
    Normalize residuals using the provided normalizer function.

    Parameters
    ----------
    residuals : np.ndarray
        Array of residuals.
    normalizer : Optional[Callable]
        Function to normalize the residuals.

    Returns
    ------
    np.ndarray
        Normalized residuals.
    """
    if normalizer is not None:
        return normalizer(residuals)
    return residuals

def compute_metrics(
    residuals: np.ndarray,
    metric_func: Callable
) -> Dict[str, float]:
    """
    Compute metrics for the residuals.

    Parameters
    ----------
    residuals : np.ndarray
        Array of residuals.
    metric_func : Callable
        Function to compute the metrics.

    Returns
    ------
    Dict[str, float]
        Dictionary of computed metrics.
    """
    return {"metric": metric_func(residuals)}

def martingale_residuals_fit(
    time: np.ndarray,
    event: np.ndarray,
    predictions: np.ndarray,
    normalizer: Optional[Callable] = None,
    metric_func: Callable = np.mean,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, Optional]]]:
    """
    Compute martingale residuals with optional normalization and metrics.

    Parameters
    ----------
    time : np.ndarray
        Array of event times.
    event : np.ndarray
        Array indicating if the event occurred (1) or was censored (0).
    predictions : np.ndarray
        Predicted survival probabilities.
    normalizer : Optional[Callable]
        Function to normalize the residuals. Default is None.
    metric_func : Callable
        Function to compute metrics on residuals. Default is np.mean.

    Returns
    ------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, Optional]]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> time = np.array([1.0, 2.0, 3.0])
    >>> event = np.array([1, 0, 1])
    >>> predictions = np.array([0.5, 0.3, 0.7])
    >>> result = martingale_residuals_fit(time, event, predictions)
    """
    validate_inputs(time, event, predictions, normalizer)

    residuals = compute_martingale_residuals(time, event, predictions)
    normalized_residuals = normalize_residuals(residuals, normalizer)
    metrics = compute_metrics(normalized_residuals, metric_func)

    return {
        "result": normalized_residuals,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric_func": metric_func.__name__
        },
        "warnings": {}
    }

################################################################################
# schonfeld_residuals
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def schonfeld_residuals_fit(
    time: np.ndarray,
    event: np.ndarray,
    covariates: np.ndarray,
    baseline_hazard: Callable[[np.ndarray], np.ndarray],
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    solver: str = 'newton',
    tol: float = 1e-6,
    max_iter: int = 1000,
    metric: str = 'mse',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Compute Schonfeld residuals for a Cox proportional hazards model.

    Parameters
    ----------
    time : np.ndarray
        Array of event/censoring times.
    event : np.ndarray
        Binary array indicating if the event occurred (1) or was censored (0).
    covariates : np.ndarray
        Matrix of shape (n_samples, n_features) containing the covariates.
    baseline_hazard : Callable[[np.ndarray], np.ndarray]
        Function that computes the baseline hazard at given times.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize covariates, by default None.
    solver : str, optional
        Solver method ('newton', 'gradient_descent'), by default 'newton'.
    tol : float, optional
        Tolerance for convergence, by default 1e-6.
    max_iter : int, optional
        Maximum number of iterations, by default 1000.
    metric : str, optional
        Metric to evaluate ('mse', 'mae'), by default 'mse'.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function, by default None.
    **kwargs : dict
        Additional solver-specific parameters.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], str]]
        Dictionary containing:
        - 'result': Array of Schonfeld residuals.
        - 'metrics': Dictionary of computed metrics.
        - 'params_used': Dictionary of parameters used.
        - 'warnings': List of warnings.

    Example
    -------
    >>> time = np.array([1.0, 2.0, 3.0])
    >>> event = np.array([1, 0, 1])
    >>> covariates = np.random.rand(3, 2)
    >>> baseline_hazard = lambda t: np.ones_like(t)
    >>> result = schonfeld_residuals_fit(time, event, covariates, baseline_hazard)
    """
    # Validate inputs
    _validate_inputs(time, event, covariates)

    # Normalize covariates if specified
    if normalizer is not None:
        covariates = normalizer(covariates)

    # Estimate parameters
    beta = _estimate_parameters(time, event, covariates, baseline_hazard, solver, tol, max_iter, **kwargs)

    # Compute Schonfeld residuals
    residuals = _compute_schonfeld_residuals(time, event, covariates, beta, baseline_hazard)

    # Compute metrics
    metrics = _compute_metrics(residuals, covariates, metric, custom_metric)

    return {
        'result': residuals,
        'metrics': metrics,
        'params_used': {
            'solver': solver,
            'tol': tol,
            'max_iter': max_iter,
            'metric': metric
        },
        'warnings': []
    }

def _validate_inputs(time: np.ndarray, event: np.ndarray, covariates: np.ndarray) -> None:
    """Validate input arrays."""
    if time.shape[0] != event.shape[0]:
        raise ValueError("time and event must have the same length")
    if covariates.shape[0] != time.shape[0]:
        raise ValueError("covariates must have the same number of samples as time and event")
    if np.any(np.isnan(time)) or np.any(np.isinf(time)):
        raise ValueError("time contains NaN or inf values")
    if np.any(np.isnan(event)) or np.any(np.isinf(event)):
        raise ValueError("event contains NaN or inf values")
    if np.any(np.isnan(covariates)) or np.any(np.isinf(covariates)):
        raise ValueError("covariates contains NaN or inf values")

def _estimate_parameters(
    time: np.ndarray,
    event: np.ndarray,
    covariates: np.ndarray,
    baseline_hazard: Callable[[np.ndarray], np.ndarray],
    solver: str,
    tol: float,
    max_iter: int,
    **kwargs
) -> np.ndarray:
    """Estimate parameters using the specified solver."""
    if solver == 'newton':
        return _newton_solver(time, event, covariates, baseline_hazard, tol, max_iter)
    elif solver == 'gradient_descent':
        return _gradient_descent_solver(time, event, covariates, baseline_hazard, tol, max_iter, **kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _newton_solver(
    time: np.ndarray,
    event: np.ndarray,
    covariates: np.ndarray,
    baseline_hazard: Callable[[np.ndarray], np.ndarray],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Newton-Raphson solver for Cox model."""
    beta = np.zeros(covariates.shape[1])
    for _ in range(max_iter):
        score, hessian = _compute_score_and_hessian(time, event, covariates, beta, baseline_hazard)
        if np.linalg.norm(score) < tol:
            break
        beta -= np.linalg.solve(hessian, score)
    return beta

def _gradient_descent_solver(
    time: np.ndarray,
    event: np.ndarray,
    covariates: np.ndarray,
    baseline_hazard: Callable[[np.ndarray], np.ndarray],
    tol: float,
    max_iter: int,
    learning_rate: float = 0.01
) -> np.ndarray:
    """Gradient descent solver for Cox model."""
    beta = np.zeros(covariates.shape[1])
    for _ in range(max_iter):
        score = _compute_score(time, event, covariates, beta, baseline_hazard)
        if np.linalg.norm(score) < tol:
            break
        beta -= learning_rate * score
    return beta

def _compute_score_and_hessian(
    time: np.ndarray,
    event: np.ndarray,
    covariates: np.ndarray,
    beta: np.ndarray,
    baseline_hazard: Callable[[np.ndarray], np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute score and Hessian for Newton-Raphson."""
    risk = covariates @ beta
    exp_risk = np.exp(risk)
    sum_exp_risk = np.sum(exp_risk)

    # Compute score
    score = covariates.T @ (event - exp_risk / sum_exp_risk)

    # Compute Hessian
    hessian = -covariates.T @ (exp_risk * (1 - exp_risk) / sum_exp_risk**2) @ covariates

    return score, hessian

def _compute_score(
    time: np.ndarray,
    event: np.ndarray,
    covariates: np.ndarray,
    beta: np.ndarray,
    baseline_hazard: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Compute score for gradient descent."""
    risk = covariates @ beta
    exp_risk = np.exp(risk)
    sum_exp_risk = np.sum(exp_risk)

    return covariates.T @ (event - exp_risk / sum_exp_risk)

def _compute_schonfeld_residuals(
    time: np.ndarray,
    event: np.ndarray,
    covariates: np.ndarray,
    beta: np.ndarray,
    baseline_hazard: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Compute Schonfeld residuals."""
    risk = covariates @ beta
    exp_risk = np.exp(risk)
    sum_exp_risk = np.sum(exp_risk)

    # Compute martingale residuals
    martingale_residuals = event - exp_risk / sum_exp_risk

    # Compute Schonfeld residuals
    schonfeld_residuals = covariates * (martingale_residuals / baseline_hazard(time)[:, np.newaxis])

    return schonfeld_residuals

def _compute_metrics(
    residuals: np.ndarray,
    covariates: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute metrics for the residuals."""
    metrics = {}
    if custom_metric is not None:
        metrics['custom'] = custom_metric(residuals, covariates)
    else:
        if metric == 'mse':
            metrics['mse'] = np.mean(residuals**2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(residuals))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    return metrics

################################################################################
# deviances_residuals
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> None:
    """Validate input arrays for deviances residuals calculation."""
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-dimensional arrays")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or infinite values")
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError("y_pred contains NaN or infinite values")
    if weights is not None:
        if len(weights) != len(y_true):
            raise ValueError("weights must have the same length as y_true")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")

def compute_deviance_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute deviance residuals for survival regression."""
    validate_inputs(y_true, y_pred, weights)
    if weights is None:
        weights = np.ones_like(y_true)

    # Calculate deviance residuals (simplified for demonstration)
    residuals = y_true - y_pred
    if np.any(y_pred <= 0):
        raise ValueError("y_pred must be strictly positive for deviance calculation")
    return residuals * np.sqrt(2 * (y_true * np.log(y_true / y_pred) + (1 - y_true) * np.log((1 - y_true) / (1 - y_pred))))

def compute_metrics(
    residuals: np.ndarray,
    metric_funcs: Dict[str, Callable[[np.ndarray], float]],
    weights: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute various metrics from residuals."""
    if weights is None:
        weights = np.ones_like(residuals)

    metrics = {}
    for name, func in metric_funcs.items():
        try:
            if weights is not None:
                metrics[name] = func(residuals * np.sqrt(weights))
            else:
                metrics[name] = func(residuals)
        except Exception as e:
            metrics[f"{name}_error"] = str(e)

    return metrics

def deviances_residuals_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_funcs: Dict[str, Callable[[np.ndarray], float]] = None,
    weights: Optional[np.ndarray] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Compute deviance residuals and metrics for survival regression.

    Parameters:
    -----------
    y_true : np.ndarray
        True values (0 or 1 for binary survival)
    y_pred : np.ndarray
        Predicted probabilities/survival times
    metric_funcs : Dict[str, Callable]
        Dictionary of metric functions to compute (default: MSE and MAE)
    weights : Optional[np.ndarray]
        Sample weights

    Returns:
    --------
    Dict containing:
        - "result": deviance residuals
        - "metrics": computed metrics
        - "params_used": parameters used in computation
        - "warnings": any warnings generated

    Example:
    --------
    >>> y_true = np.array([0, 1, 0, 1])
    >>> y_pred = np.array([0.2, 0.8, 0.3, 0.7])
    >>> result = deviances_residuals_fit(y_true, y_pred)
    """
    if metric_funcs is None:
        metric_funcs = {
            "mse": lambda x: np.mean(x**2),
            "mae": lambda x: np.mean(np.abs(x))
        }

    warnings = []

    try:
        residuals = compute_deviance_residuals(y_true, y_pred, weights)
    except ValueError as e:
        return {
            "result": None,
            "metrics": {},
            "params_used": {"metric_funcs": metric_funcs, "weights": weights},
            "warnings": [str(e)]
        }

    metrics = compute_metrics(residuals, metric_funcs, weights)

    return {
        "result": residuals,
        "metrics": metrics,
        "params_used": {"metric_funcs": metric_funcs, "weights": weights},
        "warnings": warnings
    }

################################################################################
# cox_snell_residuals
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def cox_snell_residuals_fit(
    survival_times: np.ndarray,
    event_occurred: np.ndarray,
    model_predictions: np.ndarray,
    *,
    normalization: str = 'none',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Compute Cox-Snell residuals for survival analysis.

    Parameters:
    -----------
    survival_times : np.ndarray
        Array of observed survival times.
    event_occurred : np.ndarray
        Binary array indicating if the event occurred (1) or was censored (0).
    model_predictions : np.ndarray
        Predicted survival probabilities from the model.
    normalization : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : str or callable, optional
        Metric to evaluate residuals: 'mse', 'mae', 'r2', or custom callable.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', or 'newton'.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.

    Returns:
    --------
    Dict containing:
        - 'result': Computed Cox-Snell residuals.
        - 'metrics': Dictionary of computed metrics.
        - 'params_used': Parameters used in the computation.
        - 'warnings': List of warnings encountered.

    Example:
    --------
    >>> survival_times = np.array([1.2, 3.4, 5.6])
    >>> event_occurred = np.array([1, 0, 1])
    >>> model_predictions = np.array([0.8, 0.3, 0.6])
    >>> result = cox_snell_residuals_fit(survival_times, event_occurred, model_predictions)
    """
    # Validate inputs
    _validate_inputs(survival_times, event_occurred, model_predictions)

    # Normalize data if required
    normalized_times = _normalize(survival_times, method=normalization)

    # Compute Cox-Snell residuals
    residuals = _compute_cox_snell_residuals(normalized_times, event_occurred, model_predictions)

    # Choose and compute metric
    metrics = _compute_metrics(residuals, event_occurred, metric=metric, custom_metric=custom_metric)

    # Prepare output
    result_dict = {
        'result': residuals,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'solver': solver,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return result_dict

def _validate_inputs(
    survival_times: np.ndarray,
    event_occurred: np.ndarray,
    model_predictions: np.ndarray
) -> None:
    """Validate input arrays."""
    if survival_times.shape != event_occurred.shape or survival_times.shape != model_predictions.shape:
        raise ValueError("Input arrays must have the same shape.")
    if np.any(survival_times <= 0):
        raise ValueError("Survival times must be positive.")
    if np.any((event_occurred != 0) & (event_occurred != 1)):
        raise ValueError("Event occurred must be binary (0 or 1).")
    if np.any((model_predictions < 0) | (model_predictions > 1)):
        raise ValueError("Model predictions must be between 0 and 1.")

def _normalize(
    data: np.ndarray,
    method: str = 'none'
) -> np.ndarray:
    """Normalize data using specified method."""
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

def _compute_cox_snell_residuals(
    survival_times: np.ndarray,
    event_occurred: np.ndarray,
    model_predictions: np.ndarray
) -> np.ndarray:
    """Compute Cox-Snell residuals."""
    # Transform survival times to cumulative hazard
    cum_hazard = -np.log(1 - model_predictions)
    # Compute residuals
    residuals = event_occurred * (survival_times - cum_hazard)
    return residuals

def _compute_metrics(
    residuals: np.ndarray,
    event_occurred: np.ndarray,
    metric: Union[str, Callable] = 'mse',
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Compute specified metrics for residuals."""
    metrics_dict = {}

    if metric == 'mse':
        metrics_dict['mse'] = np.mean(residuals**2)
    elif metric == 'mae':
        metrics_dict['mae'] = np.mean(np.abs(residuals))
    elif metric == 'r2':
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((event_occurred - np.mean(event_occurred))**2)
        metrics_dict['r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    elif callable(metric):
        metrics_dict['custom'] = metric(residuals, event_occurred)
    elif custom_metric is not None:
        metrics_dict['custom'] = custom_metric(residuals, event_occurred)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics_dict

################################################################################
# stratified_cox_model
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def stratified_cox_model_fit(
    time: np.ndarray,
    event: np.ndarray,
    covariates: np.ndarray,
    strata: Optional[np.ndarray] = None,
    normalization: str = 'standard',
    metric: Union[str, Callable] = 'logloss',
    solver: str = 'newton',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_solver: Optional[Callable] = None
) -> Dict:
    """
    Fit a stratified Cox proportional hazards model.

    Parameters:
    -----------
    time : np.ndarray
        Array of survival times.
    event : np.ndarray
        Binary array indicating if the event occurred (1) or was censored (0).
    covariates : np.ndarray
        2D array of shape (n_samples, n_features) containing the covariates.
    strata : Optional[np.ndarray]
        Array of shape (n_samples,) indicating stratum for each sample.
    normalization : str
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : Union[str, Callable]
        Metric to evaluate model performance: 'logloss', 'concordance', or custom callable.
    solver : str
        Solver method: 'newton', 'gradient_descent', or custom callable.
    regularization : Optional[str]
        Regularization type: 'none', 'l1', 'l2', or 'elasticnet'.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable]
        Custom metric function if not using built-in metrics.
    custom_solver : Optional[Callable]
        Custom solver function if not using built-in solvers.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(time, event, covariates, strata)

    # Normalize covariates
    if normalization != 'none':
        covariates = _normalize(covariates, method=normalization)

    # Initialize parameters
    n_features = covariates.shape[1]
    beta = np.zeros(n_features)

    # Choose solver
    if custom_solver is not None:
        beta = custom_solver(covariates, event, time, strata, beta, tol, max_iter)
    else:
        if solver == 'newton':
            beta = _newton_solver(covariates, event, time, strata, beta, tol, max_iter)
        elif solver == 'gradient_descent':
            beta = _gradient_descent_solver(covariates, event, time, strata, beta, tol, max_iter)
        else:
            raise ValueError("Unsupported solver. Choose 'newton', 'gradient_descent', or provide a custom solver.")

    # Apply regularization if specified
    if regularization is not None:
        beta = _apply_regularization(beta, covariates, event, time, strata, regularization)

    # Compute metrics
    if custom_metric is not None:
        metric_value = custom_metric(covariates, event, time, strata, beta)
    else:
        if metric == 'logloss':
            metric_value = _compute_logloss(covariates, event, time, strata, beta)
        elif metric == 'concordance':
            metric_value = _compute_concordance(covariates, event, time, strata, beta)
        else:
            raise ValueError("Unsupported metric. Choose 'logloss', 'concordance', or provide a custom metric.")

    # Prepare output
    result = {
        'result': beta,
        'metrics': {'metric_name': metric, 'value': metric_value},
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return result

def _validate_inputs(time: np.ndarray, event: np.ndarray, covariates: np.ndarray, strata: Optional[np.ndarray]) -> None:
    """
    Validate input arrays.
    """
    if len(time) != len(event):
        raise ValueError("Time and event arrays must have the same length.")
    if covariates.shape[0] != len(time):
        raise ValueError("Number of samples in covariates must match length of time and event arrays.")
    if strata is not None and len(strata) != len(time):
        raise ValueError("Strata array must have the same length as time and event arrays.")
    if np.any(np.isnan(time)) or np.any(np.isinf(time)):
        raise ValueError("Time array contains NaN or infinite values.")
    if np.any(np.isnan(event)) or np.any(np.isinf(event)):
        raise ValueError("Event array contains NaN or infinite values.")
    if np.any(np.isnan(covariates)) or np.any(np.isinf(covariates)):
        raise ValueError("Covariates array contains NaN or infinite values.")

def _normalize(covariates: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Normalize covariates.
    """
    if method == 'standard':
        mean = np.mean(covariates, axis=0)
        std = np.std(covariates, axis=0)
        return (covariates - mean) / std
    elif method == 'minmax':
        min_val = np.min(covariates, axis=0)
        max_val = np.max(covariates, axis=0)
        return (covariates - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(covariates, axis=0)
        iqr = np.subtract(*np.percentile(covariates, [75, 25], axis=0))
        return (covariates - median) / (iqr + 1e-8)
    else:
        raise ValueError("Unsupported normalization method.")

def _newton_solver(
    covariates: np.ndarray,
    event: np.ndarray,
    time: np.ndarray,
    strata: Optional[np.ndarray],
    beta: np.ndarray,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """
    Newton-Raphson solver for Cox model.
    """
    for _ in range(max_iter):
        # Compute gradient and Hessian
        grad, hess = _compute_grad_hess(covariates, event, time, strata, beta)

        # Update beta
        delta = np.linalg.solve(hess, grad)
        beta -= delta

        # Check convergence
        if np.linalg.norm(delta) < tol:
            break
    return beta

def _gradient_descent_solver(
    covariates: np.ndarray,
    event: np.ndarray,
    time: np.ndarray,
    strata: Optional[np.ndarray],
    beta: np.ndarray,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """
    Gradient descent solver for Cox model.
    """
    learning_rate = 0.01
    for _ in range(max_iter):
        # Compute gradient
        grad = _compute_grad(covariates, event, time, strata, beta)

        # Update beta
        beta -= learning_rate * grad

        # Check convergence
        if np.linalg.norm(grad) < tol:
            break
    return beta

def _compute_grad_hess(
    covariates: np.ndarray,
    event: np.ndarray,
    time: np.ndarray,
    strata: Optional[np.ndarray],
    beta: np.ndarray
) -> tuple:
    """
    Compute gradient and Hessian for Newton-Raphson.
    """
    risk = np.exp(np.dot(covariates, beta))
    grad = np.zeros_like(beta)
    hess = np.zeros((len(beta), len(beta)))

    # Implement stratified computation if strata is provided
    if strata is not None:
        unique_strata = np.unique(strata)
        for s in unique_strata:
            mask = strata == s
            grad_s, hess_s = _compute_grad_hess_stratum(covariates[mask], event[mask], time[mask], beta)
            grad += grad_s
            hess += hess_s
    else:
        grad, hess = _compute_grad_hess_stratum(covariates, event, time, beta)

    return grad, hess

def _compute_grad_hess_stratum(
    covariates: np.ndarray,
    event: np.ndarray,
    time: np.ndarray,
    beta: np.ndarray
) -> tuple:
    """
    Compute gradient and Hessian for a single stratum.
    """
    risk = np.exp(np.dot(covariates, beta))
    grad = np.zeros_like(beta)
    hess = np.zeros((len(beta), len(beta)))

    # Implement Cox model gradient and Hessian computation
    # This is a placeholder for the actual implementation
    return grad, hess

def _compute_grad(
    covariates: np.ndarray,
    event: np.ndarray,
    time: np.ndarray,
    strata: Optional[np.ndarray],
    beta: np.ndarray
) -> np.ndarray:
    """
    Compute gradient for gradient descent.
    """
    grad = np.zeros_like(beta)

    # Implement stratified computation if strata is provided
    if strata is not None:
        unique_strata = np.unique(strata)
        for s in unique_strata:
            mask = strata == s
            grad_s = _compute_grad_stratum(covariates[mask], event[mask], time[mask], beta)
            grad += grad_s
    else:
        grad = _compute_grad_stratum(covariates, event, time, beta)

    return grad

def _compute_grad_stratum(
    covariates: np.ndarray,
    event: np.ndarray,
    time: np.ndarray,
    beta: np.ndarray
) -> np.ndarray:
    """
    Compute gradient for a single stratum.
    """
    grad = np.zeros_like(beta)

    # Implement Cox model gradient computation
    # This is a placeholder for the actual implementation
    return grad

def _apply_regularization(
    beta: np.ndarray,
    covariates: np.ndarray,
    event: np.ndarray,
    time: np.ndarray,
    strata: Optional[np.ndarray],
    regularization: str
) -> np.ndarray:
    """
    Apply regularization to the coefficients.
    """
    if regularization == 'l1':
        # L1 regularization (Lasso)
        pass
    elif regularization == 'l2':
        # L2 regularization (Ridge)
        pass
    elif regularization == 'elasticnet':
        # Elastic net regularization
        pass
    else:
        raise ValueError("Unsupported regularization type.")
    return beta

def _compute_logloss(
    covariates: np.ndarray,
    event: np.ndarray,
    time: np.ndarray,
    strata: Optional[np.ndarray],
    beta: np.ndarray
) -> float:
    """
    Compute log-loss metric.
    """
    # Implement stratified computation if strata is provided
    if strata is not None:
        unique_strata = np.unique(strata)
        logloss = 0.0
        for s in unique_strata:
            mask = strata == s
            logloss += _compute_logloss_stratum(covariates[mask], event[mask], time[mask], beta)
        return logloss / len(unique_strata)
    else:
        return _compute_logloss_stratum(covariates, event, time, beta)

def _compute_logloss_stratum(
    covariates: np.ndarray,
    event: np.ndarray,
    time: np.ndarray,
    beta: np.ndarray
) -> float:
    """
    Compute log-loss for a single stratum.
    """
    # Implement Cox model log-loss computation
    # This is a placeholder for the actual implementation
    return 0.0

def _compute_concordance(
    covariates: np.ndarray,
    event: np.ndarray,
    time: np.ndarray,
    strata: Optional[np.ndarray],
    beta: np.ndarray
) -> float:
    """
    Compute concordance metric.
    """
    # Implement stratified computation if strata is provided
    if strata is not None:
        unique_strata = np.unique(strata)
        concordance = 0.0
        for s in unique_strata:
            mask = strata == s
            concordance += _compute_concordance_stratum(covariates[mask], event[mask], time[mask], beta)
        return concordance / len(unique_strata)
    else:
        return _compute_concordance_stratum(covariates, event, time, beta)

def _compute_concordance_stratum(
    covariates: np.ndarray,
    event: np.ndarray,
    time: np.ndarray,
    beta: np.ndarray
) -> float:
    """
    Compute concordance for a single stratum.
    """
    # Implement Cox model concordance computation
    # This is a placeholder for the actual implementation
    return 0.0

################################################################################
# time_varying_effects
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def time_varying_effects_fit(
    X: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
    time_points: Optional[np.ndarray] = None,
    normalizer: str = "standard",
    metric: Union[str, Callable] = "mse",
    solver: str = "gradient_descent",
    regularization: Optional[str] = None,
    max_iter: int = 1000,
    tol: float = 1e-4,
    learning_rate: float = 0.01,
    alpha: float = 0.0,
    beta: float = 0.0,
    random_state: Optional[int] = None
) -> Dict:
    """
    Fit a time-varying effects model for survival analysis.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    T : np.ndarray
        Survival times of shape (n_samples,).
    E : np.ndarray
        Event indicators (1 if event occurred, 0 otherwise) of shape (n_samples,).
    time_points : np.ndarray, optional
        Time points at which to evaluate the effects. If None, uses unique sorted T.
    normalizer : str
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : str or callable
        Metric to optimize: 'mse', 'mae', 'r2', or custom callable.
    solver : str
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'.
    regularization : str, optional
        Regularization type: 'none', 'l1', 'l2', or 'elasticnet'.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for convergence.
    learning_rate : float
        Learning rate for gradient-based solvers.
    alpha : float
        L1 regularization strength (if applicable).
    beta : float
        L2 regularization strength (if applicable).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Fitted model parameters.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings generated.

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> T = np.random.exponential(size=100)
    >>> E = np.random.randint(0, 2, size=100)
    >>> result = time_varying_effects_fit(X, T, E)
    """
    # Validate inputs
    _validate_inputs(X, T, E)

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Default time points
    if time_points is None:
        time_points = np.unique(T)

    # Normalize features
    X_normalized, normalizer_used = _normalize_features(X, method=normalizer)

    # Initialize parameters
    n_features = X_normalized.shape[1]
    coefs = np.zeros((len(time_points), n_features))

    # Choose solver
    if solver == "closed_form":
        coefs = _closed_form_solver(X_normalized, T, E, time_points)
    elif solver == "gradient_descent":
        coefs = _gradient_descent_solver(
            X_normalized, T, E, time_points,
            max_iter=max_iter, tol=tol,
            learning_rate=learning_rate,
            alpha=alpha, beta=beta
        )
    elif solver == "newton":
        coefs = _newton_solver(X_normalized, T, E, time_points)
    elif solver == "coordinate_descent":
        coefs = _coordinate_descent_solver(
            X_normalized, T, E, time_points,
            max_iter=max_iter, tol=tol
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_normalized, T, E, coefs, time_points, metric)

    # Prepare output
    result = {
        "result": coefs,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer_used,
            "metric": metric,
            "solver": solver,
            "regularization": regularization,
            "max_iter": max_iter,
            "tol": tol,
            "learning_rate": learning_rate,
            "alpha": alpha,
            "beta": beta
        },
        "warnings": []
    }

    return result

def _validate_inputs(X: np.ndarray, T: np.ndarray, E: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if T.ndim != 1:
        raise ValueError("T must be a 1D array")
    if E.ndim != 1:
        raise ValueError("E must be a 1D array")
    if len(X) != len(T) or len(X) != len(E):
        raise ValueError("X, T, and E must have the same length")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(T <= 0):
        raise ValueError("T must contain positive values only")
    if np.any((E != 0) & (E != 1)):
        raise ValueError("E must contain only 0 or 1")

def _normalize_features(X: np.ndarray, method: str = "standard") -> tuple:
    """Normalize features."""
    X_normalized = X.copy()
    normalizer_used = method

    if method == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_normalized = (X - mean) / std
    elif method == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_normalized = (X - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_normalized = (X - median) / (iqr + 1e-8)
    elif method != "none":
        raise ValueError(f"Unknown normalization method: {method}")

    return X_normalized, normalizer_used

def _closed_form_solver(X: np.ndarray, T: np.ndarray, E: np.ndarray,
                        time_points: np.ndarray) -> np.ndarray:
    """Closed form solution for time-varying effects."""
    # Placeholder implementation
    n_samples, n_features = X.shape
    coefs = np.zeros((len(time_points), n_features))

    for i, t in enumerate(time_points):
        # This is a simplified placeholder
        # Actual implementation would involve survival analysis specific calculations
        mask = T >= t
        if np.sum(mask) > 0:
            X_subset = X[mask]
            E_subset = E[mask]
            coefs[i] = np.linalg.pinv(X_subset.T @ X_subset) @ X_subset.T @ E_subset

    return coefs

def _gradient_descent_solver(X: np.ndarray, T: np.ndarray, E: np.ndarray,
                             time_points: np.ndarray, max_iter: int = 1000,
                             tol: float = 1e-4, learning_rate: float = 0.01,
                             alpha: float = 0.0, beta: float = 0.0) -> np.ndarray:
    """Gradient descent solver for time-varying effects."""
    n_samples, n_features = X.shape
    coefs = np.zeros((len(time_points), n_features))

    for i, t in enumerate(time_points):
        mask = T >= t
        if np.sum(mask) > 0:
            X_subset = X[mask]
            E_subset = E[mask]

            # Initialize coefficients
            current_coefs = np.zeros(n_features)

            for _ in range(max_iter):
                # Compute predictions
                preds = X_subset @ current_coefs

                # Compute gradient (simplified)
                grad = -X_subset.T @ (E_subset - preds) / n_samples

                # Add regularization if needed
                if alpha > 0:
                    grad += alpha * np.sign(current_coefs)
                if beta > 0:
                    grad += 2 * beta * current_coefs

                # Update coefficients
                new_coefs = current_coefs - learning_rate * grad

                # Check convergence
                if np.linalg.norm(new_coefs - current_coefs) < tol:
                    break

                current_coefs = new_coefs

            coefs[i] = current_coefs

    return coefs

def _newton_solver(X: np.ndarray, T: np.ndarray, E: np.ndarray,
                   time_points: np.ndarray) -> np.ndarray:
    """Newton's method solver for time-varying effects."""
    # Placeholder implementation
    n_samples, n_features = X.shape
    coefs = np.zeros((len(time_points), n_features))

    for i, t in enumerate(time_points):
        mask = T >= t
        if np.sum(mask) > 0:
            X_subset = X[mask]
            E_subset = E[mask]

            # Initialize coefficients
            current_coefs = np.zeros(n_features)

            for _ in range(100):  # Fixed iterations for example
                # Compute predictions and residuals
                preds = X_subset @ current_coefs
                residuals = E_subset - preds

                # Compute Hessian (simplified)
                hessian = X_subset.T @ X_subset / n_samples

                # Compute gradient
                grad = -X_subset.T @ residuals / n_samples

                # Update coefficients using Newton's method
                current_coefs = current_coefs - np.linalg.pinv(hessian) @ grad

            coefs[i] = current_coefs

    return coefs

def _coordinate_descent_solver(X: np.ndarray, T: np.ndarray, E: np.ndarray,
                               time_points: np.ndarray, max_iter: int = 1000,
                               tol: float = 1e-4) -> np.ndarray:
    """Coordinate descent solver for time-varying effects."""
    # Placeholder implementation
    n_samples, n_features = X.shape
    coefs = np.zeros((len(time_points), n_features))

    for i, t in enumerate(time_points):
        mask = T >= t
        if np.sum(mask) > 0:
            X_subset = X[mask]
            E_subset = E[mask]

            # Initialize coefficients
            current_coefs = np.zeros(n_features)

            for _ in range(max_iter):
                old_coefs = current_coefs.copy()

                for j in range(n_features):
                    # Compute residual without feature j
                    X_j = X_subset[:, j]
                    residual = E_subset - (X_subset @ current_coefs - X_j * current_coefs[j])

                    # Compute optimal value for feature j
                    numerator = X_j.T @ residual
                    denominator = X_j.T @ X_j

                    if denominator != 0:
                        current_coefs[j] = numerator / denominator
                    else:
                        current_coefs[j] = 0

                # Check convergence
                if np.linalg.norm(current_coefs - old_coefs) < tol:
                    break

            coefs[i] = current_coefs

    return coefs

def _compute_metrics(X: np.ndarray, T: np.ndarray, E: np.ndarray,
                     coefs: np.ndarray, time_points: np.ndarray,
                     metric: Union[str, Callable]) -> Dict:
    """Compute metrics for the fitted model."""
    n_samples = X.shape[0]
    metrics = {}

    if callable(metric):
        # Custom metric
        preds = np.zeros(n_samples)
        for i, t in enumerate(time_points):
            mask = T >= t
            preds[mask] = X[mask] @ coefs[i]
        metrics["custom"] = metric(E, preds)
    else:
        # Built-in metrics
        for i, t in enumerate(time_points):
            mask = T >= t
            if np.sum(mask) > 0:
                preds = X[mask] @ coefs[i]

                if metric == "mse":
                    metrics[f"mse_{i}"] = np.mean((E[mask] - preds) ** 2)
                elif metric == "mae":
                    metrics[f"mae_{i}"] = np.mean(np.abs(E[mask] - preds))
                elif metric == "r2":
                    ss_res = np.sum((E[mask] - preds) ** 2)
                    ss_tot = np.sum((E[mask] - np.mean(E[mask])) ** 2)
                    metrics[f"r2_{i}"] = 1 - ss_res / (ss_tot + 1e-8)

    return metrics
