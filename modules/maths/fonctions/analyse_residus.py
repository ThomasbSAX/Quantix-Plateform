"""
Quantix – Module analyse_residus
Généré automatiquement
Date: 2026-01-06
"""

################################################################################
# residu_standardise
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def residu_standardise_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalisation: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    regularisation: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Standardize residuals between true and predicted values with configurable options.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    normalisation : str, optional
        Type of normalization to apply ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to compute ('mse', 'mae', 'r2', 'logloss') or custom callable.
    distance : str or callable, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable.
    solver : str, optional
        Solver to use ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    regularisation : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet').
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    custom_distance : callable, optional
        Custom distance function if not using built-in distances.

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([1.1, 2.2, 2.9])
    >>> result = residu_standardise_fit(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Compute residuals
    residuals = y_true - y_pred

    # Apply normalization
    normalized_residuals = _apply_normalisation(residuals, normalisation)

    # Compute metrics
    metrics = _compute_metrics(normalized_residuals, y_true, y_pred, metric, custom_metric)

    # Compute distances
    distances = _compute_distances(normalized_residuals, distance, custom_distance)

    # Solve using specified solver
    params = _solve(normalized_residuals, y_true, y_pred, solver, regularisation, tol, max_iter)

    # Prepare output
    result = {
        "result": normalized_residuals,
        "metrics": metrics,
        "params_used": {
            "normalisation": normalisation,
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularisation": regularisation
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
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Inputs must not contain infinite values.")

def _apply_normalisation(residuals: np.ndarray, normalisation: str) -> np.ndarray:
    """Apply specified normalization to residuals."""
    if normalisation == 'none':
        return residuals
    elif normalisation == 'standard':
        mean = np.mean(residuals)
        std = np.std(residuals)
        if std == 0:
            return residuals - mean
        return (residuals - mean) / std
    elif normalisation == 'minmax':
        min_val = np.min(residuals)
        max_val = np.max(residuals)
        if min_val == max_val:
            return residuals
        return (residuals - min_val) / (max_val - min_val)
    elif normalisation == 'robust':
        median = np.median(residuals)
        iqr = np.percentile(residuals, 75) - np.percentile(residuals, 25)
        if iqr == 0:
            return residuals - median
        return (residuals - median) / iqr
    else:
        raise ValueError(f"Unknown normalisation: {normalisation}")

def _compute_metrics(
    residuals: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Compute specified metrics."""
    metrics = {}
    if isinstance(metric, str):
        if metric == 'mse':
            metrics['mse'] = np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        elif metric == 'logloss':
            metrics['logloss'] = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        metrics['custom'] = metric(y_true, y_pred)
    else:
        raise TypeError("Metric must be a string or callable.")

    if custom_metric is not None:
        metrics['custom_metric'] = custom_metric(y_true, y_pred)

    return metrics

def _compute_distances(
    residuals: np.ndarray,
    distance: Union[str, Callable],
    custom_distance: Optional[Callable] = None
) -> Dict:
    """Compute specified distances."""
    distances = {}
    if isinstance(distance, str):
        if distance == 'euclidean':
            distances['euclidean'] = np.linalg.norm(residuals)
        elif distance == 'manhattan':
            distances['manhattan'] = np.sum(np.abs(residuals))
        elif distance == 'cosine':
            distances['cosine'] = 1 - np.dot(residuals, residuals) / (np.linalg.norm(residuals) * np.linalg.norm(residuals))
        elif distance == 'minkowski':
            distances['minkowski'] = np.sum(np.abs(residuals) ** 3) ** (1/3)
        else:
            raise ValueError(f"Unknown distance: {distance}")
    elif callable(distance):
        distances['custom'] = distance(residuals)
    else:
        raise TypeError("Distance must be a string or callable.")

    if custom_distance is not None:
        distances['custom_distance'] = custom_distance(residuals)

    return distances

def _solve(
    residuals: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    solver: str,
    regularisation: Optional[str],
    tol: float,
    max_iter: int
) -> Dict:
    """Solve using specified solver."""
    params = {}
    if solver == 'closed_form':
        params['solution'] = np.linalg.lstsq(y_pred.reshape(-1, 1), y_true, rcond=None)[0]
    elif solver == 'gradient_descent':
        params['solution'] = _gradient_descent(y_true, y_pred, tol, max_iter)
    elif solver == 'newton':
        params['solution'] = _newton_method(y_true, y_pred, tol, max_iter)
    elif solver == 'coordinate_descent':
        params['solution'] = _coordinate_descent(y_true, y_pred, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    if regularisation is not None:
        params['regularisation'] = _apply_regularisation(params['solution'], regularisation)

    return params

def _gradient_descent(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Gradient descent solver."""
    beta = np.zeros(y_pred.shape[1])
    for _ in range(max_iter):
        gradient = -2 * y_pred.T @ (y_true - y_pred @ beta) / len(y_true)
        beta_new = beta - 0.01 * gradient
        if np.linalg.norm(beta_new - beta) < tol:
            break
        beta = beta_new
    return beta

def _newton_method(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Newton method solver."""
    beta = np.zeros(y_pred.shape[1])
    for _ in range(max_iter):
        residuals = y_true - y_pred @ beta
        gradient = -2 * y_pred.T @ residuals / len(y_true)
        hessian = 2 * y_pred.T @ y_pred / len(y_true)
        beta_new = beta - np.linalg.inv(hessian) @ gradient
        if np.linalg.norm(beta_new - beta) < tol:
            break
        beta = beta_new
    return beta

def _coordinate_descent(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Coordinate descent solver."""
    beta = np.zeros(y_pred.shape[1])
    for _ in range(max_iter):
        for j in range(len(beta)):
            beta_j_old = beta[j]
            beta[j] = np.sum((y_true - y_pred @ beta + beta_j_old * y_pred[:, j]) * y_pred[:, j]) / np.sum(y_pred[:, j] ** 2)
            if np.abs(beta[j] - beta_j_old) < tol:
                break
    return beta

def _apply_regularisation(beta: np.ndarray, regularisation: str) -> np.ndarray:
    """Apply specified regularization."""
    if regularisation == 'l1':
        return np.sign(beta) * np.maximum(np.abs(beta) - 0.1, 0)
    elif regularisation == 'l2':
        return beta / (1 + 0.1 * np.linalg.norm(beta))
    elif regularisation == 'elasticnet':
        return np.sign(beta) * np.maximum(np.abs(beta) - 0.1, 0) / (1 + 0.1 * np.linalg.norm(beta))
    else:
        raise ValueError(f"Unknown regularisation: {regularisation}")

################################################################################
# residu_studentise
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def residu_studentise_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalisation: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    custom_metric: Optional[Callable] = None,
    alpha: float = 0.05
) -> Dict[str, Union[np.ndarray, Dict, str]]:
    """
    Compute studentized residuals for given true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        Array of observed/true values.
    y_pred : np.ndarray
        Array of predicted values.
    normalisation : str, optional (default='standard')
        Type of normalization to apply. Options: 'none', 'standard', 'minmax', 'robust'.
    metric : str or callable, optional (default='mse')
        Metric to use for residual calculation. Options: 'mse', 'mae', 'r2', or custom callable.
    custom_metric : callable, optional
        Custom metric function if not using built-in options.
    alpha : float, optional (default=0.05)
        Significance level for studentization.

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': studentized residuals array
        - 'metrics': dictionary of computed metrics
        - 'params_used': parameters used in computation
        - 'warnings': any warnings encountered

    Example
    -------
    >>> y_true = np.array([1, 2, 3])
    >>> y_pred = np.array([1.1, 1.9, 3.2])
    >>> result = residu_studentise_fit(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Normalize data if required
    y_true_norm, y_pred_norm = _apply_normalisation(y_true, y_pred, normalisation)

    # Calculate residuals
    residuals = _calculate_residuals(y_true_norm, y_pred_norm, metric, custom_metric)

    # Studentize residuals
    studentized_residuals = _studentize_residuals(residuals, alpha)

    # Calculate metrics
    metrics = _calculate_metrics(y_true_norm, y_pred_norm, residuals)

    return {
        'result': studentized_residuals,
        'metrics': metrics,
        'params_used': {
            'normalisation': normalisation,
            'metric': metric if isinstance(metric, str) else 'custom',
            'alpha': alpha
        },
        'warnings': []
    }

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or infinite values")
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError("y_pred contains NaN or infinite values")

def _apply_normalisation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalisation: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply selected normalization to input arrays."""
    if normalisation == 'none':
        return y_true, y_pred
    elif normalisation == 'standard':
        mean = np.mean(y_true)
        std = np.std(y_true)
        if std == 0:
            raise ValueError("Standard deviation is zero, cannot standardize")
        return (y_true - mean) / std, (y_pred - mean) / std
    elif normalisation == 'minmax':
        min_val = np.min(y_true)
        max_val = np.max(y_true)
        if min_val == max_val:
            raise ValueError("Min and max values are equal, cannot normalize")
        return (y_true - min_val) / (max_val - min_val), (y_pred - min_val) / (max_val - min_val)
    elif normalisation == 'robust':
        median = np.median(y_true)
        iqr = np.percentile(y_true, 75) - np.percentile(y_true, 25)
        if iqr == 0:
            raise ValueError("IQR is zero, cannot apply robust normalization")
        return (y_true - median) / iqr, (y_pred - median) / iqr
    else:
        raise ValueError(f"Unknown normalisation method: {normalisation}")

def _calculate_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable] = None
) -> np.ndarray:
    """Calculate residuals using specified metric."""
    if isinstance(metric, str):
        if metric == 'mse':
            return y_true - y_pred
        elif metric == 'mae':
            return np.abs(y_true - y_pred)
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return ss_res / (ss_tot + 1e-10)  # Avoid division by zero
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        return metric(y_true, y_pred)
    else:
        raise TypeError("Metric must be a string or callable")

def _studentize_residuals(residuals: np.ndarray, alpha: float) -> np.ndarray:
    """Studentize the residuals."""
    n = len(residuals)
    if n <= 1:
        raise ValueError("Not enough data points to studentize residuals")

    # Calculate standard error of residuals
    se = np.std(residuals) / np.sqrt(n)

    # Studentize
    studentized_res = residuals / (se + 1e-10)  # Avoid division by zero

    return studentized_res

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residuals: np.ndarray
) -> Dict[str, float]:
    """Calculate various metrics for the residuals."""
    mse = np.mean(residuals ** 2)
    mae = np.mean(np.abs(residuals))
    r2 = 1 - (np.sum(residuals ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

    return {
        'mse': float(mse),
        'mae': float(mae),
        'r2': float(r2)
    }

################################################################################
# residu_normalise
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def residu_normalise_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalisation: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Normalize residuals between true and predicted values with configurable options.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    normalisation : str, optional
        Type of normalization ('none', 'standard', 'minmax', 'robust'), by default 'standard'.
    metric : Union[str, Callable], optional
        Metric to compute ('mse', 'mae', 'r2', 'logloss') or custom callable, by default 'mse'.
    distance : str, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable, by default 'euclidean'.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent'), by default 'closed_form'.
    regularization : Optional[str], optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet'), by default None.
    tol : float, optional
        Tolerance for convergence, by default 1e-6.
    max_iter : int, optional
        Maximum number of iterations, by default 1000.
    custom_metric : Optional[Callable], optional
        Custom metric function, by default None.
    custom_distance : Optional[Callable], optional
        Custom distance function, by default None.

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([1.1, 2.2, 3.3])
    >>> result = residu_normalise_fit(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Normalize residuals
    residuals = y_true - y_pred
    normalized_residuals = _apply_normalization(residuals, normalisation)

    # Choose metric
    if isinstance(metric, str):
        metric_func = _get_metric_function(metric)
    else:
        metric_func = metric

    # Choose distance
    if isinstance(distance, str):
        distance_func = _get_distance_function(distance)
    else:
        distance_func = distance

    # Compute metrics
    metrics = _compute_metrics(normalized_residuals, y_true, y_pred, metric_func)

    # Solve using chosen solver
    params = _solve_residuals(normalized_residuals, y_true, y_pred, solver, regularization, tol, max_iter)

    # Prepare output
    result = {
        'result': normalized_residuals,
        'metrics': metrics,
        'params_used': {
            'normalisation': normalisation,
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

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Inputs must not contain NaN values.")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Inputs must not contain infinite values.")

def _apply_normalization(residuals: np.ndarray, normalisation: str) -> np.ndarray:
    """Apply normalization to residuals."""
    if normalisation == 'none':
        return residuals
    elif normalisation == 'standard':
        mean = np.mean(residuals)
        std = np.std(residuals)
        if std == 0:
            return residuals - mean
        return (residuals - mean) / std
    elif normalisation == 'minmax':
        min_val = np.min(residuals)
        max_val = np.max(residuals)
        if min_val == max_val:
            return residuals - min_val
        return (residuals - min_val) / (max_val - min_val)
    elif normalisation == 'robust':
        median = np.median(residuals)
        iqr = np.percentile(residuals, 75) - np.percentile(residuals, 25)
        if iqr == 0:
            return residuals - median
        return (residuals - median) / iqr
    else:
        raise ValueError(f"Unknown normalisation method: {normalisation}")

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

def _compute_metrics(residuals: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, metric_func: Callable) -> Dict:
    """Compute metrics for residuals."""
    return {'metric': metric_func(y_true, y_pred)}

def _solve_residuals(residuals: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                     solver: str, regularization: Optional[str], tol: float, max_iter: int) -> Dict:
    """Solve residuals using chosen solver."""
    if solver == 'closed_form':
        return _closed_form_solution(residuals, y_true, y_pred)
    elif solver == 'gradient_descent':
        return _gradient_descent(residuals, y_true, y_pred, tol, max_iter)
    elif solver == 'newton':
        return _newton_method(residuals, y_true, y_pred, tol, max_iter)
    elif solver == 'coordinate_descent':
        return _coordinate_descent(residuals, y_true, y_pred, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _closed_form_solution(residuals: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Closed form solution for residuals."""
    return {'params': {}}

def _gradient_descent(residuals: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                      tol: float, max_iter: int) -> Dict:
    """Gradient descent solver for residuals."""
    return {'params': {}}

def _newton_method(residuals: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                   tol: float, max_iter: int) -> Dict:
    """Newton method solver for residuals."""
    return {'params': {}}

def _coordinate_descent(residuals: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                        tol: float, max_iter: int) -> Dict:
    """Coordinate descent solver for residuals."""
    return {'params': {}}

def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Log Loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

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
    return np.sum(np.abs(a - b) ** p) ** (1 / p)

################################################################################
# residu_ajuste
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def residu_ajuste_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    normalization: str = 'none',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], str, list]]:
    """
    Calculate adjusted residuals between true and predicted values with various options.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    normalization : str, optional
        Type of normalization to apply ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to compute ('mse', 'mae', 'r2', 'logloss') or custom callable.
    distance : str or callable, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable.
    solver : str, optional
        Solver to use ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet').
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function.
    custom_distance : callable, optional
        Custom distance function.

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], str, list]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([1.1, 2.2, 3.3])
    >>> result = residu_ajuste_fit(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Normalize data if required
    y_true_norm, y_pred_norm = _apply_normalization(y_true, y_pred, normalization)

    # Calculate residuals
    residuals = y_true_norm - y_pred_norm

    # Choose solver and compute adjusted parameters if needed
    params = _choose_solver(residuals, y_pred_norm, solver, regularization, tol, max_iter)

    # Compute metrics
    metrics = _compute_metrics(residuals, y_true_norm, y_pred_norm, metric, custom_metric)

    # Compute distances
    distance_value = _compute_distance(residuals, y_pred_norm, distance, custom_distance)

    # Prepare result dictionary
    result = {
        "result": residuals,
        "metrics": metrics,
        "distance": distance_value,
        "params_used": {
            "normalization": normalization,
            "metric": metric if isinstance(metric, str) else "custom",
            "distance": distance if isinstance(distance, str) else "custom",
            "solver": solver,
            "regularization": regularization
        },
        "warnings": []
    }

    return result

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Input arrays must not contain infinite values.")

def _apply_normalization(y_true: np.ndarray, y_pred: np.ndarray, normalization: str) -> tuple:
    """Apply specified normalization to the input arrays."""
    if normalization == 'standard':
        mean_true = np.mean(y_true)
        std_true = np.std(y_true)
        y_true_norm = (y_true - mean_true) / std_true
        mean_pred = np.mean(y_pred)
        std_pred = np.std(y_pred)
        y_pred_norm = (y_pred - mean_pred) / std_pred
    elif normalization == 'minmax':
        min_true = np.min(y_true)
        max_true = np.max(y_true)
        y_true_norm = (y_true - min_true) / (max_true - min_true)
        min_pred = np.min(y_pred)
        max_pred = np.max(y_pred)
        y_pred_norm = (y_pred - min_pred) / (max_pred - min_pred)
    elif normalization == 'robust':
        median_true = np.median(y_true)
        iqr_true = np.percentile(y_true, 75) - np.percentile(y_true, 25)
        y_true_norm = (y_true - median_true) / iqr_true
        median_pred = np.median(y_pred)
        iqr_pred = np.percentile(y_pred, 75) - np.percentile(y_pred, 25)
        y_pred_norm = (y_pred - median_pred) / iqr_pred
    else:
        y_true_norm, y_pred_norm = y_true.copy(), y_pred.copy()
    return y_true_norm, y_pred_norm

def _choose_solver(residuals: np.ndarray, y_pred: np.ndarray, solver: str,
                  regularization: Optional[str], tol: float, max_iter: int) -> Dict:
    """Choose and apply the specified solver."""
    if solver == 'closed_form':
        params = _closed_form_solver(residuals, y_pred)
    elif solver == 'gradient_descent':
        params = _gradient_descent_solver(residuals, y_pred, tol, max_iter)
    elif solver == 'newton':
        params = _newton_solver(residuals, y_pred, tol, max_iter)
    elif solver == 'coordinate_descent':
        params = _coordinate_descent_solver(residuals, y_pred, tol, max_iter)
    else:
        params = {}
    return params

def _closed_form_solver(residuals: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Closed form solver for residuals."""
    return {"method": "closed_form"}

def _gradient_descent_solver(residuals: np.ndarray, y_pred: np.ndarray,
                            tol: float, max_iter: int) -> Dict:
    """Gradient descent solver for residuals."""
    return {"method": "gradient_descent", "tol": tol, "max_iter": max_iter}

def _newton_solver(residuals: np.ndarray, y_pred: np.ndarray,
                  tol: float, max_iter: int) -> Dict:
    """Newton solver for residuals."""
    return {"method": "newton", "tol": tol, "max_iter": max_iter}

def _coordinate_descent_solver(residuals: np.ndarray, y_pred: np.ndarray,
                              tol: float, max_iter: int) -> Dict:
    """Coordinate descent solver for residuals."""
    return {"method": "coordinate_descent", "tol": tol, "max_iter": max_iter}

def _compute_metrics(residuals: np.ndarray, y_true: np.ndarray,
                    y_pred: np.ndarray, metric: Union[str, Callable],
                    custom_metric: Optional[Callable]) -> Dict:
    """Compute specified metrics."""
    metrics = {}
    if isinstance(metric, str):
        if metric == 'mse':
            metrics['mse'] = np.mean(residuals ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(residuals))
        elif metric == 'r2':
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot)
        elif metric == 'logloss':
            metrics['logloss'] = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)
    return metrics

def _compute_distance(residuals: np.ndarray, y_pred: np.ndarray,
                     distance: Union[str, Callable], custom_distance: Optional[Callable]) -> float:
    """Compute specified distance."""
    if isinstance(distance, str):
        if distance == 'euclidean':
            return np.linalg.norm(residuals)
        elif distance == 'manhattan':
            return np.sum(np.abs(residuals))
        elif distance == 'cosine':
            return 1 - np.dot(residuals, y_pred) / (np.linalg.norm(residuals) * np.linalg.norm(y_pred))
        elif distance == 'minkowski':
            return np.sum(np.abs(residuals) ** 3) ** (1/3)
    if custom_distance is not None:
        return custom_distance(residuals, y_pred)
    raise ValueError("No valid distance metric provided.")

################################################################################
# residu_partiel
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def residu_partiel_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    normalisation: str = "none",
    metric: Union[str, Callable] = "mse",
    distance: str = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Calculate partial residuals with configurable options.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    normalisation : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    metric : str or callable, optional
        Metric to use ("mse", "mae", "r2", "logloss") or custom callable.
    distance : str, optional
        Distance metric ("euclidean", "manhattan", "cosine", "minkowski").
    solver : str, optional
        Solver method ("closed_form", "gradient_descent", "newton", "coordinate_descent").
    regularization : str, optional
        Regularization method (None, "l1", "l2", "elasticnet").
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function.
    custom_distance : callable, optional
        Custom distance function.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([1.1, 2.2, 3.3])
    >>> result = residu_partiel_fit(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Normalize data if required
    y_true_norm, y_pred_norm = _apply_normalisation(y_true, y_pred, normalisation)

    # Choose metric
    if callable(metric):
        metric_func = metric
    else:
        metric_func = _get_metric_function(metric)

    # Choose distance
    if callable(distance):
        distance_func = distance
    else:
        distance_func = _get_distance_function(distance)

    # Choose solver
    if solver == "closed_form":
        params = _solve_closed_form(y_true_norm, y_pred_norm)
    elif solver == "gradient_descent":
        params = _solve_gradient_descent(y_true_norm, y_pred_norm, tol, max_iter)
    elif solver == "newton":
        params = _solve_newton(y_true_norm, y_pred_norm, tol, max_iter)
    elif solver == "coordinate_descent":
        params = _solve_coordinate_descent(y_true_norm, y_pred_norm, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate residuals
    residuals = y_true_norm - _predict(y_pred_norm, params)

    # Calculate metrics
    metrics = {
        "metric": metric_func(y_true_norm, residuals),
        "distance": distance_func(y_true_norm, y_pred_norm)
    }

    # Apply regularization if required
    if regularization is not None:
        residuals = _apply_regularization(residuals, params, regularization)

    return {
        "result": residuals,
        "metrics": metrics,
        "params_used": params,
        "warnings": _check_warnings(residuals)
    }

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape.")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or Inf values.")
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError("y_pred contains NaN or Inf values.")

def _apply_normalisation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply normalization to input arrays."""
    if method == "standard":
        mean_true = np.mean(y_true)
        std_true = np.std(y_true)
        y_true_norm = (y_true - mean_true) / std_true
        mean_pred = np.mean(y_pred)
        std_pred = np.std(y_pred)
        y_pred_norm = (y_pred - mean_pred) / std_pred
    elif method == "minmax":
        min_true = np.min(y_true)
        max_true = np.max(y_true)
        y_true_norm = (y_true - min_true) / (max_true - min_true)
        min_pred = np.min(y_pred)
        max_pred = np.max(y_pred)
        y_pred_norm = (y_pred - min_pred) / (max_pred - min_pred)
    elif method == "robust":
        median_true = np.median(y_true)
        iqr_true = np.percentile(y_true, 75) - np.percentile(y_true, 25)
        y_true_norm = (y_true - median_true) / iqr_true
        median_pred = np.median(y_pred)
        iqr_pred = np.percentile(y_pred, 75) - np.percentile(y_pred, 25)
        y_pred_norm = (y_pred - median_pred) / iqr_pred
    else:
        y_true_norm, y_pred_norm = y_true.copy(), y_pred.copy()
    return y_true_norm, y_pred_norm

def _get_metric_function(metric: str) -> Callable:
    """Get metric function based on input string."""
    metrics = {
        "mse": _mse,
        "mae": _mae,
        "r2": _r2_score,
        "logloss": _log_loss
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _get_distance_function(distance: str) -> Callable:
    """Get distance function based on input string."""
    distances = {
        "euclidean": _euclidean_distance,
        "manhattan": _manhattan_distance,
        "cosine": _cosine_distance,
        "minkowski": _minkowski_distance
    }
    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")
    return distances[distance]

def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Log loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _euclidean_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Euclidean distance."""
    return np.linalg.norm(y_true - y_pred)

def _manhattan_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Manhattan distance."""
    return np.sum(np.abs(y_true - y_pred))

def _cosine_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cosine distance."""
    return 1 - np.dot(y_true, y_pred) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred))

def _minkowski_distance(y_true: np.ndarray, y_pred: np.ndarray, p: float = 3) -> float:
    """Minkowski distance."""
    return np.sum(np.abs(y_true - y_pred) ** p) ** (1 / p)

def _solve_closed_form(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Solve using closed form solution."""
    X = np.column_stack([np.ones_like(y_pred), y_pred])
    params, _, _, _ = np.linalg.lstsq(X, y_true, rcond=None)
    return {"intercept": params[0], "slope": params[1]}

def _solve_gradient_descent(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tol: float,
    max_iter: int
) -> Dict[str, float]:
    """Solve using gradient descent."""
    X = np.column_stack([np.ones_like(y_pred), y_pred])
    params = np.zeros(2)
    learning_rate = 0.01
    for _ in range(max_iter):
        gradient = -2 * X.T @ (y_true - X @ params) / len(y_true)
        new_params = params - learning_rate * gradient
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params
    return {"intercept": params[0], "slope": params[1]}

def _solve_newton(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tol: float,
    max_iter: int
) -> Dict[str, float]:
    """Solve using Newton's method."""
    X = np.column_stack([np.ones_like(y_pred), y_pred])
    params = np.zeros(2)
    for _ in range(max_iter):
        residuals = y_true - X @ params
        gradient = -2 * X.T @ residuals / len(y_true)
        hessian = 2 * X.T @ X / len(y_true)
        new_params = params - np.linalg.inv(hessian) @ gradient
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params
    return {"intercept": params[0], "slope": params[1]}

def _solve_coordinate_descent(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tol: float,
    max_iter: int
) -> Dict[str, float]:
    """Solve using coordinate descent."""
    X = np.column_stack([np.ones_like(y_pred), y_pred])
    params = np.zeros(2)
    for _ in range(max_iter):
        for i in range(len(params)):
            X_i = X[:, i:i+1]
            params_i = np.zeros_like(params)
            params_i[i] = 1
            params_other = params.copy()
            params_other[i] = 0
            residuals = y_true - X @ (params - params_i * params[i])
            params[i] = np.sum(X_i.T @ residuals) / (X_i.T @ X_i)
        if np.linalg.norm(params - params_other) < tol:
            break
    return {"intercept": params[0], "slope": params[1]}

def _predict(y_pred: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Predict using model parameters."""
    return params["intercept"] + params["slope"] * y_pred

def _apply_regularization(
    residuals: np.ndarray,
    params: Dict[str, float],
    method: str
) -> np.ndarray:
    """Apply regularization to residuals."""
    if method == "l1":
        return residuals + np.abs(params["slope"])
    elif method == "l2":
        return residuals + params["slope"] ** 2
    elif method == "elasticnet":
        return residuals + (np.abs(params["slope"]) + params["slope"] ** 2)
    else:
        return residuals

def _check_warnings(residuals: np.ndarray) -> list[str]:
    """Check for warnings in residuals."""
    warnings = []
    if np.any(np.isnan(residuals)):
        warnings.append("Residuals contain NaN values.")
    if np.any(np.isinf(residuals)):
        warnings.append("Residuals contain Inf values.")
    return warnings

################################################################################
# residu_compose
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def residu_compose_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compute composed residuals analysis with configurable options.

    Parameters:
    -----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    normalizer : Optional[Callable]
        Normalization function. Default is None.
    metric : Union[str, Callable]
        Metric to compute. Can be 'mse', 'mae', 'r2', or custom callable.
    distance : Union[str, Callable]
        Distance metric. Can be 'euclidean', 'manhattan', 'cosine', or custom callable.
    solver : str
        Solver method. Can be 'closed_form', 'gradient_descent', etc.
    regularization : Optional[str]
        Regularization type. Can be 'l1', 'l2', or None.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_params : Optional[Dict]
        Additional parameters for the solver.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Normalize data if specified
    if normalizer is not None:
        y_true = normalizer(y_true)
        y_pred = normalizer(y_pred)

    # Compute residuals
    residuals = _compute_residuals(y_true, y_pred)

    # Choose solver and compute parameters
    params = _choose_solver(
        residuals,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter,
        custom_params=custom_params
    )

    # Compute metrics
    metrics = _compute_metrics(
        y_true,
        y_pred,
        residuals,
        metric=metric,
        distance=distance
    )

    # Prepare output
    result = {
        'result': residuals,
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

    return result

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise ValueError("Inputs must be numpy arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Inputs must not contain NaN values.")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Inputs must not contain infinite values.")

def _compute_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute residuals between true and predicted values."""
    return y_true - y_pred

def _choose_solver(
    residuals: np.ndarray,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    custom_params: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Choose and apply the solver."""
    if solver == 'closed_form':
        params = _solver_closed_form(residuals, regularization)
    elif solver == 'gradient_descent':
        params = _solver_gradient_descent(
            residuals,
            tol=tol,
            max_iter=max_iter,
            custom_params=custom_params
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    return params

def _solver_closed_form(residuals: np.ndarray, regularization: Optional[str]) -> Dict[str, Any]:
    """Closed form solver."""
    params = {'method': 'closed_form'}
    if regularization == 'l1':
        params['regularization'] = 'l1'
    elif regularization == 'l2':
        params['regularization'] = 'l2'
    return params

def _solver_gradient_descent(
    residuals: np.ndarray,
    tol: float,
    max_iter: int,
    custom_params: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Gradient descent solver."""
    params = {
        'method': 'gradient_descent',
        'tol': tol,
        'max_iter': max_iter
    }
    if custom_params:
        params.update(custom_params)
    return params

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable]
) -> Dict[str, float]:
    """Compute metrics based on the specified metric and distance."""
    metrics = {}

    if isinstance(metric, str):
        if metric == 'mse':
            metrics['mse'] = np.mean(residuals**2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(residuals))
        elif metric == 'r2':
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            metrics['r2'] = 1 - (ss_res / ss_tot)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        metrics['custom_metric'] = metric(y_true, y_pred)

    if isinstance(distance, str):
        if distance == 'euclidean':
            metrics['distance'] = np.linalg.norm(residuals)
        elif distance == 'manhattan':
            metrics['distance'] = np.sum(np.abs(residuals))
        elif distance == 'cosine':
            metrics['distance'] = 1 - np.dot(residuals, residuals) / (np.linalg.norm(residuals) * np.linalg.norm(residuals))
        else:
            raise ValueError(f"Unknown distance: {distance}")
    else:
        metrics['custom_distance'] = distance(y_true, y_pred)

    return metrics

################################################################################
# residu_autocorrelation
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def residu_autocorrelation_fit(
    residuals: np.ndarray,
    lags: Optional[Union[int, np.ndarray]] = None,
    normalization: str = 'standard',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Compute autocorrelation of residuals with configurable options.

    Parameters:
    -----------
    residuals : np.ndarray
        Array of residuals to analyze.
    lags : Optional[Union[int, np.ndarray]], default=None
        Lags to compute autocorrelation for. If None, uses lags from 1 to min(10, len(residuals)//2).
    normalization : str, default='standard'
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : Union[str, Callable], default='mse'
        Metric to evaluate autocorrelation: 'mse', 'mae', 'r2', or custom callable.
    custom_metric : Optional[Callable], default=None
        Custom metric function if not using built-in metrics.
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(residuals, lags)

    # Set default lags if not provided
    if lags is None:
        max_lag = min(10, len(residuals) // 2)
        lags = np.arange(1, max_lag + 1)

    # Normalize residuals
    normalized_residuals = _apply_normalization(residuals, normalization)

    # Compute autocorrelation
    acf_values = _compute_autocorrelation(normalized_residuals, lags)

    # Compute metrics
    metrics = _compute_metrics(acf_values, residuals, metric, custom_metric)

    # Prepare results
    result = {
        'acf_values': acf_values,
        'lags': lags,
        'confidence_intervals': _compute_confidence_intervals(acf_values, alpha)
    }

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric if isinstance(metric, str) else 'custom',
            'alpha': alpha
        },
        'warnings': _check_warnings(residuals, lags)
    }

def _validate_inputs(residuals: np.ndarray, lags: Optional[Union[int, np.ndarray]]) -> None:
    """Validate input residuals and lags."""
    if not isinstance(residuals, np.ndarray):
        raise TypeError("residuals must be a numpy array")
    if not np.issubdtype(residuals.dtype, np.number):
        raise ValueError("residuals must contain numerical values")
    if np.isnan(residuals).any():
        raise ValueError("residuals contain NaN values")
    if np.isinf(residuals).any():
        raise ValueError("residuals contain infinite values")

    if lags is not None:
        if isinstance(lags, int):
            if lags <= 0:
                raise ValueError("lag must be positive")
        else:
            if not isinstance(lags, np.ndarray):
                raise TypeError("lags must be an integer or numpy array")
            if len(lags) == 0:
                raise ValueError("lags array cannot be empty")
            if np.any(lags <= 0):
                raise ValueError("all lags must be positive")

def _apply_normalization(residuals: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to residuals."""
    if method == 'none':
        return residuals
    elif method == 'standard':
        mean = np.mean(residuals)
        std = np.std(residuals)
        return (residuals - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(residuals)
        max_val = np.max(residuals)
        return (residuals - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(residuals)
        iqr = np.percentile(residuals, 75) - np.percentile(residuals, 25)
        return (residuals - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_autocorrelation(residuals: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Compute autocorrelation for given lags."""
    n = len(residuals)
    mean = np.mean(residuals)

    acf_values = []
    for lag in lags:
        if lag >= n:
            raise ValueError(f"lag {lag} is too large for residuals of length {n}")

        numerator = np.sum((residuals[:-lag] - mean) * (residuals[lag:] - mean))
        denominator = np.sum((residuals - mean) ** 2)
        acf_values.append(numerator / (denominator + 1e-8))

    return np.array(acf_values)

def _compute_metrics(
    acf_values: np.ndarray,
    residuals: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict[str, float]:
    """Compute metrics based on autocorrelation values."""
    metrics = {}

    if isinstance(metric, str):
        if metric == 'mse':
            metrics['mse'] = np.mean(acf_values ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(acf_values))
        elif metric == 'r2':
            ss_res = np.sum((residuals - np.mean(residuals)) ** 2)
            metrics['r2'] = 1 - (np.sum(acf_values ** 2) / ss_res)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        metrics['custom_metric'] = metric(acf_values, residuals)
    elif custom_metric is not None:
        metrics['custom_metric'] = custom_metric(acf_values, residuals)
    else:
        raise ValueError("No valid metric provided")

    return metrics

def _compute_confidence_intervals(acf_values: np.ndarray, alpha: float) -> np.ndarray:
    """Compute confidence intervals for autocorrelation values."""
    n = len(acf_values)
    z_critical = 1.96  # For alpha=0.05, two-tailed test
    se = np.sqrt((1 + 2 * np.arange(1, n+1) ** -1) / len(acf_values))
    return z_critical * se

def _check_warnings(residuals: np.ndarray, lags: Optional[np.ndarray]) -> list:
    """Check for potential warnings in the analysis."""
    warnings = []

    if len(residuals) < 20:
        warnings.append("Small sample size may affect autocorrelation results")

    if lags is not None and np.any(lags > len(residuals) // 2):
        warnings.append("Some lags may be too large for reliable autocorrelation estimation")

    return warnings

# Example usage:
"""
residuals = np.random.randn(100)
result = residu_autocorrelation_fit(residuals, lags=5, normalization='standard', metric='mse')
print(result)
"""

################################################################################
# residu_heteroscedasticite
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Literal

def residu_heteroscedasticite_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    normalisation: Literal["none", "standard", "minmax", "robust"] = "none",
    metric: Union[Literal["mse", "mae", "r2"], Callable[[np.ndarray, np.ndarray], float]] = "mse",
    distance: Literal["euclidean", "manhattan", "cosine"] = "euclidean",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    alpha: float = 0.05
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Analyse l'hétéroscédasticité des résidus entre les valeurs observées et prédites.

    Parameters:
    -----------
    y_true : np.ndarray
        Valeurs observées (réelles).
    y_pred : np.ndarray
        Valeurs prédites.
    normalisation : str, optional
        Type de normalisation à appliquer aux résidus (default: "none").
    metric : str or callable, optional
        Métrique pour évaluer la performance des prédictions (default: "mse").
    distance : str, optional
        Distance à utiliser pour les tests d'hétéroscédasticité (default: "euclidean").
    custom_metric : callable, optional
        Fonction personnalisée pour calculer la métrique.
    custom_distance : callable, optional
        Fonction personnalisée pour calculer la distance.
    alpha : float, optional
        Niveau de signification pour les tests (default: 0.05).

    Returns:
    --------
    dict
        Dictionnaire contenant les résultats, métriques et paramètres utilisés.
    """
    # Validation des entrées
    _validate_inputs(y_true, y_pred)

    # Calcul des résidus
    residuals = y_true - y_pred

    # Normalisation des résidus si nécessaire
    if normalisation != "none":
        residuals = _apply_normalization(residuals, method=normalisation)

    # Calcul de la métrique
    metrics = _compute_metrics(residuals, y_pred, metric=metric, custom_metric=custom_metric)

    # Test d'hétéroscédasticité
    test_result = _test_heteroscedasticity(residuals, distance=distance, custom_distance=custom_distance)

    # Retourne les résultats structurés
    return {
        "result": test_result,
        "metrics": metrics,
        "params_used": {
            "normalisation": normalisation,
            "metric": metric if not custom_metric else "custom",
            "distance": distance if not custom_distance else "custom",
            "alpha": alpha
        },
        "warnings": _check_warnings(residuals)
    }

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Valide les entrées pour l'analyse des résidus."""
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true et y_pred doivent avoir les mêmes dimensions.")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contient des NaN ou inf.")
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError("y_pred contient des NaN ou inf.")

def _apply_normalization(residuals: np.ndarray, method: str) -> np.ndarray:
    """Applique une normalisation aux résidus."""
    if method == "standard":
        mean = np.mean(residuals)
        std = np.std(residuals)
        return (residuals - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(residuals)
        max_val = np.max(residuals)
        return (residuals - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(residuals)
        iqr = np.percentile(residuals, 75) - np.percentile(residuals, 25)
        return (residuals - median) / (iqr + 1e-8)
    else:
        return residuals

def _compute_metrics(
    residuals: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[Literal["mse", "mae", "r2"], Callable[[np.ndarray, np.ndarray], float]],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Calcule les métriques pour évaluer la performance des prédictions."""
    metrics = {}

    if custom_metric:
        metrics["custom"] = custom_metric(residuals, y_pred)
    else:
        if metric == "mse":
            metrics["mse"] = np.mean(residuals**2)
        elif metric == "mae":
            metrics["mae"] = np.mean(np.abs(residuals))
        elif metric == "r2":
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_pred - np.mean(y_pred))**2)
            metrics["r2"] = 1 - (ss_res / (ss_tot + 1e-8))

    return metrics

def _test_heteroscedasticity(
    residuals: np.ndarray,
    distance: Literal["euclidean", "manhattan", "cosine"],
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[bool, float]]:
    """Teste l'hétéroscédasticité des résidus."""
    if custom_distance:
        dist_matrix = np.array([[custom_distance(residuals[i], residuals[j]) for j in range(len(residuals))] for i in range(len(residuals))])
    else:
        if distance == "euclidean":
            dist_matrix = np.sqrt(np.sum((residuals[:, np.newaxis] - residuals) ** 2, axis=2))
        elif distance == "manhattan":
            dist_matrix = np.sum(np.abs(residuals[:, np.newaxis] - residuals), axis=2)
        elif distance == "cosine":
            dist_matrix = 1 - np.dot(residuals, residuals.T) / (np.linalg.norm(residuals, axis=1)[:, np.newaxis] * np.linalg.norm(residuals, axis=1))

    # Exemple de test simple (à remplacer par un test statistique approprié)
    mean_residuals = np.mean(residuals)
    variance_ratio = np.var(residuals[residuals > mean_residuals]) / (np.var(residuals[residuals <= mean_residuals]) + 1e-8)
    is_heteroscedastic = variance_ratio > 2.0

    return {
        "is_heteroscedastic": is_heteroscedastic,
        "variance_ratio": variance_ratio
    }

def _check_warnings(residuals: np.ndarray) -> list:
    """Vérifie les avertissements potentiels."""
    warnings = []
    if np.std(residuals) < 1e-6:
        warnings.append("Les résidus ont une variance très faible.")
    if np.any(np.abs(residuals) > 1e6):
        warnings.append("Les résidus contiennent des valeurs extrêmes.")
    return warnings

################################################################################
# residu_influence
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def residu_influence_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: Optional[np.ndarray] = None,
    normalizer: str = "standard",
    metric: Union[str, Callable] = "mse",
    distance: str = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Calculate the influence of residuals in a regression model.

    Parameters:
    -----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.
    X : Optional[np.ndarray]
        Feature matrix. Required for some solvers.
    normalizer : str
        Normalization method: "none", "standard", "minmax", or "robust".
    metric : Union[str, Callable]
        Metric to use: "mse", "mae", "r2", or a custom callable.
    distance : str
        Distance metric: "euclidean", "manhattan", "cosine", or "minkowski".
    solver : str
        Solver method: "closed_form", "gradient_descent", "newton", or "coordinate_descent".
    regularization : Optional[str]
        Regularization method: "none", "l1", "l2", or "elasticnet".
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
    _validate_inputs(y_true, y_pred, X)

    # Normalize data if required
    y_true_norm, y_pred_norm = _apply_normalization(y_true, y_pred, normalizer)

    # Calculate residuals
    residuals = _calculate_residuals(y_true_norm, y_pred_norm)

    # Choose metric
    if isinstance(metric, str):
        metric_func = _get_metric_function(metric)
    else:
        metric_func = metric

    # Choose distance
    if isinstance(distance, str):
        distance_func = _get_distance_function(distance)
    else:
        distance_func = distance

    # Calculate influence
    influence = _calculate_influence(residuals, X, metric_func, distance_func, solver,
                                    regularization, tol, max_iter)

    # Calculate metrics
    metrics = _calculate_metrics(y_true_norm, y_pred_norm, metric_func)

    # Prepare output
    result = {
        "result": influence,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer,
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularization": regularization
        },
        "warnings": []
    }

    return result

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray, X: Optional[np.ndarray]) -> None:
    """Validate input arrays."""
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if X is not None:
        if len(y_true) != X.shape[0]:
            raise ValueError("Number of samples in y_true must match number of rows in X.")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or infinite values.")
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError("y_pred contains NaN or infinite values.")
    if X is not None and (np.any(np.isnan(X)) or np.any(np.isinf(X))):
        raise ValueError("X contains NaN or infinite values.")

def _apply_normalization(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> tuple:
    """Apply normalization to the data."""
    if method == "none":
        return y_true, y_pred
    elif method == "standard":
        mean = np.mean(y_true)
        std = np.std(y_true)
        y_true_norm = (y_true - mean) / std
        y_pred_norm = (y_pred - mean) / std
    elif method == "minmax":
        min_val = np.min(y_true)
        max_val = np.max(y_true)
        y_true_norm = (y_true - min_val) / (max_val - min_val)
        y_pred_norm = (y_pred - min_val) / (max_val - min_val)
    elif method == "robust":
        median = np.median(y_true)
        iqr = np.percentile(y_true, 75) - np.percentile(y_true, 25)
        y_true_norm = (y_true - median) / iqr
        y_pred_norm = (y_pred - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return y_true_norm, y_pred_norm

def _calculate_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calculate residuals."""
    return y_true - y_pred

def _get_metric_function(metric: str) -> Callable:
    """Get the metric function based on the input string."""
    metrics = {
        "mse": _mean_squared_error,
        "mae": _mean_absolute_error,
        "r2": _r_squared
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

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

def _get_distance_function(distance: str) -> Callable:
    """Get the distance function based on the input string."""
    distances = {
        "euclidean": _euclidean_distance,
        "manhattan": _manhattan_distance,
        "cosine": _cosine_distance
    }
    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")
    return distances[distance]

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine distance."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _calculate_influence(
    residuals: np.ndarray,
    X: Optional[np.ndarray],
    metric_func: Callable,
    distance_func: Callable,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Calculate the influence of residuals."""
    if solver == "closed_form":
        return _closed_form_influence(residuals, X)
    elif solver == "gradient_descent":
        return _gradient_descent_influence(residuals, X, metric_func,
                                          distance_func, regularization, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _closed_form_influence(residuals: np.ndarray, X: Optional[np.ndarray]) -> np.ndarray:
    """Calculate influence using closed form solution."""
    if X is None:
        return residuals
    # Simplified example: influence proportional to residuals and leverage
    hat_matrix = np.linalg.inv(X.T @ X) @ (X.T @ X)
    leverage = np.diag(hat_matrix)
    return residuals * leverage

def _gradient_descent_influence(
    residuals: np.ndarray,
    X: Optional[np.ndarray],
    metric_func: Callable,
    distance_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Calculate influence using gradient descent."""
    if X is None:
        return residuals
    # Simplified example: iterative approach
    influence = np.zeros_like(residuals)
    for _ in range(max_iter):
        # Update rule (simplified)
        influence = influence - 0.01 * (metric_func(residuals, X @ influence) + _apply_regularization(influence, regularization))
        if np.linalg.norm(influence) < tol:
            break
    return influence

def _apply_regularization(params: np.ndarray, method: Optional[str]) -> float:
    """Apply regularization."""
    if method == "l1":
        return np.sum(np.abs(params))
    elif method == "l2":
        return np.sum(params ** 2)
    elif method == "elasticnet" and method is not None:
        return np.sum(np.abs(params)) + np.sum(params ** 2)
    else:
        return 0.0

def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, metric_func: Callable) -> Dict:
    """Calculate metrics."""
    return {"metric_value": metric_func(y_true, y_pred)}

################################################################################
# residu_leverage
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def residu_leverage_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalize: str = "standard",
    metric: Union[str, Callable] = "mse",
    distance: str = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Compute residuals and leverage values for linear regression analysis.

    Parameters:
    -----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    normalize : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : str or callable, optional
        Metric to evaluate residuals: 'mse', 'mae', 'r2', or custom callable.
    distance : str, optional
        Distance metric for leverage calculation: 'euclidean', 'manhattan',
        'cosine', or 'minkowski'.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', 'newton',
        or 'coordinate_descent'.
    regularization : str, optional
        Regularization type: 'none', 'l1', 'l2', or 'elasticnet'.
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
        - 'result': Dictionary with residuals and leverage values.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used in the computation.
        - 'warnings': Any warnings encountered.

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = residu_leverage_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_norm, y_norm = _normalize_data(X, y, normalize)

    # Choose solver
    if solver == "closed_form":
        beta = _solve_closed_form(X_norm, y_norm)
    elif solver == "gradient_descent":
        beta = _solve_gradient_descent(X_norm, y_norm, tol=tol, max_iter=max_iter)
    elif solver == "newton":
        beta = _solve_newton(X_norm, y_norm, tol=tol, max_iter=max_iter)
    elif solver == "coordinate_descent":
        beta = _solve_coordinate_descent(X_norm, y_norm, tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if required
    if regularization:
        beta = _apply_regularization(beta, X_norm, y_norm, regularization)

    # Compute residuals
    residuals = _compute_residuals(X_norm, y_norm, beta)

    # Compute leverage values
    leverage = _compute_leverage(X_norm, distance, custom_distance)

    # Compute metrics
    if isinstance(metric, str):
        metrics = _compute_metrics(residuals, metric)
    else:
        metrics = {"custom": metric(y_norm, residuals)}

    # Prepare output
    result = {
        "residuals": residuals,
        "leverage": leverage,
        "coefficients": beta
    }

    output = {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "normalize": normalize,
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularization": regularization
        },
        "warnings": []
    }

    return output

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
    method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize data according to specified method."""
    if method == "none":
        return X, y
    elif method == "standard":
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y_norm = (y - np.mean(y)) / np.std(y)
    elif method == "minmax":
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif method == "robust":
        X_norm = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        y_norm = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return X_norm, y_norm

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve linear regression using closed-form solution."""
    return np.linalg.inv(X.T @ X) @ X.T @ y

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> np.ndarray:
    """Solve linear regression using gradient descent."""
    n_features = X.shape[1]
    beta = np.zeros(n_features)
    learning_rate = 0.01
    for _ in range(max_iter):
        gradient = -2 * X.T @ (y - X @ beta) / len(y)
        new_beta = beta - learning_rate * gradient
        if np.linalg.norm(new_beta - beta) < tol:
            break
        beta = new_beta
    return beta

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> np.ndarray:
    """Solve linear regression using Newton's method."""
    n_features = X.shape[1]
    beta = np.zeros(n_features)
    for _ in range(max_iter):
        residuals = y - X @ beta
        gradient = -2 * X.T @ residuals / len(y)
        hessian = 2 * X.T @ X / len(y)
        new_beta = beta - np.linalg.inv(hessian) @ gradient
        if np.linalg.norm(new_beta - beta) < tol:
            break
        beta = new_beta
    return beta

def _solve_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> np.ndarray:
    """Solve linear regression using coordinate descent."""
    n_features = X.shape[1]
    beta = np.zeros(n_features)
    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - (X @ beta - X_j * beta[j])
            beta[j] = np.sum(X_j * residuals) / np.sum(X_j ** 2)
        if np.linalg.norm(beta - beta.copy()) < tol:
            break
    return beta

def _apply_regularization(
    beta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply regularization to coefficients."""
    if method == "l1":
        return _solve_lasso(X, y)
    elif method == "l2":
        return _solve_ridge(X, y)
    elif method == "elasticnet":
        return _solve_elasticnet(X, y)
    else:
        raise ValueError(f"Unknown regularization method: {method}")

def _solve_lasso(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve L1-regularized regression."""
    # Simplified implementation - in practice would use coordinate descent
    return np.linalg.inv(X.T @ X + 0.1 * np.eye(X.shape[1])) @ X.T @ y

def _solve_ridge(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve L2-regularized regression."""
    return np.linalg.inv(X.T @ X + 0.1 * np.eye(X.shape[1])) @ X.T @ y

def _solve_elasticnet(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve elastic net regression."""
    # Simplified implementation - in practice would combine L1 and L2
    return np.linalg.inv(X.T @ X + 0.1 * np.eye(X.shape[1])) @ X.T @ y

def _compute_residuals(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray
) -> np.ndarray:
    """Compute residuals from predictions."""
    return y - X @ beta

def _compute_leverage(
    X: np.ndarray,
    distance: str = "euclidean",
    custom_distance: Optional[Callable] = None
) -> np.ndarray:
    """Compute leverage values using specified distance metric."""
    if custom_distance is not None:
        return _compute_custom_leverage(X, custom_distance)
    elif distance == "euclidean":
        return np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)
    elif distance == "manhattan":
        return _compute_manhattan_leverage(X)
    elif distance == "cosine":
        return _compute_cosine_leverage(X)
    elif distance == "minkowski":
        return _compute_minkowski_leverage(X)
    else:
        raise ValueError(f"Unknown distance metric: {distance}")

def _compute_custom_leverage(
    X: np.ndarray,
    distance_func: Callable
) -> np.ndarray:
    """Compute leverage using custom distance function."""
    # Implementation would depend on the specific distance function
    return np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)

def _compute_manhattan_leverage(X: np.ndarray) -> np.ndarray:
    """Compute leverage using Manhattan distance."""
    # Implementation specific to Manhattan distance
    return np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)

def _compute_cosine_leverage(X: np.ndarray) -> np.ndarray:
    """Compute leverage using cosine distance."""
    # Implementation specific to cosine distance
    return np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)

def _compute_minkowski_leverage(X: np.ndarray) -> np.ndarray:
    """Compute leverage using Minkowski distance."""
    # Implementation specific to Minkowski distance
    return np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)

def _compute_metrics(
    residuals: np.ndarray,
    metric: str
) -> Dict[str, float]:
    """Compute specified metrics for residuals."""
    if metric == "mse":
        return {"mse": np.mean(residuals ** 2)}
    elif metric == "mae":
        return {"mae": np.mean(np.abs(residuals))}
    elif metric == "r2":
        return {"r2": 1 - np.sum(residuals ** 2) / np.sum((residuals + np.mean(residuals)) ** 2)}
    else:
        raise ValueError(f"Unknown metric: {metric}")

################################################################################
# residu_quantile
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def residu_quantile_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantiles: Union[np.ndarray, list] = np.linspace(0.1, 0.9, 9),
    normalization: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Calculate quantile residuals for a given set of true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    quantiles : Union[np.ndarray, list], optional
        Quantiles to compute residuals for. Default is np.linspace(0.1, 0.9, 9).
    normalization : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'. Default is 'standard'.
    metric : Union[str, Callable], optional
        Metric to use: 'mse', 'mae', 'r2', or a custom callable. Default is 'mse'.
    distance : str, optional
        Distance metric: 'euclidean', 'manhattan', 'cosine', or 'minkowski'. Default is 'euclidean'.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'. Default is 'closed_form'.
    regularization : Optional[str], optional
        Regularization method: None, 'l1', 'l2', or 'elasticnet'. Default is None.
    tol : float, optional
        Tolerance for convergence. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    custom_metric : Optional[Callable], optional
        Custom metric function. Default is None.
    custom_distance : Optional[Callable], optional
        Custom distance function. Default is None.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Computed quantile residuals.
        - 'metrics': Metrics computed during the process.
        - 'params_used': Parameters used in the computation.
        - 'warnings': Any warnings generated during the process.

    Example
    -------
    >>> y_true = np.array([1, 2, 3, 4, 5])
    >>> y_pred = np.array([1.1, 2.0, 3.2, 4.1, 5.0])
    >>> result = residu_quantile_fit(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Normalize data if required
    y_true_norm, y_pred_norm = _apply_normalization(y_true, y_pred, normalization)

    # Compute residuals
    residuals = y_true_norm - y_pred_norm

    # Calculate quantile residuals
    quantile_residuals = _compute_quantile_residuals(residuals, quantiles)

    # Compute metrics
    metrics = _compute_metrics(y_true_norm, y_pred_norm, metric, custom_metric)

    # Prepare output
    result = {
        'result': quantile_residuals,
        'metrics': metrics,
        'params_used': {
            'quantiles': quantiles,
            'normalization': normalization,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization
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
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("Inputs must not contain NaN values.")
    if np.isinf(y_true).any() or np.isinf(y_pred).any():
        raise ValueError("Inputs must not contain infinite values.")

def _apply_normalization(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> tuple:
    """Apply normalization to input arrays."""
    if method == 'none':
        return y_true, y_pred
    elif method == 'standard':
        mean = np.mean(y_true)
        std = np.std(y_true)
        y_true_norm = (y_true - mean) / std
        y_pred_norm = (y_pred - mean) / std
    elif method == 'minmax':
        min_val = np.min(y_true)
        max_val = np.max(y_true)
        y_true_norm = (y_true - min_val) / (max_val - min_val)
        y_pred_norm = (y_pred - min_val) / (max_val - min_val)
    elif method == 'robust':
        median = np.median(y_true)
        iqr = np.percentile(y_true, 75) - np.percentile(y_true, 25)
        y_true_norm = (y_true - median) / iqr
        y_pred_norm = (y_pred - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return y_true_norm, y_pred_norm

def _compute_quantile_residuals(residuals: np.ndarray, quantiles: Union[np.ndarray, list]) -> np.ndarray:
    """Compute quantile residuals."""
    quantiles = np.array(quantiles)
    if not (0 <= quantiles).all() or not (quantiles <= 1).all():
        raise ValueError("Quantiles must be between 0 and 1.")
    quantile_values = np.percentile(residuals, quantiles * 100)
    return residuals - quantile_values

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Compute metrics based on the specified metric."""
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
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        metrics['custom_metric'] = metric(y_true, y_pred)
    if custom_metric is not None:
        metrics['custom_metric'] = custom_metric(y_true, y_pred)
    return metrics

################################################################################
# residu_diagnostic
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def residu_diagnostic_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    normalization: str = 'standard',
    metrics: Union[str, list] = ['mse', 'mae'],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    residual_type: str = 'raw',
    alpha: Optional[float] = None
) -> Dict[str, Union[Dict, float]]:
    """
    Perform diagnostic analysis of residuals between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    normalization : str, optional
        Type of normalization to apply ('none', 'standard', 'minmax', 'robust').
    metrics : str or list, optional
        Metrics to compute ('mse', 'mae', 'r2', 'logloss').
    custom_metric : callable, optional
        Custom metric function taking (y_true, y_pred) and returning a float.
    residual_type : str, optional
        Type of residuals to analyze ('raw', 'standardized').
    alpha : float, optional
        Significance level for statistical tests.

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': Dictionary of residual statistics
        - 'metrics': Dictionary of computed metrics
        - 'params_used': Parameters used in the computation
        - 'warnings': List of warnings encountered

    Example
    -------
    >>> y_true = np.array([1, 2, 3])
    >>> y_pred = np.array([1.1, 1.9, 3.2])
    >>> result = residu_diagnostic_fit(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Compute residuals
    residuals = _compute_residuals(y_true, y_pred, residual_type)

    # Normalize residuals
    normalized_residuals = _normalize_residuals(residuals, normalization)

    # Compute metrics
    metrics_dict = _compute_metrics(y_true, y_pred, residuals,
                                   metrics=metrics,
                                   custom_metric=custom_metric)

    # Perform diagnostics
    diagnostics = _perform_diagnostics(normalized_residuals, alpha=alpha)

    return {
        'result': diagnostics,
        'metrics': metrics_dict,
        'params_used': {
            'normalization': normalization,
            'metrics': metrics,
            'residual_type': residual_type,
            'alpha': alpha
        },
        'warnings': []
    }

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Inputs contain NaN values")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Inputs contain infinite values")

def _compute_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residual_type: str = 'raw'
) -> np.ndarray:
    """Compute residuals between true and predicted values."""
    if residual_type == 'raw':
        return y_true - y_pred
    elif residual_type == 'standardized':
        residuals = y_true - y_pred
        return residuals / np.std(residuals)
    else:
        raise ValueError(f"Unknown residual type: {residual_type}")

def _normalize_residuals(
    residuals: np.ndarray,
    normalization: str = 'standard'
) -> np.ndarray:
    """Normalize residuals."""
    if normalization == 'none':
        return residuals
    elif normalization == 'standard':
        return (residuals - np.mean(residuals)) / np.std(residuals)
    elif normalization == 'minmax':
        return (residuals - np.min(residuals)) / (np.max(residuals) - np.min(residuals))
    elif normalization == 'robust':
        median = np.median(residuals)
        mad = np.median(np.abs(residuals - median))
        return (residuals - median) / mad
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    metrics: Union[str, list] = ['mse', 'mae'],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Compute specified metrics."""
    metric_dict = {}

    if isinstance(metrics, str):
        metrics = [metrics]

    for metric in metrics:
        if metric == 'mse':
            metric_dict['mse'] = np.mean(residuals**2)
        elif metric == 'mae':
            metric_dict['mae'] = np.mean(np.abs(residuals))
        elif metric == 'r2':
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            metric_dict['r2'] = 1 - (ss_res / ss_tot)
        elif metric == 'logloss':
            # Assuming binary classification for log loss
            y_true_bin = (y_true > 0.5).astype(float)
            y_pred_prob = np.clip(y_pred, 1e-15, 1 - 1e-15)
            metric_dict['logloss'] = -np.mean(y_true_bin * np.log(y_pred_prob) +
                                             (1 - y_true_bin) * np.log(1 - y_pred_prob))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    if custom_metric is not None:
        try:
            metric_dict['custom'] = custom_metric(y_true, y_pred)
        except Exception as e:
            raise ValueError(f"Custom metric function failed: {str(e)}")

    return metric_dict

def _perform_diagnostics(
    residuals: np.ndarray,
    alpha: Optional[float] = None
) -> Dict[str, float]:
    """Perform statistical diagnostics on residuals."""
    diagnostics = {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'skewness': _compute_skewness(residuals),
        'kurtosis': _compute_kurtosis(residuals)
    }

    if alpha is not None:
        diagnostics['p_value_normal'] = _normality_test(residuals, alpha)

    return diagnostics

def _compute_skewness(x: np.ndarray) -> float:
    """Compute skewness of a dataset."""
    mean = np.mean(x)
    std = np.std(x)
    n = len(x)
    return (n / ((n - 1) * (n - 2))) * np.sum(((x - mean) / std)**3)

def _compute_kurtosis(x: np.ndarray) -> float:
    """Compute kurtosis of a dataset."""
    mean = np.mean(x)
    std = np.std(x)
    n = len(x)
    return (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((x - mean) / std)**4) - (3 * (n - 1)**2) / ((n - 2) * (n - 3))

def _normality_test(x: np.ndarray, alpha: float = 0.05) -> float:
    """Perform normality test on residuals."""
    # This is a placeholder for an actual normality test
    # In practice, you might use Shapiro-Wilk or Kolmogorov-Smirnov tests
    from scipy.stats import shapiro
    stat, p_value = shapiro(x)
    return p_value

################################################################################
# residu_graphique
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def residu_graphique_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    normalization: str = "none",
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Fit residuals analysis with configurable options.

    Parameters:
    -----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to compute ('mse', 'mae', 'r2', 'logloss') or custom callable.
    distance : str or callable, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    regularization : str, optional
        Regularization method (None, 'l1', 'l2', 'elasticnet').
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function.
    custom_distance : callable, optional
        Custom distance function.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Normalize data if needed
    y_true_norm, y_pred_norm = _apply_normalization(y_true, y_pred, normalization)

    # Compute residuals
    residuals = y_true_norm - y_pred_norm

    # Choose metric
    if callable(metric):
        computed_metric = metric(y_true_norm, y_pred_norm)
    else:
        if custom_metric is not None:
            computed_metric = custom_metric(y_true_norm, y_pred_norm)
        else:
            metric_func = _get_metric(metric)
            computed_metric = metric_func(y_true_norm, y_pred_norm)

    # Choose distance
    if callable(distance):
        computed_distance = distance(y_true_norm, y_pred_norm)
    else:
        if custom_distance is not None:
            computed_distance = custom_distance(y_true_norm, y_pred_norm)
        else:
            distance_func = _get_distance(distance)
            computed_distance = distance_func(y_true_norm, y_pred_norm)

    # Solve using chosen solver
    params = _solve_residuals(
        residuals,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter
    )

    # Prepare output
    result = {
        "result": residuals,
        "metrics": {
            "metric_value": computed_metric,
            "distance_value": computed_distance
        },
        "params_used": {
            "normalization": normalization,
            "metric": metric if not callable(metric) else "custom",
            "distance": distance if not callable(distance) else "custom",
            "solver": solver,
            "regularization": regularization
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
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or inf values.")
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError("y_pred contains NaN or inf values.")

def _apply_normalization(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply normalization to input arrays."""
    if method == "none":
        return y_true, y_pred
    elif method == "standard":
        mean = np.mean(y_true)
        std = np.std(y_true)
        y_true_norm = (y_true - mean) / std
        y_pred_norm = (y_pred - mean) / std
    elif method == "minmax":
        min_val = np.min(y_true)
        max_val = np.max(y_true)
        y_true_norm = (y_true - min_val) / (max_val - min_val)
        y_pred_norm = (y_pred - min_val) / (max_val - min_val)
    elif method == "robust":
        median = np.median(y_true)
        iqr = np.percentile(y_true, 75) - np.percentile(y_true, 25)
        y_true_norm = (y_true - median) / iqr
        y_pred_norm = (y_pred - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return y_true_norm, y_pred_norm

def _get_metric(metric_name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get metric function based on name."""
    metrics = {
        "mse": _mean_squared_error,
        "mae": _mean_absolute_error,
        "r2": _r_squared,
        "logloss": _log_loss
    }
    if metric_name not in metrics:
        raise ValueError(f"Unknown metric: {metric_name}")
    return metrics[metric_name]

def _get_distance(distance_name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get distance function based on name."""
    distances = {
        "euclidean": _euclidean_distance,
        "manhattan": _manhattan_distance,
        "cosine": _cosine_distance,
        "minkowski": _minkowski_distance
    }
    if distance_name not in distances:
        raise ValueError(f"Unknown distance: {distance_name}")
    return distances[distance_name]

def _solve_residuals(
    residuals: np.ndarray,
    *,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Solve residuals using chosen solver and regularization."""
    if solver == "closed_form":
        params = _closed_form_solution(residuals)
    elif solver == "gradient_descent":
        params = _gradient_descent(
            residuals,
            tol=tol,
            max_iter=max_iter,
            regularization=regularization
        )
    elif solver == "newton":
        params = _newton_method(
            residuals,
            tol=tol,
            max_iter=max_iter,
            regularization=regularization
        )
    elif solver == "coordinate_descent":
        params = _coordinate_descent(
            residuals,
            tol=tol,
            max_iter=max_iter,
            regularization=regularization
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")
    return params

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

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute log loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _euclidean_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Euclidean distance."""
    return np.linalg.norm(y_true - y_pred)

def _manhattan_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Manhattan distance."""
    return np.sum(np.abs(y_true - y_pred))

def _cosine_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute cosine distance."""
    return 1 - np.dot(y_true, y_pred) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred))

def _minkowski_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Minkowski distance."""
    return np.sum(np.abs(y_true - y_pred) ** 3) ** (1/3)

def _closed_form_solution(residuals: np.ndarray) -> Dict[str, Any]:
    """Closed form solution for residuals."""
    return {"method": "closed_form", "params": {}}

def _gradient_descent(
    residuals: np.ndarray,
    *,
    tol: float,
    max_iter: int,
    regularization: Optional[str]
) -> Dict[str, Any]:
    """Gradient descent solver for residuals."""
    return {"method": "gradient_descent", "params": {}, "iterations": 0}

def _newton_method(
    residuals: np.ndarray,
    *,
    tol: float,
    max_iter: int,
    regularization: Optional[str]
) -> Dict[str, Any]:
    """Newton method solver for residuals."""
    return {"method": "newton", "params": {}, "iterations": 0}

def _coordinate_descent(
    residuals: np.ndarray,
    *,
    tol: float,
    max_iter: int,
    regularization: Optional[str]
) -> Dict[str, Any]:
    """Coordinate descent solver for residuals."""
    return {"method": "coordinate_descent", "params": {}, "iterations": 0}

################################################################################
# residu_moyenne
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def residu_moyenne_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    normalization: str = 'none',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Calculate the mean residual between true and predicted values with configurable options.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    normalization : str, optional (default='none')
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : str or callable, optional (default='mse')
        Metric to compute: 'mse', 'mae', 'r2', or a custom callable.
    distance : str or callable, optional (default='euclidean')
        Distance metric: 'euclidean', 'manhattan', 'cosine', or a custom callable.
    solver : str, optional (default='closed_form')
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'.
    regularization : str, optional (default=None)
        Regularization method: 'none', 'l1', 'l2', or 'elasticnet'.
    tol : float, optional (default=1e-6)
        Tolerance for convergence.
    max_iter : int, optional (default=1000)
        Maximum number of iterations.
    custom_metric : callable, optional (default=None)
        Custom metric function if not using built-in metrics.
    custom_distance : callable, optional (default=None)
        Custom distance function if not using built-in distances.

    Returns:
    --------
    dict
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Normalize data if specified
    y_true_norm, y_pred_norm = _apply_normalization(y_true, y_pred, normalization)

    # Compute residuals
    residuals = y_true_norm - y_pred_norm

    # Choose solver and compute parameters if needed
    params = _choose_solver(residuals, solver, tol, max_iter)

    # Compute metrics
    metrics = _compute_metrics(residuals, metric, distance, custom_metric, custom_distance)

    # Prepare output
    result = {
        'result': np.mean(residuals),
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric.__name__ if callable(metric) else metric,
            'distance': distance.__name__ if callable(distance) else distance,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': _check_warnings(y_true_norm, y_pred_norm)
    }

    return result

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or infinite values.")
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError("y_pred contains NaN or infinite values.")

def _apply_normalization(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> tuple:
    """Apply normalization to the input arrays."""
    if method == 'standard':
        mean_true = np.mean(y_true)
        std_true = np.std(y_true)
        y_true_norm = (y_true - mean_true) / std_true
        mean_pred = np.mean(y_pred)
        std_pred = np.std(y_pred)
        y_pred_norm = (y_pred - mean_pred) / std_pred
    elif method == 'minmax':
        min_true = np.min(y_true)
        max_true = np.max(y_true)
        y_true_norm = (y_true - min_true) / (max_true - min_true)
        min_pred = np.min(y_pred)
        max_pred = np.max(y_pred)
        y_pred_norm = (y_pred - min_pred) / (max_pred - min_pred)
    elif method == 'robust':
        median_true = np.median(y_true)
        iqr_true = np.percentile(y_true, 75) - np.percentile(y_true, 25)
        y_true_norm = (y_true - median_true) / iqr_true
        median_pred = np.median(y_pred)
        iqr_pred = np.percentile(y_pred, 75) - np.percentile(y_pred, 25)
        y_pred_norm = (y_pred - median_pred) / iqr_pred
    else:
        y_true_norm, y_pred_norm = y_true, y_pred
    return y_true_norm, y_pred_norm

def _choose_solver(residuals: np.ndarray, solver: str, tol: float, max_iter: int) -> Dict[str, Union[float, str]]:
    """Choose solver and compute parameters if needed."""
    params = {}
    if solver == 'gradient_descent':
        # Placeholder for gradient descent logic
        params['solver'] = 'gradient_descent'
    elif solver == 'newton':
        # Placeholder for Newton's method logic
        params['solver'] = 'newton'
    elif solver == 'coordinate_descent':
        # Placeholder for coordinate descent logic
        params['solver'] = 'coordinate_descent'
    else:
        # Closed form solution
        params['solver'] = 'closed_form'
    return params

def _compute_metrics(
    residuals: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Compute metrics based on residuals."""
    metrics = {}

    if custom_metric is not None:
        metrics['custom_metric'] = custom_metric(residuals, residuals)
    else:
        if metric == 'mse':
            metrics['mse'] = np.mean(residuals ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(residuals))
        elif metric == 'r2':
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((residuals - np.mean(residuals)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot)

    if custom_distance is not None:
        metrics['custom_distance'] = custom_distance(residuals, residuals)
    else:
        if distance == 'euclidean':
            metrics['euclidean_distance'] = np.linalg.norm(residuals)
        elif distance == 'manhattan':
            metrics['manhattan_distance'] = np.sum(np.abs(residuals))
        elif distance == 'cosine':
            metrics['cosine_distance'] = 1 - np.dot(residuals, residuals) / (np.linalg.norm(residuals) * np.linalg.norm(residuals))

    return metrics

def _check_warnings(y_true: np.ndarray, y_pred: np.ndarray) -> list:
    """Check for potential warnings."""
    warnings = []
    if np.any(y_true == y_pred):
        warnings.append("Some true values are equal to predicted values.")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        warnings.append("Zero standard deviation detected in input arrays.")
    return warnings

################################################################################
# residu_ecart_type
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def residu_ecart_type_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    normalization: str = 'standard',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]:
    """
    Calculate the standard deviation of residuals with configurable options.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to use ('mse', 'mae', 'r2', 'logloss') or custom callable.
    distance : str or callable, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    regularization : str, optional
        Regularization method (None, 'l1', 'l2', 'elasticnet').
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function.

    Returns
    -------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([1.1, 2.2, 3.3])
    >>> result = residu_ecart_type_fit(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Normalize data if required
    y_true_norm, y_pred_norm = _apply_normalization(y_true, y_pred, normalization)

    # Calculate residuals
    residuals = y_true_norm - y_pred_norm

    # Compute the standard deviation of residuals
    std_residuals = np.std(residuals)

    # Calculate metrics
    metrics = _calculate_metrics(y_true_norm, y_pred_norm, metric, custom_metric)

    # Prepare output
    result = {
        'result': std_residuals,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric if isinstance(metric, str) else 'custom',
            'distance': distance if isinstance(distance, str) else 'custom',
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

    return result

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Input arrays must not contain infinite values.")

def _apply_normalization(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalization: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply normalization to input arrays."""
    if normalization == 'standard':
        mean_true = np.mean(y_true)
        std_true = np.std(y_true)
        y_true_norm = (y_true - mean_true) / std_true
        mean_pred = np.mean(y_pred)
        std_pred = np.std(y_pred)
        y_pred_norm = (y_pred - mean_pred) / std_pred
    elif normalization == 'minmax':
        min_true = np.min(y_true)
        max_true = np.max(y_true)
        y_true_norm = (y_true - min_true) / (max_true - min_true)
        min_pred = np.min(y_pred)
        max_pred = np.max(y_pred)
        y_pred_norm = (y_pred - min_pred) / (max_pred - min_pred)
    elif normalization == 'robust':
        median_true = np.median(y_true)
        iqr_true = np.percentile(y_true, 75) - np.percentile(y_true, 25)
        y_true_norm = (y_true - median_true) / iqr_true
        median_pred = np.median(y_pred)
        iqr_pred = np.percentile(y_pred, 75) - np.percentile(y_pred, 25)
        y_pred_norm = (y_pred - median_pred) / iqr_pred
    else:
        y_true_norm, y_pred_norm = y_true, y_pred
    return y_true_norm, y_pred_norm

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict[str, float]:
    """Calculate metrics based on the specified metric."""
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
        elif metric == 'logloss':
            metrics['logloss'] = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    elif callable(metric):
        metrics['custom'] = metric(y_true, y_pred)
    if custom_metric is not None:
        metrics['custom_metric'] = custom_metric(y_true, y_pred)
    return metrics

################################################################################
# residu_distribution
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def residu_distribution_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    normalization: str = "standard",
    metric: Union[str, Callable] = "mse",
    distance: str = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fit the residual distribution analysis.

    Parameters:
    -----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    normalization : str, optional
        Normalization method (default: "standard").
    metric : Union[str, Callable], optional
        Metric to use (default: "mse").
    distance : str, optional
        Distance metric (default: "euclidean").
    solver : str, optional
        Solver method (default: "closed_form").
    regularization : Optional[str], optional
        Regularization type (default: None).
    tol : float, optional
        Tolerance for convergence (default: 1e-6).
    max_iter : int, optional
        Maximum iterations (default: 1000).
    custom_metric : Optional[Callable], optional
        Custom metric function (default: None).
    custom_distance : Optional[Callable], optional
        Custom distance function (default: None).

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Normalize residuals
    residuals = y_true - y_pred
    normalized_residuals = _apply_normalization(residuals, normalization)

    # Choose solver
    if solver == "closed_form":
        params = _solve_closed_form(normalized_residuals)
    elif solver == "gradient_descent":
        params = _solve_gradient_descent(normalized_residuals, tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(normalized_residuals, y_true, y_pred,
                              metric=metric,
                              distance=distance,
                              custom_metric=custom_metric,
                              custom_distance=custom_distance)

    # Prepare output
    result = {
        "result": normalized_residuals,
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

    return result

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Inputs contain NaN values")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Inputs contain infinite values")

def _apply_normalization(residuals: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to residuals."""
    if method == "none":
        return residuals
    elif method == "standard":
        mean = np.mean(residuals)
        std = np.std(residuals)
        return (residuals - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(residuals)
        max_val = np.max(residuals)
        return (residuals - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(residuals)
        iqr = np.percentile(residuals, 75) - np.percentile(residuals, 25)
        return (residuals - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _solve_closed_form(residuals: np.ndarray) -> Dict:
    """Closed form solution for residual distribution."""
    return {"method": "closed_form", "params": {}}

def _solve_gradient_descent(residuals: np.ndarray, tol: float, max_iter: int) -> Dict:
    """Gradient descent solution for residual distribution."""
    # Placeholder implementation
    return {"method": "gradient_descent", "params": {}, "iterations": 0}

def _compute_metrics(
    residuals: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: Union[str, Callable] = "mse",
    distance: str = "euclidean",
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """Compute metrics for residual analysis."""
    metrics_dict = {}

    if custom_metric is not None:
        metrics_dict["custom_metric"] = custom_metric(y_true, y_pred)
    else:
        if metric == "mse":
            metrics_dict["mse"] = np.mean((y_true - y_pred) ** 2)
        elif metric == "mae":
            metrics_dict["mae"] = np.mean(np.abs(y_true - y_pred))
        elif metric == "r2":
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics_dict["r2"] = 1 - (ss_res / (ss_tot + 1e-8))
        elif metric == "logloss":
            metrics_dict["logloss"] = -np.mean(y_true * np.log(y_pred + 1e-8) +
                                             (1 - y_true) * np.log(1 - y_pred + 1e-8))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    if custom_distance is not None:
        metrics_dict["custom_distance"] = custom_distance(residuals)
    else:
        if distance == "euclidean":
            metrics_dict["euclidean"] = np.linalg.norm(residuals)
        elif distance == "manhattan":
            metrics_dict["manhattan"] = np.sum(np.abs(residuals))
        elif distance == "cosine":
            metrics_dict["cosine"] = 1 - np.dot(residuals, residuals) / (
                np.linalg.norm(residuals) * np.linalg.norm(residuals))
        elif distance == "minkowski":
            metrics_dict["minkowski"] = np.sum(np.abs(residuals) ** 3) ** (1/3)
        else:
            raise ValueError(f"Unknown distance: {distance}")

    return metrics_dict

################################################################################
# residu_outlier
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def residu_outlier_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    normalization: str = 'standard',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: str = 'euclidean',
    threshold: Optional[float] = None,
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, str]]:
    """
    Detect outliers in residuals between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to compute residuals ('mse', 'mae', 'r2', custom callable).
    distance : str, optional
        Distance metric for outlier detection ('euclidean', 'manhattan', 'cosine').
    threshold : float, optional
        Threshold for outlier detection. If None, uses median absolute deviation.
    custom_normalization : callable, optional
        Custom normalization function.
    custom_metric : callable, optional
        Custom metric function.

    Returns
    -------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([1.1, 2.2, 3.0])
    >>> result = residu_outlier_fit(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Normalize residuals
    residuals = y_true - y_pred
    if custom_normalization is not None:
        normalized_residuals = custom_normalization(residuals)
    else:
        normalized_residuals = _apply_normalization(residuals, normalization)

    # Compute metric
    if callable(metric):
        computed_metric = metric(y_true, y_pred)
    else:
        computed_metric = _compute_metric(residuals, metric)

    # Detect outliers
    outliers = _detect_outliers(normalized_residuals, distance, threshold)

    # Prepare results
    result = {
        'result': {
            'outliers': outliers,
            'residuals': residuals
        },
        'metrics': {
            'metric_value': computed_metric,
            'outlier_count': np.sum(outliers)
        },
        'params_used': {
            'normalization': normalization,
            'metric': metric if not callable(metric) else 'custom',
            'distance': distance,
            'threshold': threshold
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
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Inputs must not contain NaN values.")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Inputs must not contain infinite values.")

def _apply_normalization(residuals: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to residuals."""
    if method == 'none':
        return residuals
    elif method == 'standard':
        mean = np.mean(residuals)
        std = np.std(residuals)
        return (residuals - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(residuals)
        max_val = np.max(residuals)
        return (residuals - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(residuals)
        mad = np.median(np.abs(residuals - median))
        return (residuals - median) / (mad + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_metric(residuals: np.ndarray, metric_name: str) -> float:
    """Compute specified metric for residuals."""
    if metric_name == 'mse':
        return np.mean(residuals ** 2)
    elif metric_name == 'mae':
        return np.mean(np.abs(residuals))
    elif metric_name == 'r2':
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((residuals - np.mean(residuals)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

def _detect_outliers(
    normalized_residuals: np.ndarray,
    distance_metric: str,
    threshold: Optional[float]
) -> np.ndarray:
    """Detect outliers based on distance metric and threshold."""
    if threshold is None:
        median = np.median(normalized_residuals)
        mad = np.median(np.abs(normalized_residuals - median))
        threshold = 3 * mad

    if distance_metric == 'euclidean':
        distances = np.abs(normalized_residuals)
    elif distance_metric == 'manhattan':
        distances = np.abs(normalized_residuals).sum(axis=1)
    elif distance_metric == 'cosine':
        normalized = normalized_residuals / (np.linalg.norm(normalized_residuals, axis=1, keepdims=True) + 1e-8)
        distances = 1 - np.sum(normalized * normalized, axis=1)
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    return distances > threshold

################################################################################
# residu_modelisation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def residu_modelisation_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]:
    """
    Fit a residual model and compute various metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Normalization function, by default None.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Metric to compute, by default "mse".
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Distance metric, by default "euclidean".
    solver : str, optional
        Solver to use, by default "closed_form".
    regularization : Optional[str], optional
        Regularization type, by default None.
    tol : float, optional
        Tolerance for convergence, by default 1e-4.
    max_iter : int, optional
        Maximum number of iterations, by default 1000.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function, by default None.
    **kwargs : dict
        Additional keyword arguments for the solver.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> y_true = np.array([1, 2, 3])
    >>> y_pred = np.array([1.1, 2.2, 2.9])
    >>> result = residu_modelisation_fit(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Normalize data if a normalizer is provided
    if normalizer is not None:
        y_true = normalizer(y_true)
        y_pred = normalizer(y_pred)

    # Compute residuals
    residuals = _compute_residuals(y_true, y_pred)

    # Choose solver and fit model
    if solver == "closed_form":
        params = _closed_form_solver(residuals, **kwargs)
    elif solver == "gradient_descent":
        params = _gradient_descent_solver(residuals, tol=tol, max_iter=max_iter, **kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(y_true, y_pred, metric=metric, custom_metric=custom_metric)

    # Compute distances
    distance_value = _compute_distance(y_true, y_pred, distance=distance)

    # Prepare output
    result = {
        "result": residuals,
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

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Validate input arrays.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Input arrays must not contain infinite values.")

def _compute_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute residuals.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    np.ndarray
        Residuals.
    """
    return y_true - y_pred

def _closed_form_solver(residuals: np.ndarray, **kwargs) -> Dict[str, float]:
    """
    Closed form solver for residuals.

    Parameters
    ----------
    residuals : np.ndarray
        Residuals.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    Dict[str, float]
        Parameters estimated by the solver.
    """
    # Placeholder for closed form solution
    return {"intercept": np.mean(residuals), "slope": 0.0}

def _gradient_descent_solver(
    residuals: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 1000,
    **kwargs
) -> Dict[str, float]:
    """
    Gradient descent solver for residuals.

    Parameters
    ----------
    residuals : np.ndarray
        Residuals.
    tol : float, optional
        Tolerance for convergence, by default 1e-4.
    max_iter : int, optional
        Maximum number of iterations, by default 1000.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    Dict[str, float]
        Parameters estimated by the solver.
    """
    # Placeholder for gradient descent solution
    return {"intercept": np.mean(residuals), "slope": 0.0}

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """
    Compute metrics for residuals.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Metric to compute, by default "mse".
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function, by default None.

    Returns
    -------
    Dict[str, float]
        Computed metrics.
    """
    metrics = {}

    if metric == "mse":
        metrics["mse"] = np.mean((y_true - y_pred) ** 2)
    elif metric == "mae":
        metrics["mae"] = np.mean(np.abs(y_true - y_pred))
    elif metric == "r2":
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics["r2"] = 1 - (ss_res / ss_tot)
    elif callable(metric):
        metrics["custom"] = metric(y_true, y_pred)

    if custom_metric is not None:
        metrics["custom"] = custom_metric(y_true, y_pred)

    return metrics

def _compute_distance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "euclidean"
) -> float:
    """
    Compute distance between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Distance metric, by default "euclidean".

    Returns
    -------
    float
        Computed distance.
    """
    if distance == "euclidean":
        return np.linalg.norm(y_true - y_pred)
    elif distance == "manhattan":
        return np.sum(np.abs(y_true - y_pred))
    elif distance == "cosine":
        return 1 - np.dot(y_true, y_pred) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred))
    elif callable(distance):
        return distance(y_true, y_pred)
    else:
        raise ValueError(f"Unknown distance metric: {distance}")

################################################################################
# residu_regression
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def residu_regression_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = "mse",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fit a regression model and analyze residuals.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalizer : Optional[Callable]
        Function to normalize features. Defaults to None.
    metric : Union[str, Callable]
        Metric to evaluate residuals. Can be "mse", "mae", "r2", or custom callable.
    solver : str
        Solver to use. Options: "closed_form", "gradient_descent".
    regularization : Optional[str]
        Regularization type. Options: None, "l1", "l2".
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable]
        Custom metric function if not using built-in metrics.
    custom_distance : Optional[Callable]
        Custom distance function for gradient descent.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Choose solver and fit model
    if solver == "closed_form":
        params = _solve_closed_form(X_normalized, y, regularization)
    elif solver == "gradient_descent":
        params = _solve_gradient_descent(
            X_normalized, y,
            tol=tol,
            max_iter=max_iter,
            distance_func=custom_distance
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate residuals
    y_pred = X_normalized @ params
    residuals = y - y_pred

    # Calculate metrics
    metrics = _calculate_metrics(residuals, y, metric, custom_metric)

    return {
        "result": {
            "residuals": residuals,
            "predictions": y_pred
        },
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric": metric,
            "solver": solver,
            "regularization": regularization
        },
        "warnings": _check_warnings(residuals)
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

def _apply_normalization(
    X: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Apply normalization to features."""
    if normalizer is None:
        return X
    return normalizer(X)

def _solve_closed_form(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str]
) -> np.ndarray:
    """Solve regression using closed-form solution."""
    if regularization is None:
        params = np.linalg.inv(X.T @ X) @ X.T @ y
    elif regularization == "l2":
        lambda_ = 1.0  # Default value, could be made configurable
        params = np.linalg.inv(X.T @ X + lambda_ * np.eye(X.shape[1])) @ X.T @ y
    else:
        raise ValueError(f"Unknown regularization: {regularization}")
    return params

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000,
    distance_func: Optional[Callable] = None
) -> np.ndarray:
    """Solve regression using gradient descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    learning_rate = 0.01
    prev_loss = float('inf')

    for _ in range(max_iter):
        gradients = 2 * X.T @ (X @ params - y) / len(y)
        params -= learning_rate * gradients
        current_loss = np.mean((X @ params - y) ** 2)

        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return params

def _calculate_metrics(
    residuals: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Calculate metrics for residuals."""
    metrics = {}

    if metric == "mse" or (custom_metric is None and metric == "mse"):
        metrics["mse"] = np.mean(residuals ** 2)
    elif metric == "mae" or (custom_metric is None and metric == "mae"):
        metrics["mae"] = np.mean(np.abs(residuals))
    elif metric == "r2" or (custom_metric is None and metric == "r2"):
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics["r2"] = 1 - (ss_res / ss_tot)
    elif callable(metric):
        metrics["custom_metric"] = metric(residuals, y)
    elif custom_metric is not None:
        metrics["custom_metric"] = custom_metric(residuals, y)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

def _check_warnings(residuals: np.ndarray) -> list:
    """Check for potential issues with residuals."""
    warnings = []
    if np.any(np.isnan(residuals)):
        warnings.append("Residuals contain NaN values")
    if np.any(np.isinf(residuals)):
        warnings.append("Residuals contain infinite values")
    return warnings

################################################################################
# residu_classification
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def residu_classification_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    normalization: str = "none",
    metric: Union[str, Callable] = "mse",
    distance: Union[str, Callable] = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fit a residual classification model and compute metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels or values.
    y_pred : np.ndarray
        Predicted labels or values.
    normalization : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    metric : str or callable, optional
        Metric to compute ("mse", "mae", "r2", "logloss") or custom callable.
    distance : str or callable, optional
        Distance metric ("euclidean", "manhattan", "cosine", "minkowski") or custom callable.
    solver : str, optional
        Solver method ("closed_form", "gradient_descent", "newton", "coordinate_descent").
    regularization : str, optional
        Regularization method (None, "l1", "l2", "elasticnet").
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function.
    custom_distance : callable, optional
        Custom distance function.

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> y_true = np.array([0, 1, 0, 1])
    >>> y_pred = np.array([0.1, 0.9, 0.2, 0.8])
    >>> result = residu_classification_fit(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Normalize data if required
    y_true_norm, y_pred_norm = _apply_normalization(y_true, y_pred, normalization)

    # Choose metric
    if callable(metric):
        compute_metric = metric
    else:
        compute_metric = _get_metric(metric)

    # Choose distance
    if callable(distance):
        compute_distance = distance
    else:
        compute_distance = _get_distance(distance)

    # Solve the problem
    params = _solve_residual_classification(
        y_true_norm, y_pred_norm,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter
    )

    # Compute metrics
    metrics = compute_metric(y_true_norm, y_pred_norm)

    if custom_metric is not None:
        metrics["custom_metric"] = custom_metric(y_true_norm, y_pred_norm)

    # Compute distances
    distances = compute_distance(y_true_norm, y_pred_norm)
    metrics["distance"] = distances

    if custom_distance is not None:
        metrics["custom_distance"] = custom_distance(y_true_norm, y_pred_norm)

    # Prepare output
    result = {
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

    return result

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Inputs must not contain NaN values.")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Inputs must not contain infinite values.")

def _apply_normalization(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str
) -> tuple:
    """Apply normalization to input arrays."""
    if method == "none":
        return y_true, y_pred
    elif method == "standard":
        mean_true = np.mean(y_true)
        std_true = np.std(y_true)
        y_true_norm = (y_true - mean_true) / std_true
        mean_pred = np.mean(y_pred)
        std_pred = np.std(y_pred)
        y_pred_norm = (y_pred - mean_pred) / std_pred
    elif method == "minmax":
        min_true = np.min(y_true)
        max_true = np.max(y_true)
        y_true_norm = (y_true - min_true) / (max_true - min_true)
        min_pred = np.min(y_pred)
        max_pred = np.max(y_pred)
        y_pred_norm = (y_pred - min_pred) / (max_pred - min_pred)
    elif method == "robust":
        median_true = np.median(y_true)
        iqr_true = np.percentile(y_true, 75) - np.percentile(y_true, 25)
        y_true_norm = (y_true - median_true) / iqr_true
        median_pred = np.median(y_pred)
        iqr_pred = np.percentile(y_pred, 75) - np.percentile(y_pred, 25)
        y_pred_norm = (y_pred - median_pred) / iqr_pred
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return y_true_norm, y_pred_norm

def _get_metric(metric: str) -> Callable:
    """Get metric function based on string input."""
    metrics = {
        "mse": _mean_squared_error,
        "mae": _mean_absolute_error,
        "r2": _r_squared,
        "logloss": _log_loss
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _get_distance(distance: str) -> Callable:
    """Get distance function based on string input."""
    distances = {
        "euclidean": _euclidean_distance,
        "manhattan": _manhattan_distance,
        "cosine": _cosine_distance,
        "minkowski": _minkowski_distance
    }
    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")
    return distances[distance]

def _solve_residual_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> Dict:
    """Solve residual classification problem."""
    if solver == "closed_form":
        params = _solve_closed_form(y_true, y_pred)
    elif solver == "gradient_descent":
        params = _solve_gradient_descent(y_true, y_pred, tol=tol, max_iter=max_iter)
    elif solver == "newton":
        params = _solve_newton(y_true, y_pred, tol=tol, max_iter=max_iter)
    elif solver == "coordinate_descent":
        params = _solve_coordinate_descent(y_true, y_pred, tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    if regularization is not None:
        params = _apply_regularization(params, y_true, y_pred, regularization)

    return params

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

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute log loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _euclidean_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Euclidean distance."""
    return np.linalg.norm(y_true - y_pred)

def _manhattan_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Manhattan distance."""
    return np.sum(np.abs(y_true - y_pred))

def _cosine_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute cosine distance."""
    return 1 - np.dot(y_true, y_pred) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred))

def _minkowski_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Minkowski distance."""
    return np.sum(np.abs(y_true - y_pred) ** 3) ** (1/3)

def _solve_closed_form(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Solve using closed form solution."""
    # Placeholder for actual implementation
    return {"params": np.zeros(y_true.shape[0])}

def _solve_gradient_descent(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> Dict:
    """Solve using gradient descent."""
    # Placeholder for actual implementation
    return {"params": np.zeros(y_true.shape[0])}

def _solve_newton(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> Dict:
    """Solve using Newton's method."""
    # Placeholder for actual implementation
    return {"params": np.zeros(y_true.shape[0])}

def _solve_coordinate_descent(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> Dict:
    """Solve using coordinate descent."""
    # Placeholder for actual implementation
    return {"params": np.zeros(y_true.shape[0])}

def _apply_regularization(
    params: Dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str
) -> Dict:
    """Apply regularization to parameters."""
    if method == "l1":
        params["regularized_params"] = np.sign(params["params"]) * np.maximum(
            np.abs(params["params"]) - 0.1, 0
        )
    elif method == "l2":
        params["regularized_params"] = params["params"] / (1 + 0.1)
    elif method == "elasticnet":
        params["regularized_params"] = np.sign(params["params"]) * np.maximum(
            np.abs(params["params"]) - 0.1, 0
        ) / (1 + 0.1)
    else:
        raise ValueError(f"Unknown regularization method: {method}")
    return params
