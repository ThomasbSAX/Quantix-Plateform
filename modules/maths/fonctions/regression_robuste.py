"""
Quantix – Module regression_robuste
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# outliers
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def outliers_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    distance_metric: str = "euclidean",
    solver: str = "closed_form",
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    max_iter: int = 100,
    tol: float = 1e-4,
    alpha: float = 0.0,
    l1_ratio: Optional[float] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]:
    """
    Detect outliers in regression using robust methods.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the input data.
    distance_metric : str
        Distance metric for outlier detection ("euclidean", "manhattan", "cosine", "minkowski").
    solver : str
        Solver method ("closed_form", "gradient_descent", "newton", "coordinate_descent").
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Metric for evaluation ("mse", "mae", "r2", "logloss") or custom callable.
    max_iter : int
        Maximum number of iterations for iterative solvers.
    tol : float
        Tolerance for convergence.
    alpha : float
        Regularization strength (0 for no regularization).
    l1_ratio : Optional[float]
        ElasticNet mixing parameter (0 for L2, 1 for L1).
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom distance function.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]
        Dictionary containing:
        - "result": Outlier detection results.
        - "metrics": Computed metrics.
        - "params_used": Parameters used in the computation.
        - "warnings": List of warnings.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = outliers_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm = normalizer(X)
    y_norm = normalizer(y.reshape(-1, 1)).flatten()

    # Choose distance metric
    if custom_distance is not None:
        distance_func = custom_distance
    else:
        distance_func = _get_distance_metric(distance_metric)

    # Choose solver
    if solver == "closed_form":
        params = _solve_closed_form(X_norm, y_norm, alpha, l1_ratio)
    elif solver == "gradient_descent":
        params = _solve_gradient_descent(X_norm, y_norm, alpha, l1_ratio, max_iter, tol)
    elif solver == "newton":
        params = _solve_newton(X_norm, y_norm, alpha, l1_ratio, max_iter, tol)
    elif solver == "coordinate_descent":
        params = _solve_coordinate_descent(X_norm, y_norm, alpha, l1_ratio, max_iter, tol)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute residuals
    residuals = y_norm - np.dot(X_norm, params)

    # Detect outliers based on residuals
    outlier_indices = _detect_outliers(residuals, distance_func)

    # Compute metrics
    if isinstance(metric, str):
        metric_func = _get_metric(metric)
    else:
        metric_func = metric

    metrics = {
        "residual_mse": np.mean(residuals**2),
        "custom_metric": metric_func(y_norm, np.dot(X_norm, params))
    }

    # Prepare output
    result = {
        "outlier_indices": outlier_indices,
        "residuals": residuals,
        "params": params
    }

    return {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if hasattr(normalizer, "__name__") else "custom",
            "distance_metric": distance_metric,
            "solver": solver,
            "metric": metric if isinstance(metric, str) else "custom",
            "max_iter": max_iter,
            "tol": tol,
            "alpha": alpha,
            "l1_ratio": l1_ratio
        },
        "warnings": []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or inf values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or inf values")

def _get_distance_metric(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get distance metric function."""
    metrics = {
        "euclidean": lambda a, b: np.linalg.norm(a - b),
        "manhattan": lambda a, b: np.sum(np.abs(a - b)),
        "cosine": lambda a, b: 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)),
        "minkowski": lambda a, b: np.sum(np.abs(a - b)**3)**(1/3)
    }
    if metric not in metrics:
        raise ValueError(f"Unknown distance metric: {metric}")
    return metrics[metric]

def _get_metric(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get metric function."""
    metrics = {
        "mse": lambda y_true, y_pred: np.mean((y_true - y_pred)**2),
        "mae": lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)),
        "r2": lambda y_true, y_pred: 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2),
        "logloss": lambda y_true, y_pred: -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _solve_closed_form(X: np.ndarray, y: np.ndarray, alpha: float, l1_ratio: Optional[float]) -> np.ndarray:
    """Solve regression using closed form solution."""
    if l1_ratio is not None and 0 < l1_ratio < 1:
        raise NotImplementedError("ElasticNet not implemented for closed form")
    if alpha > 0 and l1_ratio is None:
        # Ridge regression
        I = np.eye(X.shape[1])
        params = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    else:
        # Ordinary least squares
        params = np.linalg.inv(X.T @ X) @ X.T @ y
    return params

def _solve_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, l1_ratio: Optional[float],
                           max_iter: int, tol: float) -> np.ndarray:
    """Solve regression using gradient descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ params - y)
        if alpha > 0:
            if l1_ratio is None or l1_ratio == 0:
                gradient += 2 * alpha * params
            elif l1_ratio == 1:
                gradient += alpha * np.sign(params)
            else:
                gradient += alpha * (l1_ratio * np.sign(params) + (1 - l1_ratio) * 2 * params)
        new_params = params - tol * gradient
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params
    return params

def _solve_newton(X: np.ndarray, y: np.ndarray, alpha: float, l1_ratio: Optional[float],
                  max_iter: int, tol: float) -> np.ndarray:
    """Solve regression using Newton's method."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ params - y)
        hessian = 2 * X.T @ X
        if alpha > 0:
            if l1_ratio is None or l1_ratio == 0:
                hessian += 2 * alpha * np.eye(n_features)
            elif l1_ratio == 1:
                gradient += alpha * np.sign(params)
            else:
                gradient += alpha * (l1_ratio * np.sign(params) + (1 - l1_ratio) * 2 * params)
        delta = np.linalg.inv(hessian) @ gradient
        new_params = params - delta
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params
    return params

def _solve_coordinate_descent(X: np.ndarray, y: np.ndarray, alpha: float, l1_ratio: Optional[float],
                             max_iter: int, tol: float) -> np.ndarray:
    """Solve regression using coordinate descent."""
    n_samples, n_features = X.shape
    params = np.zeros(n_features)
    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - np.dot(X, params) + params[j] * X_j
            if l1_ratio is None or l1_ratio == 0:
                # Ridge regression
                numerator = X_j.T @ residuals + alpha * params[j]
                denominator = X_j.T @ X_j + alpha
            elif l1_ratio == 1:
                # Lasso regression
                numerator = X_j.T @ residuals
                denominator = np.linalg.norm(X_j)
                soft_threshold = numerator / denominator - alpha * np.sign(numerator) if numerator != 0 else 0
                params[j] = soft_threshold * (abs(numerator) > alpha)
            else:
                # ElasticNet
                numerator = X_j.T @ residuals + alpha * (l1_ratio * np.sign(params[j]) + (1 - l1_ratio) * params[j])
                denominator = X_j.T @ X_j + alpha * (1 - l1_ratio)
                params[j] = numerator / denominator
        if np.linalg.norm(params - np.roll(params, 1)) < tol:
            break
    return params

def _detect_outliers(residuals: np.ndarray, distance_func: Callable[[np.ndarray], float]) -> np.ndarray:
    """Detect outliers based on residuals."""
    median_residual = np.median(residuals)
    mad = np.median(np.abs(residuals - median_residual))
    standardized_residuals = 0.6745 * (residuals - median_residual) / mad
    outlier_indices = np.where(np.abs(standardized_residuals) > 3)[0]
    return outlier_indices

################################################################################
# loss_functions
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def loss_functions_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loss_func: Union[str, Callable] = "mse",
    normalization: str = "none",
    metric_funcs: Optional[Union[str, Callable, list]] = None,
    weights: Optional[np.ndarray] = None,
    **kwargs
) -> Dict:
    """
    Compute loss functions and metrics for robust regression.

    Parameters:
    -----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values from the regression model.
    loss_func : str or callable, optional (default="mse")
        Loss function to compute. Can be "mse", "mae", "huber", or a custom callable.
    normalization : str, optional (default="none")
        Normalization method. Can be "none", "standard", "minmax", or "robust".
    metric_funcs : str, callable, or list, optional (default=None)
        Metrics to compute. Can be "r2", "mae", "mse", or custom callables.
    weights : np.ndarray, optional (default=None)
        Weights for weighted loss computation.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred, weights)

    # Normalize data if required
    y_true_norm, y_pred_norm = _apply_normalization(y_true, y_pred, normalization)

    # Compute loss
    loss_value = _compute_loss(
        y_true_norm,
        y_pred_norm,
        loss_func=loss_func,
        weights=weights
    )

    # Compute metrics if requested
    metrics = {}
    if metric_funcs is not None:
        metrics = _compute_metrics(
            y_true_norm,
            y_pred_norm,
            metric_funcs=metric_funcs
        )

    # Prepare output dictionary
    result = {
        "result": {"loss": loss_value},
        "metrics": metrics,
        "params_used": {
            "loss_func": loss_func if isinstance(loss_func, str) else "custom",
            "normalization": normalization,
        },
        "warnings": []
    }

    return result

def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> None:
    """Validate input arrays."""
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if weights is not None and y_true.shape != weights.shape:
        raise ValueError("weights must have the same shape as y_true and y_pred")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or Inf values")
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError("y_pred contains NaN or Inf values")
    if weights is not None and (np.any(np.isnan(weights)) or np.any(np.isinf(weights))):
        raise ValueError("weights contains NaN or Inf values")

def _apply_normalization(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str = "none"
) -> tuple:
    """Apply normalization to input arrays."""
    if method == "standard":
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
        y_true_norm, y_pred_norm = y_true.copy(), y_pred.copy()
    return y_true_norm, y_pred_norm

def _compute_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loss_func: Union[str, Callable] = "mse",
    weights: Optional[np.ndarray] = None
) -> float:
    """Compute the specified loss function."""
    residuals = y_true - y_pred

    if weights is not None:
        residuals = residuals * weights

    if loss_func == "mse":
        return np.mean(residuals ** 2)
    elif loss_func == "mae":
        return np.mean(np.abs(residuals))
    elif loss_func == "huber":
        delta = 1.0
        abs_residuals = np.abs(residuals)
        quadratic = np.minimum(abs_residuals, delta)
        linear = abs_residuals - quadratic
        return np.mean(0.5 * quadratic ** 2 + delta * linear)
    elif callable(loss_func):
        return loss_func(y_true, y_pred)
    else:
        raise ValueError(f"Unknown loss function: {loss_func}")

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_funcs: Union[str, Callable, list]
) -> Dict:
    """Compute the specified metrics."""
    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    metrics = {}

    if isinstance(metric_funcs, str):
        metric_funcs = [metric_funcs]
    elif callable(metric_funcs):
        metrics["custom"] = metric_funcs(y_true, y_pred)
        return metrics

    for func in metric_funcs:
        if func == "r2":
            metrics["r2"] = 1 - (ss_res / ss_tot)
        elif func == "mae":
            metrics["mae"] = np.mean(np.abs(residuals))
        elif func == "mse":
            metrics["mse"] = np.mean(residuals ** 2)
        elif callable(func):
            metrics["custom"] = func(y_true, y_pred)

    return metrics

################################################################################
# M_estimators
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

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
                  custom_norm: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize input data."""
    if custom_norm is not None:
        X_norm = custom_norm(X)
        y_norm = custom_norm(y.reshape(-1, 1)).flatten()
    elif normalization == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_norm = (X - X_mean) / (X_std + 1e-8)
        y_norm = (y - np.mean(y)) / (np.std(y) + 1e-8)
    elif normalization == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
        y_min = np.min(y)
        y_max = np.max(y)
        y_norm = (y - y_min) / (y_max - y_min + 1e-8)
    elif normalization == 'robust':
        X_median = np.median(X, axis=0)
        X_q75 = np.percentile(X, 75, axis=0)
        X_q25 = np.percentile(X, 25, axis=0)
        X_iqr = X_q75 - X_q25
        X_norm = (X - X_median) / (X_iqr + 1e-8)
        y_median = np.median(y)
        y_q75 = np.percentile(y, 75)
        y_q25 = np.percentile(y, 25)
        y_iqr = y_q75 - y_q25
        y_norm = (y - y_median) / (y_iqr + 1e-8)
    else:
        X_norm = X.copy()
        y_norm = y.copy()

    return X_norm, y_norm

def compute_residuals(X: np.ndarray, y: np.ndarray,
                     weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute residuals with optional weights."""
    if weights is not None and len(weights) != X.shape[0]:
        raise ValueError("Weights must have the same length as number of samples")
    return y - X @ beta

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   metric: str = 'mse',
                   custom_metric: Optional[Callable] = None) -> Dict[str, float]:
    """Compute regression metrics."""
    if custom_metric is not None:
        return {'custom': custom_metric(y_true, y_pred)}

    metrics = {}
    if metric == 'mse' or 'all':
        metrics['mse'] = np.mean((y_true - y_pred)**2)
    if metric == 'mae' or 'all':
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    if metric == 'r2' or 'all':
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        metrics['r2'] = 1 - (ss_res / ss_tot)

    return metrics

def m_estimators_fit(X: np.ndarray, y: np.ndarray,
                    normalization: str = 'standard',
                    metric: str = 'mse',
                    solver: str = 'closed_form',
                    max_iter: int = 100,
                    tol: float = 1e-4,
                    custom_norm: Optional[Callable] = None,
                    custom_metric: Optional[Callable] = None) -> Dict:
    """Fit M-estimators for robust regression."""
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = normalize_data(X, y, normalization, custom_norm)

    # Initialize parameters
    beta = np.zeros(X.shape[1])
    prev_beta = None

    # Solver selection
    if solver == 'closed_form':
        beta, _ = np.linalg.lstsq(X_norm, y_norm, rcond=None)
    elif solver == 'gradient_descent':
        learning_rate = 0.01
        for _ in range(max_iter):
            residuals = compute_residuals(X_norm, y_norm, beta)
            gradient = -2 * X_norm.T @ residuals / len(y_norm)
            beta -= learning_rate * gradient
    elif solver == 'newton':
        for _ in range(max_iter):
            residuals = compute_residuals(X_norm, y_norm, beta)
            hessian = 2 * X_norm.T @ X_norm / len(y_norm)
            gradient = -2 * X_norm.T @ residuals / len(y_norm)
            beta -= np.linalg.solve(hessian, gradient)

    # Compute predictions and metrics
    y_pred = X_norm @ beta
    metrics = compute_metrics(y_norm, y_pred, metric, custom_metric)

    return {
        'result': beta,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'solver': solver
        },
        'warnings': []
    }

# Example usage:
"""
X = np.random.rand(100, 5)
y = np.random.rand(100)
result = M_estimators_fit(X, y,
                         normalization='standard',
                         metric='mse',
                         solver='closed_form')
"""

################################################################################
# L1_regression
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for L1 regression."""
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

def normalize_data(X: np.ndarray, y: np.ndarray, normalization: str = 'standard') -> tuple:
    """Normalize input data based on specified method."""
    if normalization == 'none':
        return X, y
    elif normalization == 'standard':
        mean_X = np.mean(X, axis=0)
        std_X = np.std(X, axis=0)
        X_normalized = (X - mean_X) / std_X
        mean_y = np.mean(y)
        std_y = np.std(y)
        y_normalized = (y - mean_y) / std_y
        return X_normalized, y_normalized
    elif normalization == 'minmax':
        min_X = np.min(X, axis=0)
        max_X = np.max(X, axis=0)
        X_normalized = (X - min_X) / (max_X - min_X + 1e-8)
        min_y = np.min(y)
        max_y = np.max(y)
        y_normalized = (y - min_y) / (max_y - min_y + 1e-8)
        return X_normalized, y_normalized
    elif normalization == 'robust':
        median_X = np.median(X, axis=0)
        iqr_X = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_normalized = (X - median_X) / (iqr_X + 1e-8)
        median_y = np.median(y)
        iqr_y = np.subtract(*np.percentile(y, [75, 25]))
        y_normalized = (y - median_y) / (iqr_y + 1e-8)
        return X_normalized, y_normalized
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute specified metrics between true and predicted values."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(y_true, y_pred)
        except Exception as e:
            metrics[name] = np.nan
            print(f"Warning: Could not compute metric {name}: {str(e)}")
    return metrics

def l1_regression_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve L1 regression using closed-form solution (Lasso)."""
    from sklearn.linear_model import Lasso
    model = Lasso(alpha=1.0, fit_intercept=True)
    model.fit(X, y)
    return model.coef_

def l1_regression_coordinate_descent(X: np.ndarray, y: np.ndarray,
                                    max_iter: int = 1000, tol: float = 1e-4) -> np.ndarray:
    """Solve L1 regression using coordinate descent."""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)
    alpha = 1.0

    for _ in range(max_iter):
        old_coef = coef.copy()
        for j in range(n_features):
            X_j = X[:, j]
            r = y - np.dot(X, coef) + coef[j] * X_j
            rho = np.dot(X_j, X_j)
            if rho < 1e-10:
                continue
            c = np.dot(X_j, r) / rho

            if c < -alpha / 2:
                coef[j] = (c + alpha / 2)
            elif c > alpha / 2:
                coef[j] = (c - alpha / 2)
            else:
                coef[j] = 0

        if np.linalg.norm(coef - old_coef) < tol:
            break

    return coef

def L1_regression_fit(X: np.ndarray, y: np.ndarray,
                     normalization: str = 'standard',
                     solver: str = 'coordinate_descent',
                     max_iter: int = 1000,
                     tol: float = 1e-4,
                     metric_funcs: Optional[Dict[str, Callable]] = None) -> Dict:
    """
    Perform L1 regression with configurable parameters.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    solver : str, optional
        Solver method ('closed_form' or 'coordinate_descent')
    max_iter : int, optional
        Maximum number of iterations for iterative solvers
    tol : float, optional
        Tolerance for convergence
    metric_funcs : dict, optional
        Dictionary of metric functions to compute

    Returns
    -------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = L1_regression_fit(X, y)
    """
    # Validate inputs
    validate_inputs(X, y)

    # Initialize default metric functions if none provided
    if metric_funcs is None:
        def mse(y_true, y_pred): return np.mean((y_true - y_pred) ** 2)
        def mae(y_true, y_pred): return np.mean(np.abs(y_true - y_pred))
        metric_funcs = {'mse': mse, 'mae': mae}

    # Normalize data
    X_norm, y_norm = normalize_data(X, y, normalization)

    # Solve regression
    if solver == 'closed_form':
        coef = l1_regression_closed_form(X_norm, y_norm)
    elif solver == 'coordinate_descent':
        coef = l1_regression_coordinate_descent(X_norm, y_norm,
                                               max_iter=max_iter, tol=tol)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Make predictions
    y_pred = np.dot(X_norm, coef)

    # Compute metrics
    metrics = compute_metrics(y_norm, y_pred, metric_funcs)

    # Return results
    return {
        'result': {'coefficients': coef},
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'solver': solver,
            'max_iter': max_iter,
            'tol': tol
        },
        'warnings': []
    }

################################################################################
# L2_regression
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input matrices for L2 regression."""
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
                  custom_normalize: Optional[Callable] = None) -> tuple:
    """Normalize features and target."""
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
                   metric: str = 'mse',
                   custom_metric: Optional[Callable] = None) -> Dict[str, float]:
    """Compute regression metrics."""
    metrics = {}
    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)
    if metric == 'mse' or 'all':
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    if metric == 'mae' or 'all':
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    if metric == 'r2' or 'all':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)

    return metrics

def closed_form_solution(X: np.ndarray, y: np.ndarray,
                        alpha: float = 0.0,
                        l1_ratio: Optional[float] = None) -> np.ndarray:
    """Compute closed form solution for L2 regression."""
    if alpha > 0 and l1_ratio is not None:
        # Elastic net case
        penalty = alpha * (l1_ratio * np.eye(X.shape[1]) +
                          (1 - l1_ratio) * np.diag(np.linalg.norm(X, axis=0)))
    elif alpha > 0:
        # Ridge case
        penalty = alpha * np.eye(X.shape[1])
    else:
        # No regularization
        penalty = np.zeros((X.shape[1], X.shape[1]))

    # Add intercept term
    X_aug = np.column_stack([np.ones(X.shape[0]), X])
    penalty_aug = np.block([[0, np.zeros(X.shape[1])],
                           [np.zeros(X.shape[1]), penalty]])

    # Solve (X'X + λI)β = X'y
    beta = np.linalg.solve(X_aug.T @ X_aug + penalty_aug, X_aug.T @ y)

    return beta

def L2_regression_fit(X: np.ndarray, y: np.ndarray,
                     normalization: str = 'standard',
                     metric: str = 'mse',
                     solver: str = 'closed_form',
                     alpha: float = 0.0,
                     l1_ratio: Optional[float] = None,
                     custom_normalize: Optional[Callable] = None,
                     custom_metric: Optional[Callable] = None) -> Dict:
    """Fit L2 regression model with configurable options.

    Example:
        >>> X = np.random.rand(100, 5)
        >>> y = np.random.rand(100)
        >>> result = L2_regression_fit(X, y,
        ...                          normalization='robust',
        ...                          metric='r2',
        ...                          solver='closed_form')
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = normalize_data(X, y,
                                   normalization=normalization,
                                   custom_normalize=custom_normalize)

    # Solve regression
    if solver == 'closed_form':
        beta = closed_form_solution(X_norm, y_norm,
                                  alpha=alpha,
                                  l1_ratio=l1_ratio)
    else:
        raise ValueError(f"Solver {solver} not implemented")

    # Make predictions
    y_pred = X_norm @ beta[1:] + beta[0]

    # Compute metrics
    metrics = compute_metrics(y_norm, y_pred,
                            metric=metric,
                            custom_metric=custom_metric)

    # Prepare output
    result = {
        'result': {'coefficients': beta},
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'solver': solver,
            'alpha': alpha,
            'l1_ratio': l1_ratio
        },
        'warnings': []
    }

    return result

################################################################################
# Huber_loss
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    delta: float = 1.35
) -> None:
    """Validate input arrays and parameters."""
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Inputs contain NaN values.")
    if delta <= 0:
        raise ValueError("Delta must be positive.")

def _huber_loss_criterion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    delta: float = 1.35
) -> np.ndarray:
    """Compute Huber loss for given true and predicted values."""
    residuals = y_true - y_pred
    abs_residuals = np.abs(residuals)
    quadratic_mask = abs_residuals <= delta
    linear_mask = ~quadratic_mask

    loss = np.zeros_like(residuals)
    loss[quadratic_mask] = 0.5 * residuals[quadratic_mask]**2
    loss[linear_mask] = delta * (abs_residuals[linear_mask] - 0.5 * delta)

    return loss

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_funcs: Dict[str, Callable]
) -> Dict[str, float]:
    """Compute specified metrics using provided functions."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(y_true, y_pred)
        except Exception as e:
            metrics[name] = np.nan
            print(f"Warning: Could not compute {name} metric. Error: {str(e)}")
    return metrics

def Huber_loss_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    delta: float = 1.35,
    metric_funcs: Optional[Dict[str, Callable]] = None,
    normalize: bool = False
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, Union[str, float]], list]]:
    """
    Compute Huber loss between true and predicted values.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    delta : float, optional
        Threshold for Huber loss (default: 1.35).
    metric_funcs : Dict[str, Callable], optional
        Dictionary of metric functions to compute (default: None).
    normalize : bool, optional
        Whether to normalize inputs before computation (default: False).

    Returns:
    --------
    Dict containing:
        - "result": computed Huber loss values
        - "metrics": dictionary of computed metrics
        - "params_used": parameters used in computation
        - "warnings": list of warnings encountered

    Example:
    --------
    >>> y_true = np.array([1, 2, 3])
    >>> y_pred = np.array([1.1, 2.9, 3.0])
    >>> result = Huber_loss_fit(y_true, y_pred)
    """
    # Initialize output dictionary
    output: Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, Union[str, float]], list]] = {
        "result": None,
        "metrics": {},
        "params_used": {},
        "warnings": []
    }

    # Validate inputs
    _validate_inputs(y_true, y_pred, delta)

    # Store used parameters
    output["params_used"] = {
        "delta": delta,
        "normalize": normalize
    }

    # Normalize inputs if requested
    if normalize:
        mean = np.mean(y_true)
        std = np.std(y_true)
        y_true_norm = (y_true - mean) / std
        y_pred_norm = (y_pred - mean) / std
    else:
        y_true_norm = y_true.copy()
        y_pred_norm = y_pred.copy()

    # Compute Huber loss
    try:
        output["result"] = _huber_loss_criterion(y_true_norm, y_pred_norm, delta)
    except Exception as e:
        output["warnings"].append(f"Error computing Huber loss: {str(e)}")
        return output

    # Compute additional metrics if requested
    if metric_funcs is not None:
        try:
            output["metrics"] = _compute_metrics(y_true_norm, y_pred_norm, metric_funcs)
        except Exception as e:
            output["warnings"].append(f"Error computing metrics: {str(e)}")

    return output

################################################################################
# Tukey_loss
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or inf values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or inf values")

def normalize_data(X: np.ndarray, y: np.ndarray, normalization: str = "standard") -> tuple:
    """Normalize data based on specified method."""
    if normalization == "none":
        return X, y
    elif normalization == "standard":
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1.0
        X_normalized = (X - X_mean) / X_std
        y_mean = np.mean(y)
        y_std = np.std(y)
        if y_std == 0:
            y_std = 1.0
        y_normalized = (y - y_mean) / y_std
        return X_normalized, y_normalized
    elif normalization == "minmax":
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1.0
        X_normalized = (X - X_min) / X_range
        y_min = np.min(y)
        y_max = np.max(y)
        if y_max == y_min:
            y_normalized = np.zeros_like(y)
        else:
            y_normalized = (y - y_min) / (y_max - y_min)
        return X_normalized, y_normalized
    elif normalization == "robust":
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_iqr[X_iqr == 0] = 1.0
        X_normalized = (X - X_median) / X_iqr
        y_median = np.median(y)
        y_iqr = np.subtract(*np.percentile(y, [75, 25]))
        if y_iqr == 0:
            y_iqr = 1.0
        y_normalized = (y - y_median) / y_iqr
        return X_normalized, y_normalized
    else:
        raise ValueError("Unknown normalization method")

def tukey_loss(residuals: np.ndarray, c: float = 4.6851) -> np.ndarray:
    """Compute Tukey's biweight loss function."""
    abs_residuals = np.abs(residuals)
    mask = abs_residuals <= c
    loss = np.zeros_like(residuals)
    loss[mask] = (c**2 / 6) * (1 - (1 - (abs_residuals[mask]/c)**2)**3)
    return loss

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute specified metrics."""
    metrics = {}
    for name, func in metric_funcs.items():
        if name == "mse":
            metrics[name] = np.mean((y_true - y_pred)**2)
        elif name == "mae":
            metrics[name] = np.mean(np.abs(y_true - y_pred))
        elif name == "r2":
            ss_res = np.sum((y_true - y_pred)**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            metrics[name] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        elif name == "logloss":
            metrics[name] = -np.mean(y_true * np.log(y_pred + 1e-15) +
                                    (1 - y_true) * np.log(1 - y_pred + 1e-15))
        else:
            metrics[name] = func(y_true, y_pred)
    return metrics

def Tukey_loss_fit(X: np.ndarray, y: np.ndarray,
                   normalization: str = "standard",
                   metric_funcs: Optional[Dict[str, Union[str, Callable]]] = None,
                   solver: str = "gradient_descent",
                   max_iter: int = 1000,
                   tol: float = 1e-4,
                   c: float = 4.6851) -> Dict:
    """Fit a robust regression model using Tukey's loss function."""
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = normalize_data(X, y, normalization)

    # Initialize parameters (simple linear regression for demonstration)
    n_features = X_norm.shape[1]
    params = np.zeros(n_features + 1)  # intercept + coefficients

    # Default metrics if none provided
    default_metrics = {"mse": "mse"}
    if metric_funcs is None:
        metric_funcs = default_metrics
    else:
        # Convert string names to actual functions if needed
        for name, func in metric_funcs.items():
            if isinstance(func, str):
                if func == "mse":
                    metric_funcs[name] = lambda y_true, y_pred: np.mean((y_true - y_pred)**2)
                elif func == "mae":
                    metric_funcs[name] = lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))
                elif func == "r2":
                    metric_funcs[name] = lambda y_true, y_pred: 1 - np.sum((y_true - y_pred)**2)/np.sum((y_true - np.mean(y_true))**2)
                elif func == "logloss":
                    metric_funcs[name] = lambda y_true, y_pred: -np.mean(y_true * np.log(y_pred + 1e-15) +
                                                                        (1 - y_true) * np.log(1 - y_pred + 1e-15))

    # Solver implementation (simplified gradient descent for demonstration)
    if solver == "gradient_descent":
        for _ in range(max_iter):
            # Predictions
            y_pred = X_norm @ params[1:] + params[0]

            # Residuals
            residuals = y_norm - y_pred

            # Tukey loss and its derivative
            loss = tukey_loss(residuals, c)
            grad_intercept = -np.sum(loss) / X_norm.shape[0]
            grad_coefs = -(X_norm.T @ loss) / X_norm.shape[0]

            # Update parameters
            params[0] -= tol * grad_intercept
            params[1:] -= tol * grad_coefs

    # Final predictions and metrics
    y_pred = X_norm @ params[1:] + params[0]
    metrics = compute_metrics(y_norm, y_pred, metric_funcs)

    # Denormalize parameters if needed
    if normalization != "none":
        # For simplicity, we return normalized metrics and params here
        pass

    return {
        "result": {"params": params.tolist()},
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "solver": solver,
            "max_iter": max_iter,
            "tol": tol,
            "c": c
        },
        "warnings": []
    }

# Example usage:
"""
X = np.random.rand(100, 5)
y = X @ np.array([2.0, -1.5, 3.0, 0.5, -2.0]) + np.random.normal(0, 0.1, 100)
result = Tukey_loss_fit(X, y, normalization="standard", metric_funcs={"mse": "mse"})
"""

################################################################################
# RANSAC
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays X and y."""
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

def default_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Default metric: Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def default_solver(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Default solver: Closed form solution for linear regression."""
    X_tx = np.dot(X.T, X)
    if np.linalg.det(X_tx) == 0:
        raise ValueError("X^T X is singular, cannot compute closed form solution")
    return np.linalg.solve(X_tx, np.dot(X.T, y))

def default_normalization(X: np.ndarray) -> np.ndarray:
    """Default normalization: Standard scaling."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / (std + 1e-8)

def ransac_fit(
    X: np.ndarray,
    y: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = default_metric,
    solver: Callable[[np.ndarray, np.ndarray], np.ndarray] = default_solver,
    normalization: Optional[Callable[[np.ndarray], np.ndarray]] = default_normalization,
    max_iterations: int = 100,
    residual_threshold: float = 2.5,
    min_samples: int = 3,
    stop_probability: float = 0.99,
) -> Dict[str, Any]:
    """
    RANSAC (Random Sample Consensus) algorithm for robust regression.

    Parameters:
    - X: Input features, shape (n_samples, n_features)
    - y: Target values, shape (n_samples,)
    - metric: Function to compute the error metric
    - solver: Function to solve the regression problem
    - normalization: Function to normalize the input data
    - max_iterations: Maximum number of RANSAC iterations
    - residual_threshold: Threshold for inlier classification
    - min_samples: Minimum number of samples to fit the model
    - stop_probability: Probability to stop when no better model is found

    Returns:
    - Dictionary containing the results, metrics, parameters used, and warnings
    """
    validate_inputs(X, y)

    n_samples = X.shape[0]
    best_inliers = np.zeros(n_samples, dtype=bool)
    best_model = None
    best_error = float('inf')
    best_params_used = {}

    if normalization is not None:
        X_normalized = normalization(X)
    else:
        X_normalized = X

    for _ in range(max_iterations):
        # Randomly select a subset of samples
        sample_indices = np.random.choice(n_samples, min_samples, replace=False)
        X_sample = X_normalized[sample_indices]
        y_sample = y[sample_indices]

        try:
            # Fit the model to the sample
            model_params = solver(X_sample, y_sample)

            # Predict on all data points
            y_pred = np.dot(X_normalized, model_params)

            # Compute residuals
            residuals = np.abs(y - y_pred)
            inliers = residuals < residual_threshold

            # Compute error metric
            current_error = metric(y[inliers], y_pred[inliers])

            # Update best model if current one is better
            if current_error < best_error:
                best_inliers = inliers
                best_model = model_params
                best_error = current_error
                best_params_used = {
                    'metric': metric.__name__,
                    'solver': solver.__name__,
                    'normalization': normalization.__name__ if normalization else None,
                    'max_iterations': max_iterations,
                    'residual_threshold': residual_threshold,
                    'min_samples': min_samples,
                    'stop_probability': stop_probability
                }

        except Exception as e:
            continue

    # Final fit on all inliers
    if np.sum(best_inliers) >= min_samples:
        X_inliers = X_normalized[best_inliers]
        y_inliers = y[best_inliers]
        best_model = solver(X_inliers, y_inliers)
        y_pred_final = np.dot(X_normalized, best_model)
    else:
        # Fallback to fit on all data if no inliers found
        best_model = solver(X_normalized, y)
        y_pred_final = np.dot(X_normalized, best_model)

    # Compute final metrics
    final_metrics = {
        'inlier_ratio': np.mean(best_inliers),
        'error': metric(y, y_pred_final)
    }

    return {
        'result': best_model,
        'metrics': final_metrics,
        'params_used': best_params_used,
        'warnings': []
    }

# Example usage:
"""
X = np.random.rand(100, 2)
y = X.dot(np.array([1.5, -2.0])) + np.random.normal(0, 0.1, 100)
y[50:] += 10  # Add outliers

result = ransac_fit(X, y)
print(result['result'])
"""

################################################################################
# Theil_Sen_estimator
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate input arrays for Theil-Sen estimator.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)

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
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Inputs contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Inputs contain infinite values")

def normalize_data(X: np.ndarray, y: np.ndarray,
                  normalization: str = 'none',
                  custom_norm: Optional[Callable] = None) -> tuple:
    """
    Normalize data according to specified method.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalization : str or callable
        Normalization method ('none', 'standard', 'minmax', 'robust')
    custom_norm : callable, optional
        Custom normalization function

    Returns
    -------
    tuple
        (X_normalized, y_normalized)
    """
    if custom_norm is not None:
        X = custom_norm(X)
        y = custom_norm(y.reshape(-1, 1)).flatten()
    elif normalization == 'standard':
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y = (y - np.mean(y)) / np.std(y)
    elif normalization == 'minmax':
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif normalization == 'robust':
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        y = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    return X, y

def compute_median_slopes(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Compute median slopes for Theil-Sen estimator.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)

    Returns
    -------
    tuple
        (median_slopes, intercept)
    """
    n_samples = X.shape[0]
    slopes = np.zeros((n_samples, X.shape[1]))

    for i in range(n_samples):
        X_diff = X[i] - X
        y_diff = y[i] - y

        # Avoid division by zero
        mask = np.abs(X_diff) > 1e-10
        slopes[i, mask] = y_diff[mask] / X_diff[mask]

    median_slopes = np.median(slopes, axis=0)
    intercept = np.median(y - X @ median_slopes)

    return median_slopes, intercept

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   metric: str = 'mse',
                   custom_metric: Optional[Callable] = None) -> Dict[str, float]:
    """
    Compute regression metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    metric : str or callable
        Metric to compute ('mse', 'mae', 'r2')
    custom_metric : callable, optional
        Custom metric function

    Returns
    -------
    dict
        Dictionary of computed metrics
    """
    metrics = {}

    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)
    elif metric == 'mse':
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)

    return metrics

def Theil_Sen_estimator_fit(X: np.ndarray, y: np.ndarray,
                          normalization: str = 'none',
                          metric: str = 'mse',
                          custom_norm: Optional[Callable] = None,
                          custom_metric: Optional[Callable] = None) -> Dict:
    """
    Theil-Sen estimator for robust linear regression.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalization : str or callable, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, optional
        Metric to compute ('mse', 'mae', 'r2')
    custom_norm : callable, optional
        Custom normalization function
    custom_metric : callable, optional
        Custom metric function

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': estimated coefficients and intercept
        - 'metrics': computed metrics
        - 'params_used': parameters used in the estimation
        - 'warnings': any warnings generated during computation

    Examples
    --------
    >>> X = np.random.rand(100, 2)
    >>> y = 3*X[:, 0] + 5*X[:, 1] + np.random.normal(0, 0.1, 100)
    >>> result = Theil_Sen_estimator_fit(X, y)
    """
    # Validate inputs
    validate_inputs(X, y)

    warnings = []

    # Normalize data
    X_norm, y_norm = normalize_data(X, y, normalization, custom_norm)

    # Compute median slopes and intercept
    coefficients, intercept = compute_median_slopes(X_norm, y_norm)

    # Compute predictions
    y_pred = X_norm @ coefficients + intercept

    # Compute metrics
    metrics = compute_metrics(y_norm, y_pred, metric, custom_metric)

    # Denormalize coefficients if needed
    if normalization != 'none' and custom_norm is None:
        if normalization == 'standard':
            coefficients /= np.std(X, axis=0)
            intercept = (intercept * np.std(y) + np.mean(y)) - (np.mean(X) @ coefficients)
        elif normalization == 'minmax':
            coefficients /= (np.max(X, axis=0) - np.min(X, axis=0))
            intercept = (intercept * (np.max(y) - np.min(y)) + np.min(y)) - ((np.min(X) @ coefficients))
        elif normalization == 'robust':
            q75 = np.percentile(X, 75, axis=0)
            q25 = np.percentile(X, 25, axis=0)
            iqr = q75 - q25
            coefficients /= iqr
            y_iqr = np.percentile(y, 75) - np.percentile(y, 25)
            intercept = (intercept * y_iqr + np.median(y)) - (np.median(X) @ coefficients)

    # Prepare result dictionary
    result = {
        'result': {
            'coefficients': coefficients,
            'intercept': intercept
        },
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric
        },
        'warnings': warnings
    }

    return result

################################################################################
# Least_Trimmed_Squares
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
                  custom_normalization: Optional[Callable] = None) -> tuple:
    """Normalize data based on specified method."""
    if custom_normalization is not None:
        X_norm, y_norm = custom_normalization(X, y)
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
        X_norm, y_norm = X, y
    return X_norm, y_norm

def compute_residuals(X: np.ndarray, y: np.ndarray,
                     weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute residuals for weighted least squares."""
    if weights is not None:
        return y - X @ np.linalg.inv(X.T @ np.diag(weights) @ X) @ (X.T @ np.diag(weights) @ y)
    else:
        return y - X @ np.linalg.inv(X.T @ X) @ (X.T @ y)

def least_trimmed_squares_criterion(residuals: np.ndarray, h: int) -> float:
    """Compute the least trimmed squares criterion."""
    abs_residuals = np.abs(residuals)
    sorted_indices = np.argsort(abs_residuals)
    return np.sum(abs_residuals[sorted_indices[:h]])

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   metric: str = 'mse',
                   custom_metric: Optional[Callable] = None) -> Dict[str, float]:
    """Compute regression metrics."""
    if custom_metric is not None:
        return {'custom': custom_metric(y_true, y_pred)}
    metrics = {}
    if metric == 'mse' or 'all' in metric:
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    if metric == 'mae' or 'all' in metric:
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    if metric == 'r2' or 'all' in metric:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    return metrics

def least_trimmed_squares_fit(X: np.ndarray, y: np.ndarray,
                            h: int = None,
                            normalization: str = 'standard',
                            metric: str = 'mse',
                            solver: str = 'closed_form',
                            max_iter: int = 100,
                            tol: float = 1e-4,
                            random_state: Optional[int] = None,
                            custom_normalization: Optional[Callable] = None,
                            custom_metric: Optional[Callable] = None) -> Dict:
    """
    Least Trimmed Squares regression.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    h : int, optional
        Number of points to consider for trimming (default: n_samples // 2)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or list, optional
        Metric(s) to compute ('mse', 'mae', 'r2', 'all')
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent')
    max_iter : int, optional
        Maximum number of iterations (default: 100)
    tol : float, optional
        Tolerance for convergence (default: 1e-4)
    random_state : int, optional
        Random seed for reproducibility
    custom_normalization : callable, optional
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
    >>> result = least_trimmed_squares_fit(X, y)
    """
    # Validate inputs
    validate_inputs(X, y)

    # Set default h if not provided
    if h is None:
        h = X.shape[0] // 2

    # Normalize data
    X_norm, y_norm = normalize_data(X, y, normalization, custom_normalization)

    # Initialize parameters
    n_samples, n_features = X_norm.shape
    beta = np.zeros(n_features)

    # Main LTS algorithm
    for _ in range(max_iter):
        residuals = compute_residuals(X_norm, y_norm)
        criterion_value = least_trimmed_squares_criterion(residuals, h)

        # Update weights based on residuals
        abs_res = np.abs(residuals)
        sorted_indices = np.argsort(abs_res)
        weights = np.zeros(n_samples)
        weights[sorted_indices[:h]] = 1

        # Solve weighted least squares
        if solver == 'closed_form':
            X_weighted = np.diag(weights) @ X_norm
            beta_new = np.linalg.inv(X_weighted.T @ X_weighted) @ (X_weighted.T @ y_norm)
        else:
            # Gradient descent implementation would go here
            raise NotImplementedError("Gradient descent solver not implemented yet")

        # Check convergence
        if np.linalg.norm(beta_new - beta) < tol:
            break

        beta = beta_new

    # Compute predictions and metrics
    y_pred = X_norm @ beta
    metrics = compute_metrics(y_norm, y_pred, metric, custom_metric)

    # Denormalize coefficients if needed
    if normalization != 'none':
        if custom_normalization is None:
            if normalization == 'standard':
                beta = beta * np.std(y) / np.std(X, axis=0)
            elif normalization == 'minmax':
                beta = beta * (np.max(y) - np.min(y)) / (np.max(X, axis=0) - np.min(X, axis=0))
            elif normalization == 'robust':
                beta = beta * (np.percentile(y, 75) - np.percentile(y, 25)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))

    return {
        'result': beta,
        'metrics': metrics,
        'params_used': {
            'h': h,
            'normalization': normalization,
            'metric': metric,
            'solver': solver,
            'max_iter': max_iter,
            'tol': tol
        },
        'warnings': []
    }

################################################################################
# MM_estimators
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate input arrays for MM estimators.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)

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
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def normalize_data(X: np.ndarray, y: np.ndarray,
                  normalization: str = 'standard',
                  custom_normalize: Optional[Callable] = None) -> tuple:
    """
    Normalize input data according to specified method.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalization : str or callable
        Normalization method ('none', 'standard', 'minmax', 'robust')
    custom_normalize : callable, optional
        Custom normalization function

    Returns
    -------
    tuple
        Normalized X and y, and the normalization parameters
    """
    if custom_normalize is not None:
        X_norm, y_norm = custom_normalize(X, y)
        return X_norm, y_norm, None

    if normalization == 'none':
        return X.copy(), y.copy(), None
    elif normalization == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_norm = (X - X_mean) / X_std
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_norm = (y - y_mean) / y_std
        return X_norm, y_norm, {'X_mean': X_mean, 'X_std': X_std,
                               'y_mean': y_mean, 'y_std': y_std}
    elif normalization == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
        y_min = np.min(y)
        y_max = np.max(y)
        y_norm = (y - y_min) / (y_max - y_min + 1e-8)
        return X_norm, y_norm, {'X_min': X_min, 'X_max': X_max,
                               'y_min': y_min, 'y_max': y_max}
    elif normalization == 'robust':
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - X_median) / (X_iqr + 1e-8)
        y_median = np.median(y)
        y_iqr = np.subtract(*np.percentile(y, [75, 25]))
        y_norm = (y - y_median) / (y_iqr + 1e-8)
        return X_norm, y_norm, {'X_median': X_median, 'X_iqr': X_iqr,
                               'y_median': y_median, 'y_iqr': y_iqr}
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def compute_residuals(X: np.ndarray, y: np.ndarray,
                     beta: np.ndarray) -> np.ndarray:
    """
    Compute residuals for given parameters.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    beta : np.ndarray
        Model parameters of shape (n_features,)

    Returns
    -------
    np.ndarray
        Residuals of shape (n_samples,)
    """
    return y - X @ beta

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   metric: str = 'mse',
                   custom_metric: Optional[Callable] = None) -> Dict[str, float]:
    """
    Compute regression metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True target values of shape (n_samples,)
    y_pred : np.ndarray
        Predicted values of shape (n_samples,)
    metric : str or callable
        Metric to compute ('mse', 'mae', 'r2')
    custom_metric : callable, optional
        Custom metric function

    Returns
    -------
    dict
        Dictionary of computed metrics
    """
    if custom_metric is not None:
        return {'custom': custom_metric(y_true, y_pred)}

    metrics = {}
    if metric in ['mse', 'all']:
        mse = np.mean((y_true - y_pred) ** 2)
        metrics['mse'] = mse
    if metric in ['mae', 'all']:
        mae = np.mean(np.abs(y_true - y_pred))
        metrics['mae'] = mae
    if metric in ['r2', 'all']:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        metrics['r2'] = r2

    return metrics

def mm_estimator_closed_form(X: np.ndarray, y: np.ndarray,
                           psi_func: Callable = lambda x: x,
                           rho_func: Callable = lambda x: x**2) -> np.ndarray:
    """
    MM estimator using closed-form solution.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    psi_func : callable
        Influence function for M-estimator
    rho_func : callable
        Loss function for MM-estimator

    Returns
    -------
    np.ndarray
        Estimated parameters of shape (n_features,)
    """
    # Initial robust fit using M-estimator
    beta_init = np.linalg.inv(X.T @ X) @ X.T @ y

    # Iteratively reweighted least squares
    max_iter = 100
    tol = 1e-6
    beta = beta_init.copy()
    for _ in range(max_iter):
        residuals = compute_residuals(X, y, beta)
        weights = 1 / (rho_func(residuals) + 1e-8)
        W = np.diag(weights)
        beta_new = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
        if np.linalg.norm(beta_new - beta, ord=np.inf) < tol:
            break
        beta = beta_new

    return beta

def mm_estimator_gradient_descent(X: np.ndarray, y: np.ndarray,
                                psi_func: Callable = lambda x: x,
                                rho_func: Callable = lambda x: x**2,
                                learning_rate: float = 0.01,
                                max_iter: int = 1000) -> np.ndarray:
    """
    MM estimator using gradient descent.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    psi_func : callable
        Influence function for M-estimator
    rho_func : callable
        Loss function for MM-estimator
    learning_rate : float
        Learning rate for gradient descent
    max_iter : int
        Maximum number of iterations

    Returns
    -------
    np.ndarray
        Estimated parameters of shape (n_features,)
    """
    n_samples, n_features = X.shape
    beta = np.zeros(n_features)

    for _ in range(max_iter):
        residuals = compute_residuals(X, y, beta)
        gradient = -2 * X.T @ (psi_func(residuals) / rho_func(residuals))
        beta -= learning_rate * gradient

    return beta

def MM_estimators_fit(X: np.ndarray, y: np.ndarray,
                     solver: str = 'closed_form',
                     normalization: str = 'standard',
                     metric: str = 'mse',
                     psi_func: Optional[Callable] = None,
                     rho_func: Optional[Callable] = None,
                     custom_normalize: Optional[Callable] = None,
                     custom_metric: Optional[Callable] = None) -> Dict:
    """
    Fit MM estimators for robust regression.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    solver : str
        Solver to use ('closed_form', 'gradient_descent')
    normalization : str or callable
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable
        Metric to compute ('mse', 'mae', 'r2')
    psi_func : callable, optional
        Custom influence function for M-estimator
    rho_func : callable, optional
        Custom loss function for MM-estimator
    custom_normalize : callable, optional
        Custom normalization function
    custom_metric : callable, optional
        Custom metric function

    Returns
    -------
    dict
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    validate_inputs(X, y)

    # Default functions if not provided
    psi_func = psi_func or (lambda x: x)
    rho_func = rho_func or (lambda x: x**2)

    # Normalize data
    X_norm, y_norm, norm_params = normalize_data(X, y,
                                               normalization=normalization,
                                               custom_normalize=custom_normalize)

    # Choose solver
    if solver == 'closed_form':
        beta = mm_estimator_closed_form(X_norm, y_norm,
                                      psi_func=psi_func,
                                      rho_func=rho_func)
    elif solver == 'gradient_descent':
        beta = mm_estimator_gradient_descent(X_norm, y_norm,
                                           psi_func=psi_func,
                                           rho_func=rho_func)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute predictions and metrics
    y_pred = X_norm @ beta
    metrics = compute_metrics(y_norm, y_pred,
                            metric=metric,
                            custom_metric=custom_metric)

    # Prepare output
    result = {
        'result': beta,
        'metrics': metrics,
        'params_used': {
            'solver': solver,
            'normalization': normalization,
            'metric': metric
        },
        'warnings': []
    }

    return result

# Example usage:
"""
X = np.random.randn(100, 5)
y = X @ np.array([2.0, -1.5, 0.8, 3.2, -0.7]) + np.random.randn(100) * 0.5

result = MM_estimators_fit(X, y,
                          solver='closed_form',
                          normalization='standard',
                          metric='all')
"""

################################################################################
# S_regression
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def S_regression_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalization: str = 'standard',
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
    Perform robust regression with configurable parameters.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, optional
        Metric to optimize ('mse', 'mae', 'r2', 'logloss') or custom callable
    distance : str, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski')
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics
    custom_distance : callable, optional
        Custom distance function if not using built-in distances

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = S_regression_fit(X, y, normalization='robust', metric='mae')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = _apply_normalization(X, y, normalization)

    # Prepare solver parameters
    solver_params = {
        'tol': tol,
        'max_iter': max_iter,
        'regularization': regularization
    }

    # Select solver
    if solver == 'closed_form':
        coefficients = _solve_closed_form(X_norm, y_norm)
    elif solver == 'gradient_descent':
        coefficients = _solve_gradient_descent(X_norm, y_norm, **solver_params)
    elif solver == 'newton':
        coefficients = _solve_newton(X_norm, y_norm, **solver_params)
    elif solver == 'coordinate_descent':
        coefficients = _solve_coordinate_descent(X_norm, y_norm, **solver_params)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(X_norm, y_norm, coefficients, metric, custom_metric)

    # Prepare result dictionary
    result = {
        'result': coefficients,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric if isinstance(metric, str) else 'custom',
            'distance': distance,
            'solver': solver,
            'regularization': regularization
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

def _apply_normalization(
    X: np.ndarray,
    y: np.ndarray,
    method: str
) -> Tuple[np.ndarray, np.ndarray]:
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

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve regression using closed form solution."""
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000,
    regularization: Optional[str] = None
) -> np.ndarray:
    """Solve regression using gradient descent."""
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradient = X.T @ (X @ coefficients - y) / len(y)
        if regularization == 'l1':
            gradient += np.sign(coefficients)  # L1 penalty
        elif regularization == 'l2':
            gradient += coefficients  # L2 penalty

        new_coefficients = coefficients - learning_rate * gradient
        if np.linalg.norm(new_coefficients - coefficients) < tol:
            break
        coefficients = new_coefficients

    return coefficients

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000,
    regularization: Optional[str] = None
) -> np.ndarray:
    """Solve regression using Newton's method."""
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)

    for _ in range(max_iter):
        residuals = X @ coefficients - y
        gradient = X.T @ residuals / len(y)
        hessian = X.T @ X / len(y)

        if regularization == 'l2':
            hessian += np.eye(n_features)  # L2 penalty

        update = np.linalg.solve(hessian, -gradient)
        new_coefficients = coefficients + update

        if np.linalg.norm(update) < tol:
            break
        coefficients = new_coefficients

    return coefficients

def _solve_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000,
    regularization: Optional[str] = None
) -> np.ndarray:
    """Solve regression using coordinate descent."""
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - (X @ coefficients) + coefficients[j] * X_j

            if regularization == 'l1':
                # Soft-thresholding for L1
                rho = np.sum(X_j * residuals) / np.sum(X_j**2)
                coefficients[j] = np.sign(rho) * max(abs(rho) - 1, 0)
            else:
                # Standard OLS for coordinate
                coefficients[j] = np.sum(X_j * residuals) / np.sum(X_j**2)

        if np.linalg.norm(coefficients - _solve_closed_form(X, y)) < tol:
            break

    return coefficients

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Calculate specified metrics."""
    y_pred = X @ coefficients

    if custom_metric is not None:
        return {'custom': custom_metric(y, y_pred)}

    metrics = {}
    if metric == 'mse' or isinstance(metric, str) and 'mse' in metric:
        metrics['mse'] = np.mean((y - y_pred)**2)
    if metric == 'mae' or isinstance(metric, str) and 'mae' in metric:
        metrics['mae'] = np.mean(np.abs(y - y_pred))
    if metric == 'r2' or isinstance(metric, str) and 'r2' in metric:
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        metrics['r2'] = 1 - ss_res / ss_tot
    if metric == 'logloss' or isinstance(metric, str) and 'logloss' in metric:
        metrics['logloss'] = -np.mean(y * np.log(y_pred) + (1-y) * np.log(1-y_pred))

    return metrics

################################################################################
# quantile_regression
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def quantile_regression_fit(
    X: np.ndarray,
    y: np.ndarray,
    tau: float = 0.5,
    solver: str = 'gradient_descent',
    normalization: Optional[str] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mae',
    max_iter: int = 1000,
    tol: float = 1e-4,
    learning_rate: float = 0.01,
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
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
    tau : float, optional
        Quantile to estimate (default 0.5 for median).
    solver : str, optional
        Solver to use ('gradient_descent', 'coordinate_descent').
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to evaluate ('mae', 'mse') or custom function.
    max_iter : int, optional
        Maximum number of iterations (default 1000).
    tol : float, optional
        Tolerance for stopping criterion (default 1e-4).
    learning_rate : float, optional
        Learning rate for gradient descent (default 0.01).
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet').
    alpha : float, optional
        Regularization strength (default 1.0).
    l1_ratio : float, optional
        ElasticNet mixing parameter (default 0.5).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': Estimated coefficients
        - 'metrics': Evaluation metrics
        - 'params_used': Parameters used in fitting
        - 'warnings': Any warnings encountered

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = quantile_regression_fit(X, y, tau=0.75)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize data
    X_norm, y_norm = _apply_normalization(X, y, normalization)

    # Initialize coefficients
    n_features = X_norm.shape[1]
    coefs = np.zeros(n_features + 1)  # Include intercept

    # Choose solver
    if solver == 'gradient_descent':
        coefs = _gradient_descent_quantile_regression(
            X_norm, y_norm, tau, coefs,
            max_iter=max_iter, tol=tol,
            learning_rate=learning_rate,
            regularization=regularization,
            alpha=alpha, l1_ratio=l1_ratio
        )
    elif solver == 'coordinate_descent':
        coefs = _coordinate_descent_quantile_regression(
            X_norm, y_norm, tau, coefs,
            max_iter=max_iter, tol=tol,
            regularization=regularization,
            alpha=alpha, l1_ratio=l1_ratio
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    y_pred = _predict(X_norm, coefs)
    metrics = _calculate_metrics(y_norm, y_pred, metric)

    # Prepare output
    result = {
        'result': coefs,
        'metrics': metrics,
        'params_used': {
            'tau': tau,
            'solver': solver,
            'normalization': normalization,
            'metric': metric,
            'max_iter': max_iter,
            'tol': tol,
            'learning_rate': learning_rate,
            'regularization': regularization,
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

def _apply_normalization(
    X: np.ndarray,
    y: np.ndarray,
    method: Optional[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Apply normalization to features and target."""
    X_norm = X.copy()
    y_norm = y.copy()

    if method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_norm = (X - mean) / (std + 1e-8)

    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)

    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - median) / (iqr + 1e-8)

    return X_norm, y_norm

def _gradient_descent_quantile_regression(
    X: np.ndarray,
    y: np.ndarray,
    tau: float,
    coefs: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-4,
    learning_rate: float = 0.01,
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    l1_ratio: float = 0.5
) -> np.ndarray:
    """Quantile regression using gradient descent."""
    n_samples, n_features = X.shape
    intercept = coefs[-1]
    features_coefs = coefs[:-1]

    for _ in range(max_iter):
        # Compute residuals
        residuals = y - (X @ features_coefs + intercept)

        # Compute subgradient
        subgrad = -tau * (residuals < 0) + (1 - tau) * (residuals > 0)

        # Add regularization term
        if regularization == 'l1':
            subgrad[:-1] += alpha * np.sign(features_coefs)
        elif regularization == 'l2':
            subgrad[:-1] += 2 * alpha * features_coefs
        elif regularization == 'elasticnet':
            subgrad[:-1] += alpha * (l1_ratio * np.sign(features_coefs) +
                                    (1 - l1_ratio) * 2 * features_coefs)

        # Update coefficients
        gradient = X.T @ subgrad / n_samples
        features_coefs -= learning_rate * gradient[:-1]
        intercept -= learning_rate * gradient[-1]

        # Check convergence
        if np.linalg.norm(gradient) < tol:
            break

    return np.concatenate([features_coefs, [intercept]])

def _coordinate_descent_quantile_regression(
    X: np.ndarray,
    y: np.ndarray,
    tau: float,
    coefs: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-4,
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    l1_ratio: float = 0.5
) -> np.ndarray:
    """Quantile regression using coordinate descent."""
    n_samples, n_features = X.shape
    intercept = coefs[-1]
    features_coefs = coefs[:-1]

    for _ in range(max_iter):
        for j in range(n_features):
            # Compute residuals without current feature
            residuals = y - (X @ features_coefs + intercept)

            # Compute correlation with current feature
            Xj = X[:, j]
            corr = Xj @ residuals

            # Compute regularization term
            reg_term = 0
            if regularization == 'l1':
                reg_term = alpha * l1_ratio * np.sign(features_coefs[j])
            elif regularization == 'l2':
                reg_term = 2 * alpha * (1 - l1_ratio) * features_coefs[j]
            elif regularization == 'elasticnet':
                reg_term = alpha * (l1_ratio * np.sign(features_coefs[j]) +
                                  (1 - l1_ratio) * 2 * features_coefs[j])

            # Update coefficient
            if Xj.var() > 0:
                features_coefs[j] = _soft_threshold(
                    (corr - reg_term) / Xj.var(),
                    alpha * l1_ratio
                )

        # Update intercept
        residuals = y - (X @ features_coefs + intercept)
        intercept += np.median(residuals)

        # Check convergence
        if np.linalg.norm(X @ features_coefs + intercept - y) < tol:
            break

    return np.concatenate([features_coefs, [intercept]])

def _soft_threshold(rho: float, lambda_: float) -> float:
    """Soft thresholding operator."""
    if rho > lambda_:
        return rho - lambda_
    elif rho < -lambda_:
        return rho + lambda_
    else:
        return 0

def _predict(X: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    """Predict using fitted coefficients."""
    return X @ coefs[:-1] + coefs[-1]

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    metrics = {}

    if isinstance(metric, str):
        if metric == 'mae':
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metric == 'mse':
            metrics['mse'] = np.mean((y_true - y_pred) ** 2)
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['r2'] = 1 - ss_res / (ss_tot + 1e-8)
    else:
        metrics['custom'] = metric(y_true, y_pred)

    return metrics

################################################################################
# weighted_least_squares
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(weights, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if weights.ndim != 1:
        raise ValueError("weights must be a 1D array.")
    if X.shape[0] != y.shape[0] or X.shape[0] != weights.shape[0]:
        raise ValueError("X, y, and weights must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)) or np.any(np.isnan(weights)):
        raise ValueError("Inputs must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Inputs must not contain infinite values.")
    if np.any(weights <= 0):
        raise ValueError("Weights must be positive.")

def normalize_data(X: np.ndarray, y: np.ndarray, method: str = 'none') -> tuple:
    """Normalize data based on the specified method."""
    if method == 'none':
        return X, y
    elif method == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1.0  # Avoid division by zero
        X_normalized = (X - X_mean) / X_std
        y_mean = np.mean(y)
        y_std = np.std(y)
        if y_std == 0:
            y_std = 1.0
        y_normalized = (y - y_mean) / y_std
        return X_normalized, y_normalized
    elif method == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1.0  # Avoid division by zero
        X_normalized = (X - X_min) / X_range
        y_min = np.min(y)
        y_max = np.max(y)
        y_range = y_max - y_min
        if y_range == 0:
            y_range = 1.0
        y_normalized = (y - y_min) / y_range
        return X_normalized, y_normalized
    elif method == 'robust':
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_iqr[X_iqr == 0] = 1.0  # Avoid division by zero
        X_normalized = (X - X_median) / X_iqr
        y_median = np.median(y)
        y_iqr = np.subtract(*np.percentile(y, [75, 25]))
        if y_iqr == 0:
            y_iqr = 1.0
        y_normalized = (y - y_median) / y_iqr
        return X_normalized, y_normalized
    else:
        raise ValueError("Invalid normalization method.")

def compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str = 'mse', custom_metric: Optional[Callable] = None) -> float:
    """Compute the specified metric between true and predicted values."""
    if custom_metric is not None:
        return custom_metric(y_true, y_pred)
    if metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    elif metric == 'logloss':
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError("Invalid metric.")

def closed_form_solver(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Solve the weighted least squares problem using closed-form solution."""
    W = np.diag(weights)
    XTWX = X.T @ W @ X
    XTWy = X.T @ W @ y
    return np.linalg.solve(XTWX, XTWy)

def gradient_descent_solver(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                           learning_rate: float = 0.01, max_iter: int = 1000,
                           tol: float = 1e-4) -> np.ndarray:
    """Solve the weighted least squares problem using gradient descent."""
    n_samples, n_features = X.shape
    beta = np.zeros(n_features)
    W = np.diag(weights)

    for _ in range(max_iter):
        gradient = -2 * X.T @ W @ (y - X @ beta)
        beta_new = beta - learning_rate * gradient
        if np.linalg.norm(beta_new - beta) < tol:
            break
        beta = beta_new

    return beta

def weighted_least_squares_fit(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                             normalization: str = 'none',
                             metric: str = 'mse',
                             solver: str = 'closed_form',
                             custom_metric: Optional[Callable] = None,
                             **solver_kwargs) -> Dict:
    """
    Perform weighted least squares regression.

    Parameters:
    - X: Design matrix of shape (n_samples, n_features).
    - y: Target values of shape (n_samples,).
    - weights: Weights for each sample of shape (n_samples,).
    - normalization: Normalization method ('none', 'standard', 'minmax', 'robust').
    - metric: Metric to compute ('mse', 'mae', 'r2', 'logloss') or custom callable.
    - solver: Solver to use ('closed_form', 'gradient_descent').
    - custom_metric: Custom metric function.
    - solver_kwargs: Additional keyword arguments for the solver.

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings.
    """
    validate_inputs(X, y, weights)
    X_norm, y_norm = normalize_data(X, y, normalization)

    if solver == 'closed_form':
        beta = closed_form_solver(X_norm, y_norm, weights)
    elif solver == 'gradient_descent':
        beta = gradient_descent_solver(X_norm, y_norm, weights, **solver_kwargs)
    else:
        raise ValueError("Invalid solver.")

    y_pred = X_norm @ beta
    result_metric = compute_metric(y_norm, y_pred, metric, custom_metric)

    return {
        'result': beta,
        'metrics': {metric: result_metric},
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'solver': solver,
            'solver_kwargs': solver_kwargs
        },
        'warnings': []
    }

################################################################################
# influence_functions
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def influence_functions_fit(
    X: np.ndarray,
    y: np.ndarray,
    model_func: Callable[[np.ndarray, np.ndarray], float],
    influence_metric: str = 'mahalanobis',
    normalization: str = 'standard',
    distance_metric: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Compute influence functions for robust regression.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    model_func : Callable[[np.ndarray, np.ndarray], float]
        Function that computes the model prediction
    influence_metric : str
        Metric for influence calculation ('mahalanobis', 'cook', 'dfbetas')
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    distance_metric : str
        Distance metric for influence calculation ('euclidean', 'manhattan', 'cosine')
    solver : str
        Solver for model fitting ('closed_form', 'gradient_descent', 'newton')
    regularization : Optional[str]
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    custom_metric : Optional[Callable]
        Custom metric function if needed

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], str]]
        Dictionary containing:
        - 'result': Influence values
        - 'metrics': Computed metrics
        - 'params_used': Parameters used
        - 'warnings': Any warnings

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> def linear_model(X, params):
    ...     return X @ params
    ...
    >>> result = influence_functions_fit(X, y, linear_model)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if needed
    X_norm, y_norm = _apply_normalization(X, y, normalization)

    # Fit the model
    params = _fit_model(
        X_norm,
        y_norm,
        model_func,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter
    )

    # Compute influence functions
    influences = _compute_influence(
        X_norm,
        y_norm,
        params,
        model_func,
        metric=influence_metric,
        distance_metric=distance_metric,
        custom_metric=custom_metric
    )

    # Compute metrics
    metrics = _compute_metrics(X_norm, y_norm, params, model_func)

    return {
        'result': influences,
        'metrics': metrics,
        'params_used': {
            'influence_metric': influence_metric,
            'normalization': normalization,
            'distance_metric': distance_metric,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': _check_warnings(X, y)
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
    y: np.ndarray,
    method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply normalization to data."""
    if method == 'standard':
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y_norm = (y - np.mean(y)) / np.std(y)
    elif method == 'minmax':
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif method == 'robust':
        X_norm = (X - np.median(X, axis=0)) / _mad(X, axis=0)
        y_norm = (y - np.median(y)) / _mad(y)
    else:
        X_norm, y_norm = X.copy(), y.copy()
    return X_norm, y_norm

def _mad(data: np.ndarray, axis: int = 0) -> float:
    """Median Absolute Deviation."""
    median = np.median(data, axis=axis)
    return np.median(np.abs(data - median), axis=axis)

def _fit_model(
    X: np.ndarray,
    y: np.ndarray,
    model_func: Callable[[np.ndarray, np.ndarray], float],
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Fit the model using specified solver."""
    if solver == 'closed_form':
        params = _closed_form_solution(X, y)
    elif solver == 'gradient_descent':
        params = _gradient_descent(
            X, y, model_func,
            tol=tol,
            max_iter=max_iter
        )
    elif solver == 'newton':
        params = _newton_method(
            X, y, model_func,
            tol=tol,
            max_iter=max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    if regularization is not None:
        params = _apply_regularization(params, X, y, model_func, regularization)

    return params

def _closed_form_solution(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed form solution for linear regression."""
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    model_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Gradient descent optimization."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        predictions = model_func(X, params)
        gradient = X.T @ (predictions - y) / len(y)
        new_params = params - learning_rate * gradient

        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params

    return params

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    model_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Newton's method optimization."""
    n_features = X.shape[1]
    params = np.zeros(n_features)

    for _ in range(max_iter):
        predictions = model_func(X, params)
        gradient = X.T @ (predictions - y) / len(y)
        hessian = X.T @ X / len(y)

        try:
            new_params = params - np.linalg.pinv(hessian) @ gradient
        except np.linalg.LinAlgError:
            break

        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params

    return params

def _apply_regularization(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    model_func: Callable[[np.ndarray, np.ndarray], float],
    method: str
) -> np.ndarray:
    """Apply regularization to parameters."""
    if method == 'l1':
        params = _l1_regularization(params, X, y)
    elif method == 'l2':
        params = _l2_regularization(params, X, y)
    elif method == 'elasticnet':
        params = _elasticnet_regularization(params, X, y)
    return params

def _l1_regularization(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """L1 regularization (LASSO)."""
    alpha = 0.1
    return _gradient_descent(X, y - X @ params + alpha * np.sign(params), lambda x, p: x @ p)

def _l2_regularization(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """L2 regularization (Ridge)."""
    alpha = 0.1
    return _closed_form_solution(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)

def _elasticnet_regularization(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """Elastic Net regularization."""
    alpha = 0.1
    l1_ratio = 0.5
    return _gradient_descent(
        X, y,
        lambda x, p: x @ p + alpha * (l1_ratio * np.sum(np.abs(p)) + (1 - l1_ratio) * np.sum(p**2))
    )

def _compute_influence(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    model_func: Callable[[np.ndarray, np.ndarray], float],
    metric: str = 'mahalanobis',
    distance_metric: str = 'euclidean',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> np.ndarray:
    """Compute influence functions."""
    predictions = model_func(X, params)
    residuals = y - predictions

    if custom_metric is not None:
        return custom_metric(X, residuals)

    if metric == 'mahalanobis':
        return _mahalanobis_influence(X, residuals)
    elif metric == 'cook':
        return _cook_distance(X, residuals, params)
    elif metric == 'dfbetas':
        return _dfbetas(X, residuals)
    else:
        raise ValueError(f"Unknown influence metric: {metric}")

def _mahalanobis_influence(X: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """Mahalanobis distance influence."""
    cov = np.cov(X, rowvar=False)
    try:
        inv_cov = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.eye(X.shape[1])

    return np.sqrt(np.sum((X @ inv_cov) * X, axis=1)) * np.abs(residuals)

def _cook_distance(
    X: np.ndarray,
    residuals: np.ndarray,
    params: np.ndarray
) -> np.ndarray:
    """Cook's distance influence."""
    h = X @ np.linalg.pinv(X.T @ X) @ X.T
    leverage = np.diag(h)
    return (residuals**2) * leverage / (1 - leverage)

def _dfbetas(X: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """DFBETAS influence."""
    n = X.shape[0]
    return np.abs(residuals) * np.sqrt(n / (n - 1))

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    model_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Compute regression metrics."""
    predictions = model_func(X, params)
    residuals = y - predictions

    return {
        'mse': np.mean(residuals**2),
        'mae': np.mean(np.abs(residuals)),
        'r2': 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)
    }

def _check_warnings(X: np.ndarray, y: np.ndarray) -> list[str]:
    """Check for potential warnings."""
    warnings = []

    if X.shape[1] > X.shape[0]:
        warnings.append("Warning: Number of features exceeds number of samples")

    if np.any(np.std(X, axis=0) == 0):
        warnings.append("Warning: Zero-variance features detected")

    return warnings

################################################################################
# breakdown_point
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def breakdown_point_fit(
    X: np.ndarray,
    y: np.ndarray,
    model_func: Callable[[np.ndarray], np.ndarray],
    contamination: float = 0.5,
    metric_func: Callable[[np.ndarray, np.ndarray], float] = None,
    distance_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
    solver: str = 'closed_form',
    normalization: str = 'none',
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]:
    """
    Compute the breakdown point for a regression model.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    model_func : Callable[[np.ndarray], np.ndarray]
        Function that fits the model and returns predictions.
    contamination : float, optional
        Proportion of contamination (default is 0.5).
    metric_func : Callable[[np.ndarray, np.ndarray], float], optional
        Custom metric function (default is None).
    distance_func : Callable[[np.ndarray, np.ndarray], np.ndarray], optional
        Custom distance function (default is None).
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', etc.) (default is 'closed_form').
    normalization : str, optional
        Normalization method ('none', 'standard', etc.) (default is 'none').
    max_iter : int, optional
        Maximum number of iterations (default is 100).
    tol : float, optional
        Tolerance for convergence (default is 1e-4).
    random_state : Optional[int], optional
        Random seed for reproducibility (default is None).

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]
        Dictionary containing the breakdown point result and related information.

    Examples
    --------
    >>> X = np.random.rand(100, 2)
    >>> y = np.random.rand(100)
    >>> def model_func(X): return X @ np.linalg.pinv(X) @ y
    >>> result = breakdown_point_fit(X, y, model_func)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_norm, y_norm = _normalize_data(X, y, normalization)

    # Initialize parameters
    params_used = {
        'contamination': contamination,
        'solver': solver,
        'normalization': normalization
    }

    # Compute breakdown point
    bp = _compute_breakdown_point(
        X_norm, y_norm,
        model_func,
        contamination,
        metric_func,
        distance_func,
        solver,
        max_iter,
        tol,
        random_state
    )

    # Calculate metrics
    metrics = _calculate_metrics(X_norm, y_norm, model_func)

    # Prepare output
    result = {
        'result': bp,
        'metrics': metrics,
        'params_used': params_used,
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
    method: str = 'none'
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize data based on specified method."""
    if method == 'standard':
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y_norm = (y - np.mean(y)) / np.std(y)
    elif method == 'minmax':
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif method == 'robust':
        X_norm = (X - np.median(X, axis=0)) / np.median(np.abs(X - np.median(X, axis=0)), axis=0)
        y_norm = (y - np.median(y)) / np.median(np.abs(y - np.median(y)))
    else:
        X_norm, y_norm = X.copy(), y.copy()
    return X_norm, y_norm

def _compute_breakdown_point(
    X: np.ndarray,
    y: np.ndarray,
    model_func: Callable[[np.ndarray], np.ndarray],
    contamination: float,
    metric_func: Optional[Callable[[np.ndarray, np.ndarray], float]],
    distance_func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]],
    solver: str,
    max_iter: int,
    tol: float,
    random_state: Optional[int]
) -> float:
    """Compute the breakdown point for the given model."""
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    epsilon = int(contamination * n_samples)
    bp = 0.0

    for _ in range(max_iter):
        # Randomly select epsilon samples to flip
        idx = np.random.choice(n_samples, size=epsilon, replace=False)
        y_flipped = y.copy()
        y_flipped[idx] *= -1  # Simple flip for demonstration

        # Fit model on flipped data
        y_pred = model_func(X)

        # Calculate metric (default to MSE if not provided)
        if metric_func is None:
            metric = np.mean((y_flipped - y_pred) ** 2)
        else:
            metric = metric_func(y_flipped, y_pred)

        # Check convergence
        if abs(metric) < tol:
            bp = contamination
            break

    return bp

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    model_func: Callable[[np.ndarray], np.ndarray]
) -> Dict[str, float]:
    """Calculate various metrics for the model."""
    y_pred = model_func(X)
    mse = np.mean((y - y_pred) ** 2)
    mae = np.mean(np.abs(y - y_pred))
    r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

    return {
        'mse': mse,
        'mae': mae,
        'r2': r2
    }

################################################################################
# efficiency
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def efficiency_fit(
    X: np.ndarray,
    y: np.ndarray,
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
    Compute efficiency metrics for robust regression.

    Parameters:
    -----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalization : str, optional
        Normalization method ("none", "standard", "minmax", "robust")
    metric : str or callable, optional
        Metric to compute ("mse", "mae", "r2", "logloss") or custom callable
    distance : str, optional
        Distance metric ("euclidean", "manhattan", "cosine", "minkowski")
    solver : str, optional
        Solver method ("closed_form", "gradient_descent", "newton", "coordinate_descent")
    regularization : str, optional
        Regularization type ("none", "l1", "l2", "elasticnet")
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics
    custom_distance : callable, optional
        Custom distance function if not using built-in distances

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = efficiency_fit(X, y, normalization="robust", metric="mae")
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = _apply_normalization(X, y, normalization)

    # Initialize parameters
    params = _initialize_parameters(X_norm.shape[1])

    # Choose solver and optimize
    if solver == "closed_form":
        params = _solve_closed_form(X_norm, y_norm)
    elif solver == "gradient_descent":
        params = _solve_gradient_descent(
            X_norm, y_norm,
            metric=metric,
            distance=distance,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter
        )
    elif solver == "newton":
        params = _solve_newton(
            X_norm, y_norm,
            metric=metric,
            distance=distance,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter
        )
    elif solver == "coordinate_descent":
        params = _solve_coordinate_descent(
            X_norm, y_norm,
            metric=metric,
            distance=distance,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter
        )

    # Compute predictions and metrics
    y_pred = X_norm @ params
    metrics = _compute_metrics(y_norm, y_pred, metric, custom_metric)

    # Prepare output
    result = {
        "result": params,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularization": regularization
        },
        "warnings": _check_warnings(y_norm, y_pred)
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

def _apply_normalization(
    X: np.ndarray,
    y: np.ndarray,
    method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply specified normalization to data."""
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

def _initialize_parameters(n_features: int) -> np.ndarray:
    """Initialize parameters with zeros."""
    return np.zeros(n_features)

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve regression using closed form solution."""
    return np.linalg.pinv(X) @ y

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    metric: str = "mse",
    distance: str = "euclidean",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Solve regression using gradient descent."""
    params = _initialize_parameters(X.shape[1])
    prev_loss = float('inf')

    for _ in range(max_iter):
        grad = _compute_gradient(X, y, params, metric)
        if regularization == "l1":
            grad += np.sign(params)  # L1 penalty
        elif regularization == "l2":
            grad += 2 * params  # L2 penalty
        elif regularization == "elasticnet":
            grad += np.sign(params) + 2 * params  # Elastic net penalty

        params -= 0.01 * grad
        current_loss = _compute_metric(y, X @ params, metric)

        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return params

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    *,
    metric: str = "mse",
    distance: str = "euclidean",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Solve regression using Newton's method."""
    params = _initialize_parameters(X.shape[1])
    prev_loss = float('inf')

    for _ in range(max_iter):
        grad = _compute_gradient(X, y, params, metric)
        hessian = _compute_hessian(X, y, params, metric)

        if regularization == "l1":
            hessian += np.eye(X.shape[1])  # L1 penalty
        elif regularization == "l2":
            hessian += 2 * np.eye(X.shape[1])  # L2 penalty

        params -= np.linalg.pinv(hessian) @ grad
        current_loss = _compute_metric(y, X @ params, metric)

        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return params

def _solve_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    metric: str = "mse",
    distance: str = "euclidean",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Solve regression using coordinate descent."""
    params = _initialize_parameters(X.shape[1])
    prev_loss = float('inf')

    for _ in range(max_iter):
        for i in range(X.shape[1]):
            X_i = X[:, i]
            residual = y - (X @ params) + params[i] * X_i

            if regularization == "l1":
                params[i] = np.sign(np.sum(X_i * residual)) * np.maximum(
                    0, abs(np.sum(X_i * residual)) - 1
                )
            elif regularization == "l2":
                params[i] = np.sum(X_i * residual) / (np.sum(X_i**2) + 1)
            else:
                params[i] = np.sum(X_i * residual) / np.sum(X_i**2)

        current_loss = _compute_metric(y, X @ params, metric)
        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return params

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric: str
) -> np.ndarray:
    """Compute gradient based on specified metric."""
    y_pred = X @ params
    if metric == "mse":
        return -2 * X.T @ (y - y_pred)
    elif metric == "mae":
        return -X.T @ np.sign(y - y_pred)
    elif metric == "r2":
        return 2 * X.T @ (y_pred - y)
    else:
        raise ValueError(f"Unknown metric for gradient: {metric}")

def _compute_hessian(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric: str
) -> np.ndarray:
    """Compute Hessian based on specified metric."""
    if metric == "mse":
        return 2 * X.T @ X
    else:
        raise ValueError(f"Hessian not available for metric: {metric}")

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable] = None
) -> float:
    """Compute specified metric."""
    if custom_metric is not None:
        return custom_metric(y_true, y_pred)

    if metric == "mse":
        return np.mean((y_true - y_pred)**2)
    elif metric == "mae":
        return np.mean(np.abs(y_true - y_pred))
    elif metric == "r2":
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - ss_res / ss_tot
    elif metric == "logloss":
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _check_warnings(y_true: np.ndarray, y_pred: np.ndarray) -> list:
    """Check for potential warnings."""
    warnings = []
    if np.any(np.isnan(y_pred)):
        warnings.append("Predictions contain NaN values")
    if np.any(np.isinf(y_pred)):
        warnings.append("Predictions contain infinite values")
    return warnings

################################################################################
# bias_variance_tradeoff
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def bias_variance_tradeoff_fit(
    X: np.ndarray,
    y: np.ndarray,
    model_func: Callable[[np.ndarray], np.ndarray],
    metric_func: Callable[[np.ndarray, np.ndarray], float] = None,
    n_splits: int = 5,
    random_state: Optional[int] = None,
    normalize: bool = True
) -> Dict[str, Union[Dict, float]]:
    """
    Compute the bias-variance tradeoff for a given model.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    model_func : Callable[[np.ndarray], np.ndarray]
        A function that takes X and returns predicted y.
    metric_func : Callable[[np.ndarray, np.ndarray], float], optional
        A function to compute the metric between true and predicted values.
        Default is mean squared error if None.
    n_splits : int, optional
        Number of splits for cross-validation. Default is 5.
    random_state : Optional[int], optional
        Random seed for reproducibility. Default is None.
    normalize : bool, optional
        Whether to normalize the features. Default is True.

    Returns
    -------
    Dict[str, Union[Dict, float]]
        A dictionary containing:
        - "result": Dictionary with bias and variance.
        - "metrics": Dictionary of computed metrics.
        - "params_used": Dictionary of parameters used.
        - "warnings": List of warnings encountered.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> def simple_model(X):
    ...     return np.mean(X, axis=1)
    >>> result = bias_variance_tradeoff_fit(X, y, simple_model)
    """
    # Validate inputs
    _validate_inputs(X, y)

    if random_state is not None:
        np.random.seed(random_state)

    # Normalize features if required
    X_norm = _normalize_features(X) if normalize else X

    # Initialize results dictionary
    results = {
        "result": {"bias": 0.0, "variance": 0.0},
        "metrics": {},
        "params_used": {
            "n_splits": n_splits,
            "random_state": random_state,
            "normalize": normalize
        },
        "warnings": []
    }

    # Default metric function if none provided
    if metric_func is None:
        metric_func = _mean_squared_error

    # Perform cross-validation
    cv_indices = _generate_cv_indices(X_norm.shape[0], n_splits, random_state)

    for train_idx, test_idx in cv_indices:
        X_train, X_test = X_norm[train_idx], X_norm[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit model on training data
        y_pred_train = model_func(X_train)
        y_pred_test = model_func(X_test)

        # Compute bias and variance
        results["result"]["bias"] += metric_func(y_test, y_pred_test)
        results["result"]["variance"] += np.mean((y_pred_train - np.mean(y_pred_train, axis=0))**2)

    # Average results
    results["result"]["bias"] /= n_splits
    results["result"]["variance"] /= n_splits

    # Compute additional metrics if needed
    results["metrics"]["mse"] = _mean_squared_error(y, model_func(X_norm))
    results["metrics"]["mae"] = _mean_absolute_error(y, model_func(X_norm))

    return results

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

def _normalize_features(X: np.ndarray) -> np.ndarray:
    """Normalize features using standardization."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / (std + 1e-8)

def _generate_cv_indices(n_samples: int, n_splits: int, random_state: Optional[int] = None) -> list:
    """Generate cross-validation indices."""
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    fold_size = n_samples // n_splits
    return [
        (indices[i*fold_size:(i+1)*fold_size], np.delete(indices, range(i*fold_size, (i+1)*fold_size)))
        for i in range(n_splits)
    ]

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error."""
    return np.mean((y_true - y_pred)**2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

################################################################################
# robustness_metrics
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def robustness_metrics_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric_func: Callable[[np.ndarray, np.ndarray], float] = lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
    distance_func: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.linalg.norm(x - y),
    solver: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda X, y: np.linalg.pinv(X.T @ X) @ X.T @ y,
    regularization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_weights: Optional[np.ndarray] = None
) -> Dict[str, Union[Dict, float]]:
    """
    Compute robustness metrics for a regression model.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the input data. If None, no normalization is applied.
    metric_func : Callable[[np.ndarray, np.ndarray], float], optional
        Function to compute the metric between true and predicted values.
    distance_func : Callable[[np.ndarray, np.ndarray], float], optional
        Function to compute the distance between two vectors.
    solver : Callable[[np.ndarray, np.ndarray], np.ndarray], optional
        Function to solve the regression problem and return coefficients.
    regularization : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to apply regularization. If None, no regularization is applied.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_weights : Optional[np.ndarray], optional
        Custom weights for the samples. If None, uniform weights are used.

    Returns
    -------
    Dict[str, Union[Dict, float]]
        Dictionary containing the results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = robustness_metrics_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if a normalizer is provided
    X_normalized = _apply_normalization(X, normalizer)

    # Apply custom weights if provided
    sample_weights = _get_sample_weights(y.shape[0], custom_weights)

    # Solve the regression problem
    coefficients = _solve_regression(X_normalized, y, solver, regularization, tol, max_iter)

    # Predict the target values
    y_pred = X_normalized @ coefficients

    # Compute metrics
    metrics = _compute_metrics(y, y_pred, metric_func)

    # Compute robustness diagnostics
    diagnostics = _compute_robustness_diagnostics(X_normalized, y, y_pred, distance_func)

    return {
        "result": {
            "coefficients": coefficients,
            "y_pred": y_pred
        },
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric_func": metric_func.__name__,
            "distance_func": distance_func.__name__,
            "solver": solver.__name__,
            "regularization": regularization.__name__ if regularization else None,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": _check_warnings(y, y_pred)
    }

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

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    """Apply normalization to the input data."""
    if normalizer is not None:
        return normalizer(X)
    return X

def _get_sample_weights(n_samples: int, custom_weights: Optional[np.ndarray]) -> np.ndarray:
    """Get sample weights."""
    if custom_weights is not None:
        if len(custom_weights) != n_samples:
            raise ValueError("custom_weights must have the same length as y.")
        return custom_weights
    return np.ones(n_samples)

def _solve_regression(
    X: np.ndarray,
    y: np.ndarray,
    solver: Callable[[np.ndarray, np.ndarray], np.ndarray],
    regularization: Optional[Callable[[np.ndarray], np.ndarray]],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve the regression problem."""
    coefficients = solver(X, y)
    if regularization is not None:
        coefficients = regularization(coefficients)
    return coefficients

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Compute the metrics."""
    return {
        "metric": metric_func(y_true, y_pred)
    }

def _compute_robustness_diagnostics(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Compute robustness diagnostics."""
    residuals = y_true - y_pred
    influence = np.abs(residuals) / (np.std(residuals) + 1e-8)
    return {
        "residual_std": np.std(residuals),
        "max_influence": np.max(influence)
    }

def _check_warnings(y_true: np.ndarray, y_pred: np.ndarray) -> List[str]:
    """Check for warnings."""
    warnings = []
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        warnings.append("Predicted values contain NaN or infinite values.")
    return warnings
