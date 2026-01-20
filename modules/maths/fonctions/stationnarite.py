"""
Quantix – Module stationnarite
Généré automatiquement
Date: 2026-01-09
"""

################################################################################
# stationnarite_stricte
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def stationnarite_stricte_fit(
    data: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_params: Optional[Dict] = None
) -> Dict:
    """
    Fit a strict stationarity model to the input data.

    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features).
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Normalization function to apply to the data. Default is None.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Metric to evaluate the stationarity. Can be 'mse', 'mae', 'r2', or a custom callable.
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Distance metric for stationarity evaluation. Can be 'euclidean', 'manhattan', 'cosine',
        'minkowski', or a custom callable.
    solver : str, optional
        Solver to use for optimization. Can be 'closed_form', 'gradient_descent', 'newton',
        or 'coordinate_descent'.
    tol : float, optional
        Tolerance for convergence. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    custom_params : Optional[Dict], optional
        Additional parameters for the solver. Default is None.

    Returns
    -------
    Dict
        A dictionary containing:
        - "result": The stationarity result.
        - "metrics": Computed metrics.
        - "params_used": Parameters used in the computation.
        - "warnings": Any warnings generated during computation.

    Examples
    --------
    >>> data = np.random.randn(100, 5)
    >>> result = stationnarite_stricte_fit(data)
    """
    # Validate inputs
    _validate_inputs(data, normalizer, metric, distance)

    # Normalize data if a normalizer is provided
    normalized_data = _apply_normalization(data, normalizer)

    # Prepare solver parameters
    solver_params = custom_params if custom_params is not None else {}

    # Fit the model based on the chosen solver
    result, metrics = _fit_model(
        normalized_data,
        metric=metric,
        distance=distance,
        solver=solver,
        tol=tol,
        max_iter=max_iter,
        **solver_params
    )

    # Prepare the output dictionary
    output = {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric": metric if isinstance(metric, str) else metric.__name__,
            "distance": distance if isinstance(distance, str) else distance.__name__,
            "solver": solver,
            "tol": tol,
            "max_iter": max_iter,
            **solver_params
        },
        "warnings": []
    }

    return output

def _validate_inputs(
    data: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]],
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> None:
    """
    Validate the input data and parameters.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]]
        Normalization function.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Metric function.
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Distance function.

    Raises
    ------
    ValueError
        If the input data or parameters are invalid.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values.")
    if normalizer is not None and not callable(normalizer):
        raise ValueError("Normalizer must be a callable or None.")
    if isinstance(metric, str) and metric not in ['mse', 'mae', 'r2']:
        raise ValueError("Metric must be 'mse', 'mae', 'r2', or a custom callable.")
    if isinstance(distance, str) and distance not in ['euclidean', 'manhattan', 'cosine', 'minkowski']:
        raise ValueError("Distance must be 'euclidean', 'manhattan', 'cosine', 'minkowski', or a custom callable.")

def _apply_normalization(
    data: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """
    Apply normalization to the data if a normalizer is provided.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]]
        Normalization function.

    Returns
    -------
    np.ndarray
        Normalized data array.
    """
    if normalizer is not None:
        return normalizer(data)
    return data

def _fit_model(
    data: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    solver: str,
    tol: float,
    max_iter: int,
    **kwargs
) -> tuple:
    """
    Fit the stationarity model using the specified solver.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Metric function.
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Distance function.
    solver : str
        Solver to use.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    tuple
        A tuple containing the result and metrics.
    """
    if solver == 'closed_form':
        return _closed_form_solver(data, metric, distance)
    elif solver == 'gradient_descent':
        return _gradient_descent_solver(data, metric, distance, tol, max_iter, **kwargs)
    elif solver == 'newton':
        return _newton_solver(data, metric, distance, tol, max_iter, **kwargs)
    elif solver == 'coordinate_descent':
        return _coordinate_descent_solver(data, metric, distance, tol, max_iter, **kwargs)
    else:
        raise ValueError("Unknown solver.")

def _closed_form_solver(
    data: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> tuple:
    """
    Closed form solver for stationarity.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Metric function.
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Distance function.

    Returns
    -------
    tuple
        A tuple containing the result and metrics.
    """
    # Implement closed form solution logic here
    result = np.mean(data, axis=0)
    metrics = _compute_metrics(data, result, metric, distance)
    return result, metrics

def _gradient_descent_solver(
    data: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    tol: float,
    max_iter: int,
    **kwargs
) -> tuple:
    """
    Gradient descent solver for stationarity.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Metric function.
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Distance function.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    tuple
        A tuple containing the result and metrics.
    """
    # Implement gradient descent logic here
    learning_rate = kwargs.get('learning_rate', 0.01)
    result = np.zeros(data.shape[1])
    for _ in range(max_iter):
        # Update result using gradient descent
        pass
    metrics = _compute_metrics(data, result, metric, distance)
    return result, metrics

def _newton_solver(
    data: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    tol: float,
    max_iter: int,
    **kwargs
) -> tuple:
    """
    Newton solver for stationarity.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Metric function.
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Distance function.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    tuple
        A tuple containing the result and metrics.
    """
    # Implement Newton's method logic here
    result = np.zeros(data.shape[1])
    for _ in range(max_iter):
        # Update result using Newton's method
        pass
    metrics = _compute_metrics(data, result, metric, distance)
    return result, metrics

def _coordinate_descent_solver(
    data: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    tol: float,
    max_iter: int,
    **kwargs
) -> tuple:
    """
    Coordinate descent solver for stationarity.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Metric function.
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Distance function.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    tuple
        A tuple containing the result and metrics.
    """
    # Implement coordinate descent logic here
    result = np.zeros(data.shape[1])
    for _ in range(max_iter):
        # Update result using coordinate descent
        pass
    metrics = _compute_metrics(data, result, metric, distance)
    return result, metrics

def _compute_metrics(
    data: np.ndarray,
    result: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict:
    """
    Compute metrics for the stationarity result.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    result : np.ndarray
        Result of the stationarity model.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Metric function.
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Distance function.

    Returns
    -------
    Dict
        A dictionary of computed metrics.
    """
    metrics = {}
    if isinstance(metric, str):
        if metric == 'mse':
            metrics['mse'] = np.mean((data - result) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(data - result))
        elif metric == 'r2':
            ss_res = np.sum((data - result) ** 2)
            ss_tot = np.sum((data - np.mean(data, axis=0)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot)
    else:
        metrics['custom_metric'] = metric(data, result)

    if isinstance(distance, str):
        if distance == 'euclidean':
            metrics['distance'] = np.linalg.norm(data - result, axis=0)
        elif distance == 'manhattan':
            metrics['distance'] = np.sum(np.abs(data - result), axis=0)
        elif distance == 'cosine':
            metrics['distance'] = 1 - np.sum(data * result, axis=0) / (np.linalg.norm(data, axis=0) * np.linalg.norm(result))
        elif distance == 'minkowski':
            p = kwargs.get('p', 2)
            metrics['distance'] = np.sum(np.abs(data - result) ** p, axis=0) ** (1/p)
    else:
        metrics['custom_distance'] = distance(data, result)

    return metrics

################################################################################
# stationnarite_large
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def stationnarite_large_fit(
    data: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_params: Optional[Dict] = None
) -> Dict:
    """
    Main function to assess stationarity of a time series using various methods.

    Parameters
    ----------
    data : np.ndarray
        Input time series data.
    normalizer : Optional[Callable]
        Function to normalize the data. Default is None (no normalization).
    metric : Union[str, Callable]
        Metric to evaluate stationarity. Can be "mse", "mae", "r2", or custom callable.
    distance : Union[str, Callable]
        Distance metric for stationarity assessment. Can be "euclidean", "manhattan",
        "cosine", "minkowski", or custom callable.
    solver : str
        Solver method. Options: "closed_form", "gradient_descent", "newton",
        "coordinate_descent".
    regularization : Optional[str]
        Regularization type. Options: None, "l1", "l2", "elasticnet".
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_params : Optional[Dict]
        Additional parameters for the solver or other components.

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.random.randn(100)
    >>> result = stationnarite_large_fit(data, normalizer=np.std, metric="mse")
    """
    # Validate inputs
    _validate_inputs(data, normalizer, metric, distance, solver, regularization)

    # Normalize data if specified
    normalized_data = _apply_normalization(data, normalizer)

    # Prepare solver parameters
    solver_params = {
        "tol": tol,
        "max_iter": max_iter,
        **(custom_params or {})
    }

    # Choose solver
    if solver == "closed_form":
        params = _solve_closed_form(normalized_data, distance)
    elif solver == "gradient_descent":
        params = _solve_gradient_descent(normalized_data, distance, solver_params)
    elif solver == "newton":
        params = _solve_newton(normalized_data, distance, solver_params)
    elif solver == "coordinate_descent":
        params = _solve_coordinate_descent(normalized_data, distance, solver_params)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if specified
    if regularization:
        params = _apply_regularization(params, normalized_data, regularization)

    # Calculate metrics
    metrics = _calculate_metrics(normalized_data, params, metric)

    # Prepare results
    result = {
        "result": _assess_stationarity(params, normalized_data),
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric": metric if isinstance(metric, str) else "custom",
            "distance": distance if isinstance(distance, str) else "custom",
            "solver": solver,
            "regularization": regularization
        },
        "warnings": _check_warnings(normalized_data, params)
    }

    return result

def _validate_inputs(
    data: np.ndarray,
    normalizer: Optional[Callable],
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    solver: str,
    regularization: Optional[str]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Data must be a 1-dimensional array")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values")
    if normalizer is not None and not callable(normalizer):
        raise TypeError("Normalizer must be a callable or None")
    if isinstance(metric, str) and metric not in ["mse", "mae", "r2"]:
        raise ValueError("Metric must be 'mse', 'mae', 'r2' or a custom callable")
    if isinstance(distance, str) and distance not in ["euclidean", "manhattan", "cosine", "minkowski"]:
        raise ValueError("Distance must be 'euclidean', 'manhattan', 'cosine', 'minkowski' or a custom callable")
    if solver not in ["closed_form", "gradient_descent", "newton", "coordinate_descent"]:
        raise ValueError("Unknown solver")
    if regularization is not None and regularization not in ["l1", "l2", "elasticnet"]:
        raise ValueError("Regularization must be None, 'l1', 'l2' or 'elasticnet'")

def _apply_normalization(
    data: np.ndarray,
    normalizer: Optional[Callable]
) -> np.ndarray:
    """Apply normalization to the data."""
    if normalizer is None:
        return data
    return normalizer(data)

def _solve_closed_form(
    data: np.ndarray,
    distance: Union[str, Callable]
) -> Dict:
    """Solve stationarity assessment using closed-form solution."""
    # Placeholder for actual implementation
    return {"params": np.zeros(data.shape[1] if data.ndim > 1 else 0)}

def _solve_gradient_descent(
    data: np.ndarray,
    distance: Union[str, Callable],
    params: Dict
) -> Dict:
    """Solve stationarity assessment using gradient descent."""
    # Placeholder for actual implementation
    return {"params": np.zeros(data.shape[1] if data.ndim > 1 else 0)}

def _solve_newton(
    data: np.ndarray,
    distance: Union[str, Callable],
    params: Dict
) -> Dict:
    """Solve stationarity assessment using Newton's method."""
    # Placeholder for actual implementation
    return {"params": np.zeros(data.shape[1] if data.ndim > 1 else 0)}

def _solve_coordinate_descent(
    data: np.ndarray,
    distance: Union[str, Callable],
    params: Dict
) -> Dict:
    """Solve stationarity assessment using coordinate descent."""
    # Placeholder for actual implementation
    return {"params": np.zeros(data.shape[1] if data.ndim > 1 else 0)}

def _apply_regularization(
    params: Dict,
    data: np.ndarray,
    regularization: str
) -> Dict:
    """Apply regularization to the parameters."""
    # Placeholder for actual implementation
    return params

def _calculate_metrics(
    data: np.ndarray,
    params: Dict,
    metric: Union[str, Callable]
) -> Dict:
    """Calculate metrics for stationarity assessment."""
    if isinstance(metric, str):
        if metric == "mse":
            return {"mse": np.mean((data - params["params"]) ** 2)}
        elif metric == "mae":
            return {"mae": np.mean(np.abs(data - params["params"]))}
        elif metric == "r2":
            ss_res = np.sum((data - params["params"]) ** 2)
            ss_tot = np.sum((data - np.mean(data)) ** 2)
            return {"r2": 1 - (ss_res / ss_tot)}
    else:
        return {"custom_metric": metric(data, params["params"])}

def _assess_stationarity(
    params: Dict,
    data: np.ndarray
) -> bool:
    """Assess stationarity based on the parameters."""
    # Placeholder for actual implementation
    return True

def _check_warnings(
    data: np.ndarray,
    params: Dict
) -> List[str]:
    """Check for potential warnings during stationarity assessment."""
    warnings = []
    if np.any(np.isnan(params["params"])) or np.any(np.isinf(params["params"])):
        warnings.append("Parameters contain NaN or infinite values")
    if np.var(data) < 1e-6:
        warnings.append("Data has very low variance")
    return warnings

################################################################################
# test_dicky_fuller
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for Dickey-Fuller test."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if len(data.shape) != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values")
    if len(data) < 2:
        raise ValueError("Input data must have at least 2 observations")

def _compute_statistic(data: np.ndarray, regression_type: str = 'c', autolag: Optional[str] = None) -> float:
    """Compute the Dickey-Fuller test statistic."""
    n = len(data)
    if regression_type == 'c':
        # Constant only
        x = np.column_stack([np.arange(n), data[:-1]])
    elif regression_type == 'ct':
        # Constant and trend
        x = np.column_stack([np.arange(n), data[:-1], np.arange(1, n)])
    elif regression_type == 'ctt':
        # Constant, trend and quadratic trend
        x = np.column_stack([np.arange(n), data[:-1], np.arange(1, n), (np.arange(1, n)**2)])
    else:
        raise ValueError("Invalid regression_type. Must be 'c', 'ct' or 'ctt'")

    y = data[1:]
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ beta
    sigma_squared = np.sum(residuals**2) / (n - len(beta))
    statistic = beta[1] / np.sqrt(sigma_squared * np.linalg.inv(x.T @ x)[1, 1])
    return statistic

def _estimate_critical_values(regression_type: str = 'c') -> Dict[str, float]:
    """Estimate critical values for the Dickey-Fuller test."""
    # These are approximate values, in practice you would use precomputed tables
    if regression_type == 'c':
        return {'1%': -3.43, '5%': -2.86, '10%': -2.57}
    elif regression_type == 'ct':
        return {'1%': -3.96, '5%': -3.41, '10%': -3.12}
    elif regression_type == 'ctt':
        return {'1%': -4.07, '5%': -3.62, '10%': -3.27}
    else:
        raise ValueError("Invalid regression_type. Must be 'c', 'ct' or 'ctt'")

def test_dicky_fuller_fit(
    data: np.ndarray,
    regression_type: str = 'c',
    autolag: Optional[str] = None,
    custom_statistic_func: Optional[Callable[[np.ndarray, str], float]] = None
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Perform the Dickey-Fuller test for stationarity.

    Parameters
    ----------
    data : np.ndarray
        One-dimensional array of observations.
    regression_type : str, optional
        Type of regression to include ('c' for constant only, 'ct' for constant and trend,
        'ctt' for constant, trend and quadratic trend). Default is 'c'.
    autolag : str or None, optional
        Method to determine lag length for autocorrelation correction. Not implemented yet.
    custom_statistic_func : callable or None, optional
        Custom function to compute the test statistic. Must accept (data, regression_type).

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': test statistic value
        - 'critical_values': dictionary of critical values at different significance levels
        - 'metrics': dictionary with p-value and other metrics (not implemented yet)
        - 'params_used': parameters used in the test
        - 'warnings': any warnings generated during computation

    Example
    -------
    >>> data = np.random.randn(100)
    >>> result = test_dicky_fuller_fit(data, regression_type='ct')
    """
    _validate_input(data)

    warnings = []
    params_used = {
        'regression_type': regression_type,
        'autolag': autolag
    }

    if custom_statistic_func is not None:
        statistic = custom_statistic_func(data, regression_type)
    else:
        statistic = _compute_statistic(data, regression_type)

    critical_values = _estimate_critical_values(regression_type)

    # In a real implementation, you would compute the p-value based on the statistic
    # and critical values. This is simplified here.
    metrics = {
        'p_value': 0.5,  # Placeholder
        'statistic_name': 'Dickey-Fuller'
    }

    return {
        'result': statistic,
        'critical_values': critical_values,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

################################################################################
# test_kpss
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(series: np.ndarray) -> None:
    """Validate input series for KPSS test."""
    if not isinstance(series, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if series.ndim != 1:
        raise ValueError("Input must be a 1-dimensional array")
    if np.isnan(series).any() or np.isinf(series).any():
        raise ValueError("Input contains NaN or infinite values")

def _compute_kpss_statistic(series: np.ndarray, regression: str = 'c') -> float:
    """Compute the KPSS test statistic."""
    n = len(series)
    if regression == 'c':
        # Constant regression
        y = series - np.mean(series)
    elif regression == 'ct':
        # Constant and trend regression
        x = np.column_stack([np.ones(n), np.arange(1, n+1)])
        beta = np.linalg.inv(x.T @ x) @ x.T @ series
        y = series - x @ beta
    else:
        raise ValueError("Regression must be either 'c' or 'ct'")

    s = np.cumsum(y)
    numerator = np.sum(s**2) / (n**2)

    # Compute denominator
    if regression == 'c':
        denominator = np.sum((series - np.mean(series))**2)
    else:
        denominator = np.sum(y**2)

    statistic = (n * numerator) / denominator
    return statistic

def _compute_critical_values(alpha: float = 0.05) -> Dict[str, float]:
    """Compute critical values for the KPSS test."""
    # These are approximate values from standard tables
    critical_values = {
        0.10: 0.347,
        0.05: 0.463,
        0.01: 0.739
    }
    return {k: v for k, v in critical_values.items() if k >= alpha}

def test_kpss_fit(
    series: np.ndarray,
    regression: str = 'c',
    alpha: float = 0.05,
    custom_statistic_func: Optional[Callable[[np.ndarray, str], float]] = None
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Perform the KPSS test for stationarity.

    Parameters:
    -----------
    series : np.ndarray
        Input time series data.
    regression : str, optional
        Type of regression to use ('c' for constant, 'ct' for constant and trend).
    alpha : float, optional
        Significance level.
    custom_statistic_func : Callable, optional
        Custom function to compute the test statistic.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    _validate_input(series)

    # Compute test statistic
    if custom_statistic_func is not None:
        statistic = custom_statistic_func(series, regression)
    else:
        statistic = _compute_kpss_statistic(series, regression)

    # Compute critical values
    critical_values = _compute_critical_values(alpha)
    p_value = 1.0 - next((v for k, v in critical_values.items() if statistic < v), 1.0)

    # Determine stationarity
    is_stationary = statistic < critical_values.get(alpha, 0.463)

    return {
        "result": {
            "statistic": statistic,
            "critical_values": critical_values,
            "p_value": p_value,
            "is_stationary": is_stationary
        },
        "metrics": {},
        "params_used": {
            "regression": regression,
            "alpha": alpha
        },
        "warnings": []
    }

def test_kpss_compute(
    series: np.ndarray,
    regression: str = 'c',
    alpha: float = 0.05
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Compute the KPSS test statistic and critical values.

    Parameters:
    -----------
    series : np.ndarray
        Input time series data.
    regression : str, optional
        Type of regression to use ('c' for constant, 'ct' for constant and trend).
    alpha : float, optional
        Significance level.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    return test_kpss_fit(series, regression, alpha)

################################################################################
# autocorrelation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(series: np.ndarray) -> None:
    """Validate input series for autocorrelation computation."""
    if not isinstance(series, np.ndarray):
        raise TypeError("Input series must be a numpy array")
    if series.ndim != 1:
        raise ValueError("Input series must be one-dimensional")
    if np.isnan(series).any() or np.isinf(series).any():
        raise ValueError("Input series contains NaN or infinite values")

def _compute_autocorrelation(series: np.ndarray, lags: int) -> float:
    """Compute autocorrelation for a given lag."""
    mean = np.mean(series)
    c0 = np.sum((series - mean) ** 2)
    if c0 == 0:
        return 0.0
    return np.sum((series[:-lags] - mean) * (series[lags:] - mean)) / c0

def autocorrelation_fit(
    series: np.ndarray,
    max_lag: int = 10,
    normalization: str = "standard",
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict[int, float], Dict[str, float], Dict[str, str], list]]:
    """
    Compute autocorrelation for a time series.

    Parameters:
    -----------
    series : np.ndarray
        Input time series data.
    max_lag : int, optional
        Maximum lag to compute autocorrelation for (default: 10).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard').
    metric : Callable, optional
        Custom metric function to evaluate autocorrelation.
    custom_metric : Callable, optional
        Additional custom metric function.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    _validate_inputs(series)

    # Compute autocorrelation for each lag
    autocorrelations = {}
    for lag in range(1, max_lag + 1):
        autocorrelations[lag] = _compute_autocorrelation(series, lag)

    # Apply normalization if specified
    if normalization == "standard":
        mean = np.mean(list(autocorrelations.values()))
        std = np.std(list(autocorrelations.values()))
        if std != 0:
            autocorrelations = {k: (v - mean) / std for k, v in autocorrelations.items()}
    elif normalization == "minmax":
        min_val = min(autocorrelations.values())
        max_val = max(autocorrelations.values())
        if min_val != max_val:
            autocorrelations = {k: (v - min_val) / (max_val - min_val) for k, v in autocorrelations.items()}
    elif normalization == "robust":
        median = np.median(list(autocorrelations.values()))
        iqr = np.subtract(*np.percentile(list(autocorrelations.values()), [75, 25]))
        if iqr != 0:
            autocorrelations = {k: (v - median) / iqr for k, v in autocorrelations.items()}

    # Compute metrics if provided
    metrics = {}
    if metric is not None:
        metrics["custom_metric"] = metric(series, series)
    if custom_metric is not None:
        metrics["additional_custom_metric"] = custom_metric(series, series)

    return {
        "result": autocorrelations,
        "metrics": metrics,
        "params_used": {
            "max_lag": max_lag,
            "normalization": normalization
        },
        "warnings": []
    }

# Example usage:
# autocorrelation_fit(np.random.randn(100), max_lag=5, normalization="standard")

################################################################################
# variance_constante
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def variance_constante_fit(
    data: np.ndarray,
    normalisation: str = 'none',
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Estimate parameters for constant variance test in time series data.

    Parameters
    ----------
    data : np.ndarray
        Input time series data.
    normalisation : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to evaluate ('mse', 'mae', 'r2') or custom callable.
    distance : str or callable, optional
        Distance metric ('euclidean', 'manhattan', 'cosine') or custom callable.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton').
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
    Dict
        Dictionary containing results, metrics, parameters used and warnings.
    """
    # Validate inputs
    _validate_inputs(data)

    # Normalize data if required
    normalized_data = _apply_normalization(data, normalisation)

    # Select metric
    if callable(metric):
        metric_func = metric
    else:
        metric_func = _get_metric(metric)

    # Select distance
    if callable(distance):
        distance_func = distance
    else:
        distance_func = _get_distance(distance)

    # Select solver
    if solver == 'closed_form':
        params = _solve_closed_form(normalized_data, distance_func)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(
            normalized_data, distance_func,
            tol=tol, max_iter=max_iter
        )
    else:
        raise ValueError(f"Unsupported solver: {solver}")

    # Apply regularization if needed
    if regularization is not None:
        params = _apply_regularization(params, normalized_data, regularization)

    # Calculate metrics
    metrics = _calculate_metrics(normalized_data, params, metric_func)

    # Check for warnings
    warnings = _check_warnings(normalized_data, params)

    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalisation': normalisation,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': warnings
    }

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values")

def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to data."""
    if method == 'none':
        return data
    elif method == 'standard':
        return (data - np.mean(data)) / np.std(data)
    elif method == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == 'robust':
        return (data - np.median(data)) / (np.percentile(data, 75) -
                                          np.percentile(data, 25))
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

def _get_metric(metric_name: str) -> Callable:
    """Get metric function based on name."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared
    }
    if metric_name not in metrics:
        raise ValueError(f"Unsupported metric: {metric_name}")
    return metrics[metric_name]

def _get_distance(distance_name: str) -> Callable:
    """Get distance function based on name."""
    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance
    }
    if distance_name not in distances:
        raise ValueError(f"Unsupported distance: {distance_name}")
    return distances[distance_name]

def _solve_closed_form(data: np.ndarray, distance_func: Callable) -> Dict:
    """Solve for parameters using closed form solution."""
    # Example implementation - adjust based on actual mathematical model
    n = len(data)
    X = np.column_stack([np.ones(n), data])
    params, _, _, _ = np.linalg.lstsq(X, data, rcond=None)
    return {'intercept': params[0], 'slope': params[1]}

def _solve_gradient_descent(
    data: np.ndarray,
    distance_func: Callable,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict:
    """Solve for parameters using gradient descent."""
    # Example implementation - adjust based on actual mathematical model
    n = len(data)
    X = np.column_stack([np.ones(n), data])
    params = np.zeros(2)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradients = _compute_gradients(X, data, params)
        params -= learning_rate * gradients
        if np.linalg.norm(gradients) < tol:
            break

    return {'intercept': params[0], 'slope': params[1]}

def _compute_gradients(X: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Compute gradients for gradient descent."""
    predictions = X @ params
    errors = predictions - y
    gradients = 2 * (X.T @ errors) / len(y)
    return gradients

def _apply_regularization(
    params: Dict,
    data: np.ndarray,
    method: str
) -> Dict:
    """Apply regularization to parameters."""
    # Example implementation - adjust based on actual mathematical model
    n = len(data)
    X = np.column_stack([np.ones(n), data])
    alpha = 1.0

    if method == 'l1':
        params['intercept'] -= alpha * np.sign(params['intercept'])
        params['slope'] -= alpha * np.sign(params['slope'])
    elif method == 'l2':
        params['intercept'] -= alpha * params['intercept']
        params['slope'] -= alpha * params['slope']
    elif method == 'elasticnet':
        params['intercept'] -= alpha * (0.5 * params['intercept'] +
                                      0.5 * np.sign(params['intercept']))
        params['slope'] -= alpha * (0.5 * params['slope'] +
                                  0.5 * np.sign(params['slope']))
    return params

def _calculate_metrics(
    data: np.ndarray,
    params: Dict,
    metric_func: Callable
) -> Dict:
    """Calculate metrics for the model."""
    predictions = params['intercept'] + params['slope'] * data
    return {'value': metric_func(data, predictions)}

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
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def _check_warnings(data: np.ndarray, params: Dict) -> list:
    """Check for potential warnings."""
    warnings = []
    if np.isnan(params['intercept']) or np.isinf(params['intercept']):
        warnings.append("Intercept parameter is NaN or infinite")
    if np.isnan(params['slope']) or np.isinf(params['slope']):
        warnings.append("Slope parameter is NaN or infinite")
    if np.any(np.isnan(data)):
        warnings.append("Input data contains NaN values")
    return warnings

################################################################################
# moyenne_constante
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def moyenne_constante_fit(
    y: np.ndarray,
    normalisation: str = "none",
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    solver: str = "closed_form",
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_normalize: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]:
    """
    Estimate the constant mean of a time series and compute related metrics.

    Parameters
    ----------
    y : np.ndarray
        Input time series data.
    normalisation : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    metric : str or callable, optional
        Metric to evaluate the fit ("mse", "mae", "r2", custom callable).
    solver : str, optional
        Solver method ("closed_form", "gradient_descent").
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations for iterative solvers.
    custom_normalize : callable, optional
        Custom normalization function.
    custom_metric : callable, optional
        Custom metric function.

    Returns
    -------
    dict
        Dictionary containing:
        - "result": estimated constant mean
        - "metrics": computed metrics
        - "params_used": parameters used in the computation
        - "warnings": list of warnings

    Example
    -------
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> result = moyenne_constante_fit(y)
    """
    # Validate inputs
    _validate_inputs(y, normalisation, metric, solver)

    # Normalize data if required
    y_normalized = _apply_normalization(y, normalisation, custom_normalize)

    # Estimate parameters
    params = _estimate_parameters(y_normalized, solver, tol, max_iter)

    # Compute metrics
    metrics = _compute_metrics(y_normalized, params["mean"], metric, custom_metric)

    # Prepare output
    return {
        "result": params["mean"],
        "metrics": metrics,
        "params_used": {
            "normalisation": normalisation,
            "metric": metric if isinstance(metric, str) else "custom",
            "solver": solver,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

def _validate_inputs(
    y: np.ndarray,
    normalisation: str,
    metric: Union[str, Callable],
    solver: str
) -> None:
    """Validate input parameters."""
    if not isinstance(y, np.ndarray):
        raise TypeError("Input y must be a numpy array")
    if len(y.shape) != 1:
        raise ValueError("Input y must be a 1D array")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Input y contains NaN or infinite values")

    valid_normalisations = ["none", "standard", "minmax", "robust"]
    if normalisation not in valid_normalisations:
        raise ValueError(f"Invalid normalisation method. Choose from {valid_normalisations}")

    valid_metrics = ["mse", "mae", "r2"]
    if isinstance(metric, str) and metric not in valid_metrics:
        raise ValueError(f"Invalid metric. Choose from {valid_metrics} or provide a custom callable")

    valid_solvers = ["closed_form", "gradient_descent"]
    if solver not in valid_solvers:
        raise ValueError(f"Invalid solver. Choose from {valid_solvers}")

def _apply_normalization(
    y: np.ndarray,
    method: str,
    custom_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Apply normalization to the input data."""
    if method == "none":
        return y
    elif custom_func is not None:
        return custom_func(y)
    else:
        if method == "standard":
            mean = np.mean(y)
            std = np.std(y)
            return (y - mean) / (std + 1e-8)
        elif method == "minmax":
            min_val = np.min(y)
            max_val = np.max(y)
            return (y - min_val) / (max_val - min_val + 1e-8)
        elif method == "robust":
            median = np.median(y)
            iqr = np.percentile(y, 75) - np.percentile(y, 25)
            return (y - median) / (iqr + 1e-8)

def _estimate_parameters(
    y: np.ndarray,
    solver: str,
    tol: float,
    max_iter: int
) -> Dict[str, float]:
    """Estimate the constant mean parameter."""
    if solver == "closed_form":
        return {"mean": np.mean(y)}
    else:
        # Gradient descent implementation
        mean = y.mean()
        for _ in range(max_iter):
            gradient = np.mean(y - mean)
            new_mean = mean + gradient
            if abs(new_mean - mean) < tol:
                break
            mean = new_mean
        return {"mean": mean}

def _compute_metrics(
    y: np.ndarray,
    mean: float,
    metric: Union[str, Callable],
    custom_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    if custom_func is not None:
        return {"custom": custom_func(y, np.full_like(y, mean))}

    metrics = {}
    if isinstance(metric, str) or metric == "mse":
        mse = np.mean((y - mean) ** 2)
        metrics["mse"] = mse
    if isinstance(metric, str) or metric == "mae":
        mae = np.mean(np.abs(y - mean))
        metrics["mae"] = mae
    if isinstance(metric, str) or metric == "r2":
        ss_total = np.sum((y - y.mean()) ** 2)
        ss_residual = np.sum((y - mean) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        metrics["r2"] = r2
    return metrics

################################################################################
# differenciation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(
    series: np.ndarray,
    order: int = 1,
    normalize: str = "none",
) -> None:
    """Validate input series and parameters."""
    if not isinstance(series, np.ndarray):
        raise TypeError("Input series must be a numpy array.")
    if order < 0:
        raise ValueError("Order of differentiation must be non-negative.")
    if normalize not in ["none", "standard", "minmax", "robust"]:
        raise ValueError("Normalization must be one of: none, standard, minmax, robust.")
    if np.any(np.isnan(series)) or np.any(np.isinf(series)):
        raise ValueError("Input series must not contain NaN or infinite values.")

def _normalize_series(
    series: np.ndarray,
    method: str = "none",
) -> np.ndarray:
    """Normalize the input series."""
    if method == "standard":
        return (series - np.mean(series)) / np.std(series)
    elif method == "minmax":
        return (series - np.min(series)) / (np.max(series) - np.min(series))
    elif method == "robust":
        median = np.median(series)
        iqr = np.percentile(series, 75) - np.percentile(series, 25)
        return (series - median) / iqr
    else:
        return series.copy()

def _differentiate(
    series: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """Compute the differentiation of the series."""
    diff_series = series.copy()
    for _ in range(order):
        diff_series = np.diff(diff_series)
    return diff_series

def _compute_metrics(
    original: np.ndarray,
    differentiated: np.ndarray,
    metric_funcs: Dict[str, Callable] = None,
) -> Dict[str, float]:
    """Compute metrics between original and differentiated series."""
    if metric_funcs is None:
        metric_funcs = {
            "mse": lambda x, y: np.mean((x - y) ** 2),
            "mae": lambda x, y: np.mean(np.abs(x - y)),
        }

    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(original, differentiated)
        except Exception as e:
            metrics[f"error_{name}"] = str(e)

    return metrics

def differenciation_fit(
    series: np.ndarray,
    order: int = 1,
    normalize: str = "none",
    metric_funcs: Optional[Dict[str, Callable]] = None,
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Compute the differentiation of a time series and return results with metrics.

    Parameters:
    -----------
    series : np.ndarray
        Input time series.
    order : int, optional
        Order of differentiation (default is 1).
    normalize : str, optional
        Normalization method: "none", "standard", "minmax", or "robust" (default is "none").
    metric_funcs : Dict[str, Callable], optional
        Dictionary of custom metric functions (default is None).

    Returns:
    --------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]
        Dictionary containing:
        - "result": Differentiated series.
        - "metrics": Computed metrics.
        - "params_used": Parameters used in the computation.
        - "warnings": Any warnings encountered.

    Example:
    --------
    >>> series = np.array([1, 2, 3, 4, 5])
    >>> result = differenciation_fit(series, order=1)
    """
    _validate_input(series, order, normalize)

    normalized_series = _normalize_series(series, normalize)
    differentiated_series = _differentiate(normalized_series, order)

    metrics = _compute_metrics(series, differentiated_series, metric_funcs)

    return {
        "result": differentiated_series,
        "metrics": metrics,
        "params_used": {
            "order": order,
            "normalize": normalize,
        },
        "warnings": [],
    }

################################################################################
# transformation_box_cox
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for Box-Cox transformation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(data <= 0):
        raise ValueError("Input data must be strictly positive for Box-Cox transformation")

def _box_cox_transform(data: np.ndarray, lambda_: float) -> np.ndarray:
    """Apply Box-Cox transformation to the data."""
    if lambda_ == 0:
        return np.log(data)
    return (data ** lambda_ - 1) / lambda_

def _box_cox_inverse_transform(data: np.ndarray, lambda_: float) -> np.ndarray:
    """Apply inverse Box-Cox transformation to the data."""
    if lambda_ == 0:
        return np.exp(data)
    return (lambda_ * data + 1) ** (1 / lambda_)

def _compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def _compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def transformation_box_cox_fit(
    data: np.ndarray,
    metric: str = "mse",
    solver: str = "closed_form",
    lambda_range: Optional[np.ndarray] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, str]]:
    """
    Fit Box-Cox transformation to data.

    Parameters
    ----------
    data : np.ndarray
        Input data to transform.
    metric : str, optional
        Metric to optimize ("mse", "mae", "r2").
    solver : str, optional
        Solver to use ("closed_form").
    lambda_range : np.ndarray, optional
        Range of lambda values to test.
    custom_metric : Callable, optional
        Custom metric function.

    Returns
    -------
    Dict[str, Union[Dict, float, str]]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    _validate_input(data)

    if lambda_range is None:
        lambda_range = np.linspace(-2, 3, 100)

    best_lambda = None
    best_metric_value = float('inf')
    transformed_data = None

    for lambda_ in lambda_range:
        try:
            current_transformed = _box_cox_transform(data, lambda_)
            if metric == "mse":
                current_metric = _compute_mse(data, current_transformed)
            elif metric == "mae":
                current_metric = _compute_mae(data, current_transformed)
            elif metric == "r2":
                current_metric = _compute_r2(data, current_transformed)
            elif custom_metric is not None:
                current_metric = custom_metric(data, current_transformed)
            else:
                raise ValueError("Invalid metric or no custom metric provided")

            if current_metric < best_metric_value:
                best_metric_value = current_metric
                best_lambda = lambda_
                transformed_data = current_transformed
        except Exception as e:
            continue

    if best_lambda is None:
        raise ValueError("No valid lambda found")

    result = {
        "result": transformed_data,
        "metrics": {metric: best_metric_value},
        "params_used": {"lambda": best_lambda, "solver": solver, "metric": metric},
        "warnings": []
    }

    return result

def transformation_box_cox_compute(
    data: np.ndarray,
    lambda_: float
) -> Dict[str, Union[np.ndarray, str]]:
    """
    Compute Box-Cox transformation for a given lambda.

    Parameters
    ----------
    data : np.ndarray
        Input data to transform.
    lambda_ : float
        Lambda value for the transformation.

    Returns
    -------
    Dict[str, Union[np.ndarray, str]]
        Dictionary containing the transformed data and lambda used.
    """
    _validate_input(data)
    transformed_data = _box_cox_transform(data, lambda_)

    return {
        "result": transformed_data,
        "params_used": {"lambda": lambda_},
        "warnings": []
    }

################################################################################
# saisonnalite
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(
    series: np.ndarray,
    period: int,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> None:
    """Validate input series and parameters."""
    if not isinstance(series, np.ndarray):
        raise TypeError("Input series must be a numpy array")
    if len(series.shape) != 1:
        raise ValueError("Input series must be 1-dimensional")
    if np.any(np.isnan(series)) or np.any(np.isinf(series)):
        raise ValueError("Input series contains NaN or infinite values")
    if period <= 0:
        raise ValueError("Period must be a positive integer")

def _apply_normalization(
    series: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Apply normalization to the input series."""
    if normalizer is None:
        return series
    return normalizer(series)

def _compute_seasonal_components(
    series: np.ndarray,
    period: int
) -> Dict[str, np.ndarray]:
    """Compute seasonal components using moving averages."""
    n = len(series)
    seasonal_components = np.zeros(n)

    for i in range(n):
        start = max(0, i - period)
        end = min(n, i + period + 1)
        seasonal_components[i] = np.mean(series[start:end])

    return {"seasonal_components": seasonal_components}

def _compute_metrics(
    series: np.ndarray,
    seasonal_components: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Compute metrics between original series and seasonal components."""
    return {"metric": metric_func(series, seasonal_components)}

def saisonnalite_fit(
    series: np.ndarray,
    period: int = 12,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric_func: Callable[[np.ndarray, np.ndarray], float] = lambda y, y_pred: np.mean((y - y_pred) ** 2)
) -> Dict[str, Union[Dict[str, np.ndarray], Dict[str, float], Dict[str, Union[int, Callable]], list]]:
    """
    Fit seasonal components to a time series.

    Parameters
    ----------
    series : np.ndarray
        Input time series data.
    period : int, optional
        Period of seasonality (default is 12).
    normalizer : Callable, optional
        Function to normalize the input series (default is None).
    metric_func : Callable, optional
        Function to compute the metric between original series and seasonal components (default is MSE).

    Returns
    -------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    _validate_inputs(series, period, normalizer)
    normalized_series = _apply_normalization(series, normalizer)
    seasonal_components = _compute_seasonal_components(normalized_series, period)
    metrics = _compute_metrics(series, seasonal_components["seasonal_components"], metric_func)

    return {
        "result": seasonal_components,
        "metrics": metrics,
        "params_used": {
            "period": period,
            "normalizer": normalizer,
            "metric_func": metric_func
        },
        "warnings": []
    }

# Example usage:
# series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
# result = saisonnalite_fit(series, period=3)
