"""
Quantix – Module lissage
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# moyenne_mobile
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(
    y: np.ndarray,
    window_size: int,
    weights: Optional[np.ndarray] = None
) -> None:
    """Validate input data and parameters."""
    if not isinstance(y, np.ndarray):
        raise TypeError("Input y must be a numpy array")
    if window_size <= 0:
        raise ValueError("Window size must be positive")
    if weights is not None and len(weights) != window_size:
        raise ValueError("Weights length must match window size")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Input array contains NaN or infinite values")

def _apply_normalization(
    y: np.ndarray,
    method: str = "none"
) -> np.ndarray:
    """Apply normalization to input data."""
    if method == "standard":
        return (y - np.mean(y)) / np.std(y)
    elif method == "minmax":
        return (y - np.min(y)) / (np.max(y) - np.min(y))
    elif method == "robust":
        median = np.median(y)
        iqr = np.percentile(y, 75) - np.percentile(y, 25)
        return (y - median) / iqr
    elif method == "none":
        return y.copy()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_funcs: Dict[str, Callable]
) -> Dict[str, float]:
    """Compute specified metrics between true and predicted values."""
    metrics = {}
    for name, func in metric_funcs.items():
        if callable(func):
            metrics[name] = func(y_true, y_pred)
    return metrics

def _moving_average(
    y: np.ndarray,
    window_size: int,
    weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute moving average with optional weights."""
    if weights is None:
        weights = np.ones(window_size) / window_size
    else:
        weights = weights / np.sum(weights)

    padded_y = np.pad(y, (window_size//2, window_size-1-window_size//2), mode='reflect')
    result = np.convolve(padded_y, weights, mode='valid')

    return result

def moyenne_mobile_fit(
    y: np.ndarray,
    window_size: int = 3,
    weights: Optional[np.ndarray] = None,
    normalization: str = "none",
    metrics: Dict[str, Callable] = None,
    **kwargs
) -> Dict:
    """
    Compute moving average with configurable parameters.

    Parameters:
    - y: Input time series data
    - window_size: Size of the moving window
    - weights: Optional custom weights for each point in the window
    - normalization: Normalization method to apply ("none", "standard", "minmax", "robust")
    - metrics: Dictionary of metric functions to compute

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate inputs
    _validate_inputs(y, window_size, weights)

    # Apply normalization if specified
    y_normalized = _apply_normalization(y, normalization)

    # Compute moving average
    smoothed = _moving_average(y_normalized, window_size, weights)

    # Compute metrics if specified
    metrics_result = {}
    if metrics is not None:
        metrics_result = _compute_metrics(y_normalized, smoothed, metrics)

    # Prepare output
    result = {
        "result": smoothed,
        "metrics": metrics_result,
        "params_used": {
            "window_size": window_size,
            "normalization": normalization
        },
        "warnings": []
    }

    return result

# Example metric functions
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

# Example usage:
# result = moyenne_mobile_fit(
#     np.array([1, 2, 3, 4, 5]),
#     window_size=3,
#     metrics={"mse": mse, "mae": mae}
# )

################################################################################
# exponentielle_lissée
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    y: np.ndarray,
    alpha: float = 0.5,
    initial_value: Optional[float] = None
) -> Dict[str, np.ndarray]:
    """
    Validate and preprocess inputs for exponential smoothing.

    Parameters
    ----------
    y : np.ndarray
        Time series data to be smoothed.
    alpha : float, optional
        Smoothing factor (0 < alpha <= 1), by default 0.5.
    initial_value : Optional[float], optional
        Initial value for the smoothed series, by default None.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing validated inputs and warnings.

    Raises
    ------
    ValueError
        If input validation fails.
    """
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    if len(y.shape) != 1:
        raise ValueError("Input y must be a 1-dimensional array.")

    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Input y contains NaN or infinite values.")

    if not 0 < alpha <= 1:
        raise ValueError("Smoothing factor alpha must be in (0, 1].")

    if initial_value is not None and len(y) > 0:
        if not isinstance(initial_value, (int, float)):
            raise ValueError("Initial value must be a number.")
        if np.isnan(initial_value) or np.isinf(initial_value):
            raise ValueError("Initial value contains NaN or infinite values.")

    warnings = []
    if initial_value is None:
        initial_value = y[0] if len(y) > 0 else 0.0
        warnings.append("Initial value not provided, using first observation.")

    return {
        "y": y,
        "alpha": alpha,
        "initial_value": initial_value,
        "warnings": warnings
    }

def _compute_exponential_smoothing(
    y: np.ndarray,
    alpha: float,
    initial_value: float
) -> np.ndarray:
    """
    Compute exponential smoothing.

    Parameters
    ----------
    y : np.ndarray
        Time series data to be smoothed.
    alpha : float
        Smoothing factor (0 < alpha <= 1).
    initial_value : float
        Initial value for the smoothed series.

    Returns
    -------
    np.ndarray
        Smoothed time series.
    """
    smoothed = np.zeros_like(y, dtype=float)
    smoothed[0] = initial_value

    for t in range(1, len(y)):
        smoothed[t] = alpha * y[t] + (1 - alpha) * smoothed[t-1]

    return smoothed

def _compute_metrics(
    y: np.ndarray,
    smoothed: np.ndarray,
    metric_funcs: Dict[str, Callable]
) -> Dict[str, float]:
    """
    Compute metrics for the smoothed series.

    Parameters
    ----------
    y : np.ndarray
        Original time series data.
    smoothed : np.ndarray
        Smoothed time series.
    metric_funcs : Dict[str, Callable]
        Dictionary of metric functions to compute.

    Returns
    -------
    Dict[str, float]
        Computed metrics.
    """
    metrics = {}

    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(y, smoothed)
        except Exception as e:
            metrics[name] = np.nan
            print(f"Warning: Failed to compute metric {name}: {str(e)}")

    return metrics

def _default_mse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Default mean squared error metric.

    Parameters
    ----------
    y : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Mean squared error.
    """
    return np.mean((y - y_pred) ** 2)

def _default_mae(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Default mean absolute error metric.

    Parameters
    ----------
    y : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Mean absolute error.
    """
    return np.mean(np.abs(y - y_pred))

def exponentielle_lissée_fit(
    y: Union[np.ndarray, list],
    alpha: float = 0.5,
    initial_value: Optional[float] = None,
    metrics: Dict[str, Callable] = None
) -> Dict:
    """
    Fit exponential smoothing to time series data.

    Parameters
    ----------
    y : Union[np.ndarray, list]
        Time series data to be smoothed.
    alpha : float, optional
        Smoothing factor (0 < alpha <= 1), by default 0.5.
    initial_value : Optional[float], optional
        Initial value for the smoothed series, by default None.
    metrics : Dict[str, Callable], optional
        Dictionary of metric functions to compute, by default None.

    Returns
    -------
    Dict
        Dictionary containing:
        - "result": Smoothed time series.
        - "metrics": Computed metrics.
        - "params_used": Parameters used in the computation.
        - "warnings": Warnings generated during computation.

    Examples
    --------
    >>> y = [1, 2, 3, 4, 5]
    >>> result = exponentielle_lissée_fit(y)
    """
    # Validate inputs
    validated_inputs = _validate_inputs(y, alpha, initial_value)
    y = validated_inputs["y"]
    alpha = validated_inputs["alpha"]
    initial_value = validated_inputs["initial_value"]

    # Compute exponential smoothing
    smoothed = _compute_exponential_smoothing(y, alpha, initial_value)

    # Set default metrics if none provided
    if metrics is None:
        metrics = {
            "mse": _default_mse,
            "mae": _default_mae
        }

    # Compute metrics
    computed_metrics = _compute_metrics(y, smoothed, metrics)

    return {
        "result": smoothed,
        "metrics": computed_metrics,
        "params_used": {
            "alpha": alpha,
            "initial_value": initial_value
        },
        "warnings": validated_inputs["warnings"]
    }

################################################################################
# holt_winters
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    y: np.ndarray,
    seasonal_periods: int,
    trend: str = 'additive',
    seasonal: str = 'additive'
) -> None:
    """Validate input data and parameters."""
    if not isinstance(y, np.ndarray):
        raise TypeError("Input y must be a numpy array")
    if len(y.shape) != 1:
        raise ValueError("Input y must be a 1D array")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Input y contains NaN or infinite values")
    if seasonal_periods <= 0:
        raise ValueError("seasonal_periods must be positive")
    if trend not in ['additive', 'multiplicative']:
        raise ValueError("trend must be either 'additive' or 'multiplicative'")
    if seasonal not in ['additive', 'multiplicative']:
        raise ValueError("seasonal must be either 'additive' or 'multiplicative'")

def _initialize_parameters(
    y: np.ndarray,
    seasonal_periods: int,
    trend: str = 'additive',
    seasonal: str = 'additive'
) -> Dict[str, np.ndarray]:
    """Initialize Holt-Winters parameters."""
    level = np.mean(y[:seasonal_periods])
    if trend == 'additive':
        trend_param = np.mean(np.diff(y[:seasonal_periods]))
    else:
        trend_param = 1.0
    if seasonal == 'additive':
        seasonality = np.zeros(seasonal_periods)
        seasonality[:len(y) % seasonal_periods] = y[:len(y) % seasonal_periods] - level
    else:
        seasonality = np.ones(seasonal_periods)
        seasonality[:len(y) % seasonal_periods] = y[:len(y) % seasonal_periods] / level
    return {
        'level': np.array([level]),
        'trend': np.array([trend_param]),
        'seasonality': seasonality
    }

def _update_parameters(
    y: np.ndarray,
    params: Dict[str, np.ndarray],
    alpha: float = 0.1,
    beta: float = 0.1,
    gamma: float = 0.1,
    trend: str = 'additive',
    seasonal: str = 'additive'
) -> Dict[str, np.ndarray]:
    """Update Holt-Winters parameters."""
    level = params['level'][-1]
    trend_param = params['trend'][-1] if trend == 'additive' else params['trend'][0]
    seasonality = params['seasonality']

    for i in range(len(y)):
        if seasonal == 'additive':
            season = seasonality[i % len(seasonality)]
        else:
            season = seasonality[i % len(seasonality)]

        if trend == 'additive':
            forecast = level + i * trend_param + season
        else:
            forecast = level * (trend_param ** i) * season

        error = y[i] - forecast
        level += alpha * error
        if trend == 'additive':
            trend_param += beta * error
        else:
            trend_param *= (1 + beta * error / forecast)

        if seasonal == 'additive':
            seasonality[i % len(seasonality)] += gamma * error
        else:
            seasonality[i % len(seasonality)] *= (1 + gamma * error / forecast)

    return {
        'level': np.append(params['level'], level),
        'trend': np.append(params['trend'], trend_param) if trend == 'additive' else params['trend'],
        'seasonality': seasonality
    }

def _compute_metrics(
    y: np.ndarray,
    forecasts: np.ndarray,
    metric_funcs: Dict[str, Callable] = None
) -> Dict[str, float]:
    """Compute metrics for model evaluation."""
    if metric_funcs is None:
        metric_funcs = {
            'mse': lambda y, f: np.mean((y - f) ** 2),
            'mae': lambda y, f: np.mean(np.abs(y - f)),
            'r2': lambda y, f: 1 - np.sum((y - f) ** 2) / np.sum((y - np.mean(y)) ** 2)
        }

    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(y, forecasts)
        except Exception as e:
            metrics[f"{name}_error"] = str(e)

    return metrics

def holt_winters_fit(
    y: np.ndarray,
    seasonal_periods: int = 12,
    trend: str = 'additive',
    seasonal: str = 'additive',
    alpha: float = 0.1,
    beta: float = 0.1,
    gamma: float = 0.1,
    metric_funcs: Optional[Dict[str, Callable]] = None
) -> Dict:
    """Fit Holt-Winters model to time series data.

    Example:
        >>> y = np.array([10, 20, 30, 40, 50])
        >>> result = holt_winters_fit(y, seasonal_periods=1)
    """
    _validate_inputs(y, seasonal_periods, trend, seasonal)

    params = _initialize_parameters(y, seasonal_periods, trend, seasonal)
    params = _update_parameters(y, params, alpha, beta, gamma, trend, seasonal)

    forecasts = np.zeros_like(y)
    for i in range(len(y)):
        if seasonal == 'additive':
            season = params['seasonality'][i % len(params['seasonality'])]
        else:
            season = params['seasonality'][i % len(params['seasonality'])]

        if trend == 'additive':
            forecasts[i] = params['level'][-1] + i * params['trend'][-1] + season
        else:
            forecasts[i] = params['level'][-1] * (params['trend'][0] ** i) * season

    metrics = _compute_metrics(y, forecasts, metric_funcs)

    return {
        'result': {
            'forecasts': forecasts,
            'params': params
        },
        'metrics': metrics,
        'params_used': {
            'seasonal_periods': seasonal_periods,
            'trend': trend,
            'seasonal': seasonal,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma
        },
        'warnings': []
    }

################################################################################
# kalman_filter
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def kalman_filter_fit(
    observations: np.ndarray,
    initial_state_estimate: Optional[np.ndarray] = None,
    initial_covariance_estimate: Optional[np.ndarray] = None,
    process_noise_covariance: Optional[np.ndarray] = None,
    measurement_noise_covariance: Optional[np.ndarray] = None,
    transition_matrix: Optional[np.ndarray] = None,
    measurement_matrix: Optional[np.ndarray] = None,
    control_input_matrix: Optional[np.ndarray] = None,
    control_input: Optional[np.ndarray] = None,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: str = 'mse',
    solver: str = 'closed_form',
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]:
    """
    Fit a Kalman filter to the given observations.

    Parameters
    ----------
    observations : np.ndarray
        Array of shape (n_observations, n_features) containing the observed data.
    initial_state_estimate : np.ndarray, optional
        Initial state estimate of shape (n_states,).
    initial_covariance_estimate : np.ndarray, optional
        Initial covariance estimate of shape (n_states, n_states).
    process_noise_covariance : np.ndarray, optional
        Process noise covariance of shape (n_states, n_states).
    measurement_noise_covariance : np.ndarray, optional
        Measurement noise covariance of shape (n_features, n_features).
    transition_matrix : np.ndarray, optional
        Transition matrix of shape (n_states, n_states).
    measurement_matrix : np.ndarray, optional
        Measurement matrix of shape (n_features, n_states).
    control_input_matrix : np.ndarray, optional
        Control input matrix of shape (n_states, n_control_inputs).
    control_input : np.ndarray, optional
        Control input of shape (n_observations, n_control_inputs).
    normalizer : Callable[[np.ndarray], np.ndarray], optional
        Function to normalize the observations.
    metric : str, optional
        Metric to evaluate the performance of the filter. Options: 'mse', 'mae', 'r2'.
    solver : str, optional
        Solver to use for the Kalman filter. Options: 'closed_form'.
    tolerance : float, optional
        Tolerance for convergence.
    max_iterations : int, optional
        Maximum number of iterations.
    custom_metric : Callable[[np.ndarray, np.ndarray], float], optional
        Custom metric function.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(
        observations=observations,
        initial_state_estimate=initial_state_estimate,
        initial_covariance_estimate=initial_covariance_estimate,
        process_noise_covariance=process_noise_covariance,
        measurement_noise_covariance=measurement_noise_covariance,
        transition_matrix=transition_matrix,
        measurement_matrix=measurement_matrix,
        control_input_matrix=control_input_matrix,
        control_input=control_input
    )

    # Normalize observations if a normalizer is provided
    normalized_observations = normalizer(observations)

    # Initialize parameters with defaults if not provided
    n_observations, n_features = normalized_observations.shape
    n_states = initial_state_estimate.shape[0] if initial_state_estimate is not None else n_features
    transition_matrix = np.eye(n_states) if transition_matrix is None else transition_matrix
    measurement_matrix = np.eye(n_features, n_states) if measurement_matrix is None else measurement_matrix
    initial_state_estimate = np.zeros(n_states) if initial_state_estimate is None else initial_state_estimate
    initial_covariance_estimate = np.eye(n_states) if initial_covariance_estimate is None else initial_covariance_estimate
    process_noise_covariance = np.eye(n_states) if process_noise_covariance is None else process_noise_covariance
    measurement_noise_covariance = np.eye(n_features) if measurement_noise_covariance is None else measurement_noise_covariance

    # Initialize variables
    state_estimates = np.zeros((n_observations, n_states))
    covariance_estimates = np.zeros((n_observations, n_states, n_states))

    # Kalman filter main loop
    for t in range(n_observations):
        if t == 0:
            state_estimates[t] = initial_state_estimate
            covariance_estimates[t] = initial_covariance_estimate
        else:
            # Prediction step
            state_estimates[t], covariance_estimates[t] = _predict(
                state_estimates[t-1],
                covariance_estimates[t-1],
                transition_matrix,
                process_noise_covariance,
                control_input_matrix=control_input_matrix,
                control_input=control_input[t] if control_input is not None else None
            )

            # Update step
            state_estimates[t], covariance_estimates[t] = _update(
                state_estimates[t],
                covariance_estimations[t],
                normalized_observations[t],
                measurement_matrix,
                measurement_noise_covariance
            )

    # Calculate metrics
    metrics = _calculate_metrics(
        state_estimates,
        normalized_observations,
        measurement_matrix,
        metric=metric,
        custom_metric=custom_metric
    )

    # Return results
    return {
        'result': state_estimates,
        'metrics': metrics,
        'params_used': {
            'initial_state_estimate': initial_state_estimate.tolist(),
            'initial_covariance_estimate': initial_covariance_estimate.tolist(),
            'process_noise_covariance': process_noise_covariance.tolist(),
            'measurement_noise_covariance': measurement_noise_covariance.tolist(),
            'transition_matrix': transition_matrix.tolist(),
            'measurement_matrix': measurement_matrix.tolist(),
            'control_input_matrix': control_input_matrix.tolist() if control_input_matrix is not None else None,
            'control_input': control_input.tolist() if control_input is not None else None
        },
        'warnings': []
    }

def _validate_inputs(
    observations: np.ndarray,
    initial_state_estimate: Optional[np.ndarray],
    initial_covariance_estimate: Optional[np.ndarray],
    process_noise_covariance: Optional[np.ndarray],
    measurement_noise_covariance: Optional[np.ndarray],
    transition_matrix: Optional[np.ndarray],
    measurement_matrix: Optional[np.ndarray],
    control_input_matrix: Optional[np.ndarray],
    control_input: Optional[np.ndarray]
) -> None:
    """Validate the inputs for the Kalman filter."""
    if observations.ndim != 2:
        raise ValueError("Observations must be a 2D array.")
    if np.any(np.isnan(observations)):
        raise ValueError("Observations contain NaN values.")
    if initial_state_estimate is not None and len(initial_state_estimate) != observations.shape[1]:
        raise ValueError("Initial state estimate must have the same number of features as observations.")
    if initial_covariance_estimate is not None and (initial_covariance_estimate.shape[0] != initial_covariance_estimate.shape[1] or
                                                   initial_covariance_estimate.shape[0] != len(initial_state_estimate)):
        raise ValueError("Initial covariance estimate must be a square matrix with dimensions matching the state estimate.")
    if process_noise_covariance is not None and (process_noise_covariance.shape[0] != process_noise_covariance.shape[1]):
        raise ValueError("Process noise covariance must be a square matrix.")
    if measurement_noise_covariance is not None and (measurement_noise_covariance.shape[0] != measurement_noise_covariance.shape[1]):
        raise ValueError("Measurement noise covariance must be a square matrix.")
    if transition_matrix is not None and (transition_matrix.shape[0] != transition_matrix.shape[1]):
        raise ValueError("Transition matrix must be a square matrix.")
    if measurement_matrix is not None and (measurement_matrix.shape[1] != observations.shape[1]):
        raise ValueError("Measurement matrix must have the same number of features as observations.")
    if control_input_matrix is not None and (control_input_matrix.shape[1] != len(initial_state_estimate)):
        raise ValueError("Control input matrix must have the same number of states as initial state estimate.")
    if control_input is not None and (control_input.shape[0] != observations.shape[0]):
        raise ValueError("Control input must have the same number of rows as observations.")

def _predict(
    state_estimate: np.ndarray,
    covariance_estimate: np.ndarray,
    transition_matrix: np.ndarray,
    process_noise_covariance: np.ndarray,
    control_input_matrix: Optional[np.ndarray] = None,
    control_input: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Predict the next state and covariance."""
    if control_input_matrix is not None and control_input is not None:
        state_estimate = transition_matrix @ state_estimate + control_input_matrix @ control_input
    else:
        state_estimate = transition_matrix @ state_estimate

    covariance_estimate = (transition_matrix @ covariance_estimate @ transition_matrix.T +
                           process_noise_covariance)

    return state_estimate, covariance_estimate

def _update(
    state_estimate: np.ndarray,
    covariance_estimate: np.ndarray,
    observation: np.ndarray,
    measurement_matrix: np.ndarray,
    measurement_noise_covariance: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Update the state and covariance with the new observation."""
    innovation_covariance = (measurement_matrix @ covariance_estimate @ measurement_matrix.T +
                            measurement_noise_covariance)
    kalman_gain = covariance_estimate @ measurement_matrix.T @ np.linalg.inv(innovation_covariance)
    state_estimate = state_estimate + kalman_gain @ (observation - measurement_matrix @ state_estimate)
    covariance_estimate = (covariance_estimate -
                          kalman_gain @ measurement_matrix @ covariance_estimate)

    return state_estimate, covariance_estimate

def _calculate_metrics(
    state_estimates: np.ndarray,
    observations: np.ndarray,
    measurement_matrix: np.ndarray,
    metric: str = 'mse',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Calculate the metrics for the Kalman filter."""
    predicted_observations = measurement_matrix @ state_estimates.T
    metrics = {}

    if metric == 'mse':
        mse = np.mean((observations - predicted_observations.T) ** 2)
        metrics['mse'] = mse
    elif metric == 'mae':
        mae = np.mean(np.abs(observations - predicted_observations.T))
        metrics['mae'] = mae
    elif metric == 'r2':
        ss_res = np.sum((observations - predicted_observations.T) ** 2)
        ss_tot = np.sum((observations - np.mean(observations, axis=0)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        metrics['r2'] = r2
    elif custom_metric is not None:
        metrics['custom_metric'] = custom_metric(observations, predicted_observations.T)

    return metrics

################################################################################
# savitzky_golay
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    x: np.ndarray,
    window_length: int,
    polyorder: int,
    deriv: int = 0
) -> None:
    """Validate inputs for Savitzky-Golay filter."""
    if not isinstance(x, np.ndarray) or x.ndim != 1:
        raise ValueError("Input data must be a 1D numpy array")
    if window_length < 2:
        raise ValueError("Window length must be at least 2")
    if window_length > len(x):
        raise ValueError("Window length must be less than or equal to the input size")
    if polyorder >= window_length:
        raise ValueError("Polynomial order must be less than the window length")
    if deriv < 0:
        raise ValueError("Derivative order must be non-negative")

def _compute_savitzky_golay_coefficients(
    window_length: int,
    polyorder: int,
    deriv: int = 0
) -> np.ndarray:
    """Compute Savitzky-Golay coefficients."""
    order_range = range(polyorder + 1)
    deriv_matrix = np.zeros((polyorder + 1, polyorder + 1))
    for i in order_range:
        deriv_matrix[:, i] = [i**j for j in order_range]

    mid_point = window_length // 2
    x = np.arange(-mid_point, mid_point + 1)
    y = np.polyval(np.eye(polyorder + 1)[deriv], x)

    coefficients = np.linalg.pinv(deriv_matrix) @ y
    return coefficients

def _apply_savitzky_golay(
    x: np.ndarray,
    window_length: int,
    polyorder: int,
    deriv: int = 0
) -> np.ndarray:
    """Apply Savitzky-Golay filter to input data."""
    coefficients = _compute_savitzky_golay_coefficients(window_length, polyorder, deriv)
    mid_point = window_length // 2
    padded_x = np.pad(x, (mid_point, mid_point), mode='reflect')
    result = np.convolve(padded_x, coefficients, mode='valid')
    return result

def savitzky_golay_fit(
    x: np.ndarray,
    window_length: int = 5,
    polyorder: int = 2,
    deriv: int = 0,
    normalize: str = 'none',
    metric: Union[str, Callable] = 'mse'
) -> Dict:
    """Apply Savitzky-Golay filter to smooth or differentiate data.

    Parameters
    ----------
    x : np.ndarray
        Input 1D array to be smoothed.
    window_length : int, optional
        Length of the smoothing window (must be odd).
    polyorder : int, optional
        Order of the polynomial used to fit the samples.
    deriv : int, optional
        Order of the derivative to compute (default is 0 for smoothing).
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to evaluate the smoothing ('mse', 'mae', 'r2', or custom callable).

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Smoothed or differentiated data.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used in the computation.
        - 'warnings': Any warnings generated during processing.

    Example
    -------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> result = savitzky_golay_fit(x, window_length=3, polyorder=1)
    """
    _validate_inputs(x, window_length, polyorder, deriv)

    result = _apply_savitzky_golay(x, window_length, polyorder, deriv)

    # Normalization
    if normalize == 'standard':
        mean = np.mean(result)
        std = np.std(result)
        result = (result - mean) / std
    elif normalize == 'minmax':
        min_val = np.min(result)
        max_val = np.max(result)
        result = (result - min_val) / (max_val - min_val)
    elif normalize == 'robust':
        median = np.median(result)
        iqr = np.percentile(result, 75) - np.percentile(result, 25)
        result = (result - median) / iqr

    # Metric computation
    metrics = {}
    if isinstance(metric, str):
        if metric == 'mse':
            metrics['mse'] = np.mean((x - result) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(x - result))
        elif metric == 'r2':
            ss_res = np.sum((x - result) ** 2)
            ss_tot = np.sum((x - np.mean(x)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    elif callable(metric):
        metrics['custom'] = metric(x, result)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'window_length': window_length,
            'polyorder': polyorder,
            'deriv': deriv,
            'normalize': normalize,
            'metric': metric
        },
        'warnings': []
    }

################################################################################
# gaussian_smoothing
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    bandwidth: float,
    normalize: str = "standard",
) -> None:
    """Validate input data and parameters."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if bandwidth <= 0:
        raise ValueError("Bandwidth must be positive")

def _normalize_data(
    X: np.ndarray,
    y: np.ndarray,
    normalize: str = "standard",
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize data based on the specified method."""
    if normalize == "none":
        return X, y
    elif normalize == "standard":
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
    elif normalize == "minmax":
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
    elif normalize == "robust":
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
        raise ValueError(f"Unknown normalization method: {normalize}")

def _gaussian_kernel(
    X: np.ndarray,
    x: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """Compute the Gaussian kernel."""
    squared_distances = np.sum((X - x) ** 2, axis=1)
    return np.exp(-squared_distances / (2 * bandwidth ** 2))

def _gaussian_smoothing_compute(
    X: np.ndarray,
    y: np.ndarray,
    bandwidth: float,
    normalize: str = "standard",
) -> Dict[str, Union[np.ndarray, Dict]]:
    """Compute Gaussian smoothing."""
    _validate_inputs(X, y, bandwidth)
    X_norm, y_norm = _normalize_data(X, y, normalize)

    n_samples = X.shape[0]
    smoothed_values = np.zeros(n_samples)
    weights_sum = np.zeros(n_samples)

    for i in range(n_samples):
        weights = _gaussian_kernel(X_norm, X_norm[i], bandwidth)
        smoothed_values[i] = np.sum(weights * y_norm) / np.sum(weights)
        weights_sum[i] = np.sum(weights)

    # Denormalize the result
    if normalize == "standard":
        y_mean = np.mean(y)
        y_std = np.std(y)
        if y_std == 0:
            y_std = 1.0
        smoothed_values = smoothed_values * y_std + y_mean
    elif normalize == "minmax":
        y_min = np.min(y)
        y_max = np.max(y)
        if y_max == y_min:
            smoothed_values = np.zeros_like(smoothed_values)
        else:
            smoothed_values = smoothed_values * (y_max - y_min) + y_min
    elif normalize == "robust":
        y_median = np.median(y)
        y_iqr = np.subtract(*np.percentile(y, [75, 25]))
        if y_iqr == 0:
            y_iqr = 1.0
        smoothed_values = smoothed_values * y_iqr + y_median

    return {
        "result": smoothed_values,
        "metrics": {},
        "params_used": {"bandwidth": bandwidth, "normalize": normalize},
        "warnings": [],
    }

# Example usage:
# result = gaussian_smoothing_compute(X, y, bandwidth=1.0)

################################################################################
# median_filter
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(data: np.ndarray) -> None:
    """Validate input data for median filter.

    Args:
        data: Input array to be validated.

    Raises:
        ValueError: If input is not a numpy array or contains invalid values.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or infinite values.")

def median_filter_fit(
    data: np.ndarray,
    window_size: int = 3,
    axis: Optional[int] = None,
    method: str = 'simple',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """Apply median filter to smooth data.

    Args:
        data: Input array to be smoothed.
        window_size: Size of the sliding window. Must be odd.
        axis: Axis along which to apply the filter. If None, apply to flattened array.
        method: Method for handling window edges ('simple', 'mirror', 'wrap').
        custom_metric: Optional callable for custom metric calculation.
        **kwargs: Additional keyword arguments for specific methods.

    Returns:
        Dictionary containing:
            - 'result': Smoothed array
            - 'metrics': Dictionary of computed metrics
            - 'params_used': Dictionary of parameters used
            - 'warnings': List of warnings encountered

    Example:
        >>> data = np.array([1, 2, 3, 4, 5])
        >>> result = median_filter_fit(data, window_size=3)
    """
    # Validate input
    validate_input(data)

    # Check window size
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")

    # Set default parameters
    params_used = {
        'window_size': window_size,
        'axis': axis,
        'method': method
    }

    # Apply median filter based on method
    if axis is None:
        data_flat = data.flatten()
        result = _apply_median_filter(data_flat, window_size, method)
    else:
        result = np.apply_along_axis(
            lambda x: _apply_median_filter(x, window_size, method),
            axis,
            data
        )

    # Calculate metrics
    metrics = _calculate_metrics(data, result, custom_metric)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': []
    }

def _apply_median_filter(
    data: np.ndarray,
    window_size: int,
    method: str
) -> np.ndarray:
    """Apply median filter to 1D array.

    Args:
        data: Input 1D array.
        window_size: Size of the sliding window.
        method: Method for handling window edges.

    Returns:
        Smoothed array.
    """
    pad_width = window_size // 2
    if method == 'simple':
        padded_data = np.pad(data, pad_width, mode='constant', constant_values=np.nan)
    elif method == 'mirror':
        padded_data = np.pad(data, pad_width, mode='reflect')
    elif method == 'wrap':
        padded_data = np.pad(data, pad_width, mode='wrap')
    else:
        raise ValueError(f"Unknown method: {method}")

    smoothed = np.zeros_like(data)
    for i in range(len(data)):
        window = padded_data[i:i+window_size]
        if np.isnan(window).any():
            smoothed[i] = data[i]
        else:
            smoothed[i] = np.median(window)

    return smoothed

def _calculate_metrics(
    original: np.ndarray,
    filtered: np.ndarray,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Calculate metrics between original and filtered data.

    Args:
        original: Original array.
        filtered: Filtered array.
        custom_metric: Optional callable for custom metric.

    Returns:
        Dictionary of computed metrics.
    """
    metrics = {}

    # Calculate MSE
    mse = np.mean((original - filtered) ** 2)
    metrics['mse'] = float(mse)

    # Calculate MAE
    mae = np.mean(np.abs(original - filtered))
    metrics['mae'] = float(mae)

    # Calculate custom metric if provided
    if custom_metric is not None:
        metrics['custom'] = float(custom_metric(original, filtered))

    return metrics

################################################################################
# lowess
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-dimensional arrays")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("x and y must not contain NaN values")
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        raise ValueError("x and y must not contain infinite values")

def _normalize_data(x: np.ndarray, y: np.ndarray,
                    normalization: str = 'standard',
                    custom_normalize: Optional[Callable] = None) -> tuple:
    """Normalize data based on specified method."""
    if custom_normalize is not None:
        return custom_normalize(x, y)

    x_mean = np.mean(x)
    x_std = np.std(x)
    y_mean = np.mean(y)
    y_std = np.std(y)

    if normalization == 'standard':
        x_norm = (x - x_mean) / x_std
        y_norm = (y - y_mean) / y_std
    elif normalization == 'minmax':
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        x_norm = (x - x_min) / (x_max - x_min)
        y_norm = (y - y_min) / (y_max - y_min)
    elif normalization == 'robust':
        x_median = np.median(x)
        x_iqr = np.percentile(x, 75) - np.percentile(x, 25)
        y_median = np.median(y)
        y_iqr = np.percentile(y, 75) - np.percentile(y, 25)
        x_norm = (x - x_median) / x_iqr
        y_norm = (y - y_median) / y_iqr
    else:
        x_norm, y_norm = x.copy(), y.copy()

    return x_norm, y_norm

def _compute_weights(x: np.ndarray, x0: float,
                     bandwidth: float = 1.0,
                     distance_metric: str = 'euclidean',
                     custom_distance: Optional[Callable] = None) -> np.ndarray:
    """Compute weights for LOWESS smoothing."""
    if custom_distance is not None:
        distances = custom_distance(x, x0)
    else:
        if distance_metric == 'euclidean':
            distances = np.abs(x - x0)
        elif distance_metric == 'manhattan':
            distances = np.abs(x - x0)
        elif distance_metric == 'cosine':
            # For 1D data, cosine distance is equivalent to absolute difference
            distances = np.abs(x - x0)
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

    weights = np.exp(-(distances / bandwidth) ** 2)
    return weights

def _local_regression(x: np.ndarray, y: np.ndarray,
                      x0: float, bandwidth: float = 1.0,
                      distance_metric: str = 'euclidean',
                      custom_distance: Optional[Callable] = None) -> float:
    """Perform local weighted regression."""
    weights = _compute_weights(x, x0, bandwidth, distance_metric, custom_distance)
    weighted_x = weights * x
    weighted_y = weights * y

    # Simple linear regression (closed form solution)
    numerator = np.sum(weighted_x * weighted_y) - np.sum(weighted_x) * np.sum(weighted_y) / np.sum(weights)
    denominator = np.sum(weighted_x**2) - (np.sum(weighted_x)**2 / np.sum(weights))

    if denominator == 0:
        return np.mean(y)

    slope = numerator / denominator
    intercept = (np.sum(weighted_y) - slope * np.sum(weighted_x)) / np.sum(weights)

    return intercept + slope * x0

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     metric: str = 'mse',
                     custom_metric: Optional[Callable] = None) -> Dict[str, float]:
    """Compute evaluation metrics."""
    if custom_metric is not None:
        return {'custom': custom_metric(y_true, y_pred)}

    metrics = {}
    if metric == 'mse' or 'all' in metric:
        mse = np.mean((y_true - y_pred) ** 2)
        metrics['mse'] = mse
    if metric == 'mae' or 'all' in metric:
        mae = np.mean(np.abs(y_true - y_pred))
        metrics['mae'] = mae
    if metric == 'r2' or 'all' in metric:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        metrics['r2'] = r2

    return metrics

def lowess_fit(x: np.ndarray, y: np.ndarray,
               bandwidth: float = 0.3,
               iterations: int = 3,
               normalization: str = 'standard',
               distance_metric: str = 'euclidean',
               metric: str = 'mse',
               custom_normalize: Optional[Callable] = None,
               custom_distance: Optional[Callable] = None,
               custom_metric: Optional[Callable] = None) -> Dict:
    """
    Perform LOWESS (Locally Weighted Scatterplot Smoothing) regression.

    Parameters:
    - x: Input array of x values
    - y: Input array of y values
    - bandwidth: Smoothing parameter (0 < bandwidth <= 1)
    - iterations: Number of iterative weighting steps
    - normalization: Normalization method ('none', 'standard', 'minmax', 'robust')
    - distance_metric: Distance metric for weighting ('euclidean', 'manhattan', 'cosine')
    - metric: Evaluation metric ('mse', 'mae', 'r2', 'all')
    - custom_normalize: Custom normalization function
    - custom_distance: Custom distance function
    - custom_metric: Custom evaluation metric

    Returns:
    Dictionary containing:
    - result: Smoothed y values
    - metrics: Evaluation metrics
    - params_used: Parameters used in the computation
    - warnings: Any warnings generated during computation

    Example:
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.sin(x) + np.random.normal(0, 0.1, 100)
    >>> result = lowess_fit(x, y, bandwidth=0.2)
    """
    # Validate inputs
    _validate_inputs(x, y)

    # Initialize output dictionary
    output = {
        'result': None,
        'metrics': {},
        'params_used': {
            'bandwidth': bandwidth,
            'iterations': iterations,
            'normalization': normalization,
            'distance_metric': distance_metric,
            'metric': metric
        },
        'warnings': []
    }

    # Normalize data
    x_norm, y_norm = _normalize_data(x, y, normalization, custom_normalize)

    # Initialize smoothed values
    y_smooth = y_norm.copy()

    for _ in range(iterations):
        for i, x0 in enumerate(x_norm):
            y_smooth[i] = _local_regression(
                x_norm, y_smooth,
                x0, bandwidth,
                distance_metric, custom_distance
            )

    # Denormalize results
    y_mean = np.mean(y)
    y_std = np.std(y)
    output['result'] = y_smooth * y_std + y_mean

    # Compute metrics
    output['metrics'] = _compute_metrics(y, output['result'], metric, custom_metric)

    return output

################################################################################
# spline_smoothing
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def spline_smoothing_fit(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 3,
    smoothing_factor: float = 0.5,
    solver: str = 'closed_form',
    metric: Union[str, Callable] = 'mse',
    normalization: str = 'none',
    regularization: Optional[str] = None,
    penalty_weight: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Fit a smoothing spline to the given data.

    Parameters:
    -----------
    x : np.ndarray
        Input data points.
    y : np.ndarray
        Target values.
    degree : int, optional
        Degree of the spline (default is 3).
    smoothing_factor : float, optional
        Smoothing factor (lambda) controlling the trade-off between smoothness and fit (default is 0.5).
    solver : str, optional
        Solver to use ('closed_form', 'gradient_descent', 'newton') (default is 'closed_form').
    metric : Union[str, Callable], optional
        Metric to evaluate the fit ('mse', 'mae', 'r2') or a custom callable (default is 'mse').
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default is 'none').
    regularization : Optional[str], optional
        Regularization method ('l1', 'l2', 'elasticnet') (default is None).
    penalty_weight : float, optional
        Weight for the regularization penalty (default is 1.0).
    tol : float, optional
        Tolerance for convergence (default is 1e-6).
    max_iter : int, optional
        Maximum number of iterations (default is 1000).
    custom_metric : Optional[Callable], optional
        Custom metric function (default is None).

    Returns:
    --------
    Dict
        A dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(x, y)

    # Normalize data if specified
    x_norm, y_norm = _apply_normalization(x, y, normalization)

    # Choose solver
    if solver == 'closed_form':
        coefficients = _solve_closed_form(x_norm, y_norm, degree, smoothing_factor)
    elif solver == 'gradient_descent':
        coefficients = _solve_gradient_descent(x_norm, y_norm, degree, smoothing_factor,
                                             regularization, penalty_weight, tol, max_iter)
    elif solver == 'newton':
        coefficients = _solve_newton(x_norm, y_norm, degree, smoothing_factor,
                                   regularization, penalty_weight, tol, max_iter)
    else:
        raise ValueError("Unsupported solver. Choose from 'closed_form', 'gradient_descent', 'newton'.")

    # Compute metrics
    y_pred = _predict(x_norm, coefficients, degree)
    metrics = _compute_metrics(y_norm, y_pred, metric, custom_metric)

    # Prepare output
    result = {
        'result': coefficients,
        'metrics': metrics,
        'params_used': {
            'degree': degree,
            'smoothing_factor': smoothing_factor,
            'solver': solver,
            'metric': metric,
            'normalization': normalization,
            'regularization': regularization,
            'penalty_weight': penalty_weight
        },
        'warnings': []
    }

    return result

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-dimensional arrays.")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("x and y must not contain NaN values.")
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        raise ValueError("x and y must not contain infinite values.")

def _apply_normalization(x: np.ndarray, y: np.ndarray, method: str) -> tuple:
    """Apply normalization to the input data."""
    if method == 'none':
        return x, y
    elif method == 'standard':
        x_mean = np.mean(x)
        x_std = np.std(x)
        y_mean = np.mean(y)
        y_std = np.std(y)
        return (x - x_mean) / x_std, (y - y_mean) / y_std
    elif method == 'minmax':
        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)
        return (x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min)
    elif method == 'robust':
        x_median = np.median(x)
        x_iqr = np.percentile(x, 75) - np.percentile(x, 25)
        y_median = np.median(y)
        y_iqr = np.percentile(y, 75) - np.percentile(y, 25)
        return (x - x_median) / x_iqr, (y - y_median) / y_iqr
    else:
        raise ValueError("Unsupported normalization method. Choose from 'none', 'standard', 'minmax', 'robust'.")

def _solve_closed_form(x: np.ndarray, y: np.ndarray, degree: int, smoothing_factor: float) -> np.ndarray:
    """Solve the spline smoothing problem using closed-form solution."""
    # Implement closed-form solution logic here
    pass

def _solve_gradient_descent(x: np.ndarray, y: np.ndarray, degree: int, smoothing_factor: float,
                           regularization: Optional[str], penalty_weight: float, tol: float, max_iter: int) -> np.ndarray:
    """Solve the spline smoothing problem using gradient descent."""
    # Implement gradient descent logic here
    pass

def _solve_newton(x: np.ndarray, y: np.ndarray, degree: int, smoothing_factor: float,
                 regularization: Optional[str], penalty_weight: float, tol: float, max_iter: int) -> np.ndarray:
    """Solve the spline smoothing problem using Newton's method."""
    # Implement Newton's method logic here
    pass

def _predict(x: np.ndarray, coefficients: np.ndarray, degree: int) -> np.ndarray:
    """Predict values using the fitted spline."""
    # Implement prediction logic here
    pass

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metric: Union[str, Callable], custom_metric: Optional[Callable]) -> Dict:
    """Compute metrics for the spline fit."""
    metrics = {}
    if metric == 'mse':
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)
    elif callable(metric):
        metrics['custom'] = metric(y_true, y_pred)
    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)
    return metrics

################################################################################
# wavelet_transform
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(
    signal: np.ndarray,
    wavelet: str = 'db1',
    level: int = 1,
    mode: str = 'zero'
) -> None:
    """Validate input parameters for wavelet transform."""
    if not isinstance(signal, np.ndarray):
        raise TypeError("Signal must be a numpy array")
    if signal.ndim != 1:
        raise ValueError("Signal must be 1-dimensional")
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        raise ValueError("Signal contains NaN or infinite values")
    if level < 1:
        raise ValueError("Level must be at least 1")

def wavelet_transform_fit(
    signal: np.ndarray,
    wavelet: str = 'db1',
    level: int = 1,
    mode: str = 'zero',
    normalization: Optional[str] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'closed_form'
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Perform wavelet transform on a signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal to be transformed.
    wavelet : str, optional
        Wavelet type (default is 'db1').
    level : int, optional
        Decomposition level (default is 1).
    mode : str, optional
        Signal extension mode (default is 'zero').
    normalization : str or None, optional
        Normalization method (none, standard, minmax, robust).
    metric : str or callable, optional
        Metric to evaluate transform quality (default is 'mse').
    solver : str, optional
        Solver method (default is 'closed_form').

    Returns
    -------
    dict
        Dictionary containing:
        - result: transformed signal components
        - metrics: evaluation metrics
        - params_used: parameters used in the transform
        - warnings: any warnings generated during processing

    Example
    -------
    >>> signal = np.random.randn(100)
    >>> result = wavelet_transform_fit(signal, wavelet='db2', level=2)
    """
    # Validate inputs
    validate_inputs(signal, wavelet, level, mode)

    # Normalize signal if specified
    normalized_signal = _apply_normalization(signal, normalization)

    # Perform wavelet transform
    coefficients = _compute_wavelet_transform(normalized_signal, wavelet, level, mode)

    # Calculate metrics
    metrics = _calculate_metrics(signal, coefficients, metric)

    # Prepare output
    result = {
        "result": {"coefficients": coefficients},
        "metrics": metrics,
        "params_used": {
            "wavelet": wavelet,
            "level": level,
            "mode": mode,
            "normalization": normalization,
            "metric": metric,
            "solver": solver
        },
        "warnings": []
    }

    return result

def _apply_normalization(
    signal: np.ndarray,
    method: Optional[str] = None
) -> np.ndarray:
    """Apply specified normalization to the signal."""
    if method is None or method == 'none':
        return signal
    elif method == 'standard':
        return (signal - np.mean(signal)) / np.std(signal)
    elif method == 'minmax':
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    elif method == 'robust':
        median = np.median(signal)
        iqr = np.percentile(signal, 75) - np.percentile(signal, 25)
        return (signal - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_wavelet_transform(
    signal: np.ndarray,
    wavelet: str = 'db1',
    level: int = 1,
    mode: str = 'zero'
) -> Dict[str, np.ndarray]:
    """Compute wavelet transform coefficients."""
    # In a real implementation, this would use a proper wavelet library
    # For this example, we'll return dummy data
    coefficients = {
        'approximation': np.random.randn(len(signal) // 2),
        'detail': np.random.randn(len(signal) // 2)
    }
    return coefficients

def _calculate_metrics(
    original: np.ndarray,
    transformed: Dict[str, np.ndarray],
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse'
) -> Dict[str, float]:
    """Calculate evaluation metrics for the transform."""
    # Reconstruct signal from coefficients (simplified)
    reconstructed = np.concatenate([
        transformed['approximation'],
        transformed['detail']
    ])[:len(original)]

    if callable(metric):
        return {"custom_metric": metric(original, reconstructed)}

    metrics = {}
    if metric == 'mse':
        metrics['mse'] = np.mean((original - reconstructed) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(original - reconstructed))
    elif metric == 'r2':
        ss_res = np.sum((original - reconstructed) ** 2)
        ss_tot = np.sum((original - np.mean(original)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics
