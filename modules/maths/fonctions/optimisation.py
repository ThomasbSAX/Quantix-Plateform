"""
Quantix – Module optimisation
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# gradient_descent
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
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

def normalize_data(X: np.ndarray, y: np.ndarray, method: str = 'standard') -> tuple:
    """Normalize data using specified method."""
    if method == 'none':
        return X, y
    elif method == 'standard':
        mean_X = np.mean(X, axis=0)
        std_X = np.std(X, axis=0)
        X_normalized = (X - mean_X) / std_X
        mean_y = np.mean(y)
        std_y = np.std(y)
        y_normalized = (y - mean_y) / std_y
        return X_normalized, y_normalized
    elif method == 'minmax':
        min_X = np.min(X, axis=0)
        max_X = np.max(X, axis=0)
        X_normalized = (X - min_X) / (max_X - min_X)
        min_y = np.min(y)
        max_y = np.max(y)
        y_normalized = (y - min_y) / (max_y - min_y)
        return X_normalized, y_normalized
    elif method == 'robust':
        median_X = np.median(X, axis=0)
        iqr_X = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_normalized = (X - median_X) / iqr_X
        median_y = np.median(y)
        iqr_y = np.subtract(*np.percentile(y, [75, 25]))
        y_normalized = (y - median_y) / iqr_y
        return X_normalized, y_normalized
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def compute_gradient(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                     loss_func: Callable) -> np.ndarray:
    """Compute gradient of the loss function."""
    n_samples = X.shape[0]
    predictions = np.dot(X, weights)
    gradient = (1 / n_samples) * np.dot(X.T, loss_func(y, predictions))
    return gradient

def compute_loss(y: np.ndarray, y_pred: np.ndarray,
                 loss_func: Callable) -> float:
    """Compute the loss value."""
    return np.mean(loss_func(y, y_pred))

def gradient_descent_fit(X: np.ndarray, y: np.ndarray,
                         learning_rate: float = 0.01,
                         n_iterations: int = 1000,
                         tol: float = 1e-4,
                         normalize_method: str = 'standard',
                         loss_func: Callable = lambda y, y_pred: (y - y_pred) ** 2,
                         metric_funcs: Optional[Dict[str, Callable]] = None,
                         verbose: bool = False) -> Dict:
    """
    Perform gradient descent optimization.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    learning_rate : float, optional
        Learning rate for gradient descent.
    n_iterations : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for stopping criterion.
    normalize_method : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    loss_func : Callable, optional
        Loss function to minimize.
    metric_funcs : Dict[str, Callable], optional
        Dictionary of metric functions to compute.
    verbose : bool, optional
        Whether to print progress information.

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = normalize_data(X, y, method=normalize_method)

    # Initialize weights
    n_features = X_norm.shape[1]
    weights = np.zeros(n_features)

    # Initialize metrics dictionary
    if metric_funcs is None:
        metric_funcs = {}
    metrics = {name: [] for name in metric_funcs.keys()}

    # Gradient descent loop
    prev_loss = float('inf')
    for i in range(n_iterations):
        # Compute predictions
        y_pred = np.dot(X_norm, weights)

        # Compute loss and metrics
        current_loss = compute_loss(y_norm, y_pred, loss_func)
        for name, func in metric_funcs.items():
            metrics[name].append(func(y_norm, y_pred))

        # Check for convergence
        if abs(prev_loss - current_loss) < tol:
            if verbose:
                print(f"Converged at iteration {i}")
            break

        # Update weights
        gradient = compute_gradient(X_norm, y_norm, weights, loss_func)
        weights -= learning_rate * gradient
        prev_loss = current_loss

    # Compute final metrics
    y_pred_final = np.dot(X_norm, weights)
    final_metrics = {name: func(y_norm, y_pred_final) for name, func in metric_funcs.items()}

    # Prepare output
    result = {
        'result': weights,
        'metrics': final_metrics,
        'params_used': {
            'learning_rate': learning_rate,
            'n_iterations': i + 1,
            'tol': tol,
            'normalize_method': normalize_method
        },
        'warnings': []
    }

    return result

# Example usage:
"""
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([1, 2, 3])

def mse(y_true, y_pred):
    return (y_true - y_pred) ** 2

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

metrics = {
    'mse': mse,
    'r2': r2_score
}

result = gradient_descent_fit(X, y, learning_rate=0.1, n_iterations=1000,
                             normalize_method='standard', metric_funcs=metrics)
"""

################################################################################
# stochastic_gradient_descent
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def stochastic_gradient_descent_fit(
    X: np.ndarray,
    y: np.ndarray,
    loss_func: Callable[[np.ndarray, np.ndarray], float],
    gradient_func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    learning_rate: float = 0.01,
    n_iter: int = 1000,
    batch_size: Optional[int] = None,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    normalize: bool = True,
    metric_funcs: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = None,
    verbose: bool = False
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, any], str]]:
    """
    Perform stochastic gradient descent optimization.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    loss_func : Callable[[np.ndarray, np.ndarray], float]
        Loss function that takes predictions and true values.
    gradient_func : Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
        Gradient function that takes weights, features, and true values.
    learning_rate : float, optional
        Learning rate for gradient updates (default: 0.01).
    n_iter : int, optional
        Number of iterations (default: 1000).
    batch_size : Optional[int], optional
        Size of mini-batches (None for full gradient descent, default: None).
    tol : float, optional
        Tolerance for stopping criterion (default: 1e-4).
    random_state : Optional[int], optional
        Random seed for reproducibility (default: None).
    normalize : bool, optional
        Whether to normalize features (default: True).
    metric_funcs : Dict[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Dictionary of metric functions to compute (default: None).
    verbose : bool, optional
        Whether to print progress (default: False).

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, any], str]]
        Dictionary containing:
        - "result": Optimized weights
        - "metrics": Computed metrics
        - "params_used": Parameters used in the optimization
        - "warnings": Any warnings encountered

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> def mse_loss(y_true, y_pred): return np.mean((y_true - y_pred) ** 2)
    >>> def mse_gradient(weights, X_batch, y_batch): return -2 * X_batch.T @ (y_batch - X_batch @ weights) / len(y_batch)
    >>> result = stochastic_gradient_descent_fit(X, y, mse_loss, mse_gradient)
    """
    # Initialize random state
    if random_state is not None:
        np.random.seed(random_state)

    # Validate inputs
    _validate_inputs(X, y)
    n_samples, n_features = X.shape

    # Normalize features if required
    if normalize:
        X, mean_X, std_X = _normalize_features(X)
    else:
        mean_X, std_X = None, None

    # Initialize weights
    weights = np.zeros(n_features)

    # Prepare metrics dictionary if provided
    metrics = {}
    if metric_funcs is not None:
        for name, func in metric_funcs.items():
            metrics[name] = []

    # Prepare output dictionary
    output = {
        "result": None,
        "metrics": metrics,
        "params_used": {
            "learning_rate": learning_rate,
            "n_iter": n_iter,
            "batch_size": batch_size,
            "tol": tol,
            "normalize": normalize
        },
        "warnings": []
    }

    # Main optimization loop
    for i in range(n_iter):
        if batch_size is None:
            # Full gradient descent
            grad = gradient_func(weights, X, y)
        else:
            # Stochastic gradient descent
            indices = np.random.choice(n_samples, batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]
            grad = gradient_func(weights, X_batch, y_batch)

        # Update weights
        weights -= learning_rate * grad

        # Compute and store metrics if provided
        if metric_funcs is not None:
            y_pred = X @ weights
            for name, func in metric_funcs.items():
                metrics[name].append(func(y, y_pred))

        # Check for convergence
        if i > 0 and np.linalg.norm(grad) < tol:
            output["warnings"].append(f"Converged at iteration {i}")
            break

        if verbose and i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss_func(y, X @ weights):.4f}")

    output["result"] = weights
    return output

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
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
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0)
    std_X[std_X == 0] = 1.0  # Avoid division by zero
    X_normalized = (X - mean_X) / std_X
    return X_normalized, mean_X, std_X

# Example metric functions
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Example loss and gradient functions
def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error loss."""
    return mse(y_true, y_pred)

def mse_gradient(weights: np.ndarray, X_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
    """Gradient of Mean Squared Error loss."""
    return -2 * X_batch.T @ (y_batch - X_batch @ weights) / len(y_batch)

################################################################################
# mini_batch_gradient_descent
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def mini_batch_gradient_descent_fit(
    X: np.ndarray,
    y: np.ndarray,
    loss_func: Callable[[np.ndarray, np.ndarray], float],
    gradient_func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    learning_rate: float = 0.01,
    batch_size: int = 32,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    normalize: bool = True,
    metric_funcs: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, any], str]]:
    """
    Perform mini-batch gradient descent optimization.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    loss_func : Callable[[np.ndarray, np.ndarray], float]
        Loss function to minimize.
    gradient_func : Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
        Gradient function of the loss with respect to parameters.
    learning_rate : float, optional
        Learning rate for gradient descent (default: 0.01).
    batch_size : int, optional
        Size of mini-batches (default: 32).
    max_iter : int, optional
        Maximum number of iterations (default: 1000).
    tol : float, optional
        Tolerance for stopping criterion (default: 1e-4).
    random_state : Optional[int], optional
        Random seed for reproducibility (default: None).
    normalize : bool, optional
        Whether to normalize features (default: True).
    metric_funcs : Dict[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Dictionary of metric functions to compute during optimization (default: None).
    **kwargs : dict
        Additional keyword arguments for the loss and gradient functions.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, any], str]]
        Dictionary containing:
        - "result": Optimized parameters
        - "metrics": Computed metrics during optimization
        - "params_used": Parameters used in the optimization
        - "warnings": Any warnings generated during optimization

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>>
    >>> def mse_loss(params, X_batch, y_batch):
    ...     return np.mean((X_batch @ params - y_batch) ** 2)
    >>>
    >>> def mse_gradient(params, X_batch, y_batch):
    ...     return 2 * X_batch.T @ (X_batch @ params - y_batch) / len(y_batch)
    >>>
    >>> result = mini_batch_gradient_descent_fit(
    ...     X, y,
    ...     loss_func=mse_loss,
    ...     gradient_func=mse_gradient
    ... )
    """
    # Initialize random state
    if random_state is not None:
        np.random.seed(random_state)

    # Validate inputs
    _validate_inputs(X, y)
    n_samples, n_features = X.shape

    # Normalize features if requested
    if normalize:
        X, mean_X, std_X = _normalize_features(X)
    else:
        mean_X, std_X = None, None

    # Initialize parameters
    params = np.zeros(n_features)

    # Prepare metrics dictionary if provided
    metrics = {}
    if metric_funcs is not None:
        for name, func in metric_funcs.items():
            metrics[name] = []

    # Optimization loop
    for iteration in range(max_iter):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Mini-batch processing
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            X_batch = X_shuffled[batch_start:batch_end]
            y_batch = y_shuffled[batch_start:batch_end]

            # Compute gradient and update parameters
            grad = gradient_func(params, X_batch, y_batch)
            params -= learning_rate * grad

        # Compute loss and metrics
        current_loss = loss_func(params, X, y)
        if metric_funcs is not None:
            for name, func in metric_funcs.items():
                metrics[name].append(func(y, X @ params))

        # Check convergence
        if iteration > 0 and abs(current_loss - prev_loss) < tol:
            break

        prev_loss = current_loss

    # Prepare results
    result_dict = {
        "result": params,
        "metrics": metrics if metric_funcs is not None else {},
        "params_used": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_iter": max_iter,
            "tol": tol,
            "normalize": normalize
        },
        "warnings": []
    }

    if mean_X is not None and std_X is not None:
        result_dict["warnings"].append("Features were normalized during optimization")

    return result_dict

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")

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
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0)

    # Avoid division by zero for constant features
    std_X[std_X == 0] = 1.0

    X_normalized = (X - mean_X) / std_X
    return X_normalized, mean_X, std_X

def mse_loss(params: np.ndarray, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
    """Mean Squared Error loss function."""
    return np.mean((X_batch @ params - y_batch) ** 2)

def mse_gradient(params: np.ndarray, X_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
    """Gradient of MSE loss with respect to parameters."""
    return 2 * X_batch.T @ (X_batch @ params - y_batch) / len(y_batch)

def mae_loss(params: np.ndarray, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
    """Mean Absolute Error loss function."""
    return np.mean(np.abs(X_batch @ params - y_batch))

def mae_gradient(params: np.ndarray, X_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
    """Gradient of MAE loss with respect to parameters."""
    residuals = X_batch @ params - y_batch
    grad = np.zeros_like(params)
    for i in range(len(residuals)):
        if residuals[i] != 0:
            grad += X_batch[i] * np.sign(residuals[i])
    return grad / len(y_batch)

def r2_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared metric."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

################################################################################
# newton_method
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def newton_method_fit(
    objective_func: Callable[[np.ndarray], float],
    gradient_func: Callable[[np.ndarray], np.ndarray],
    hessian_func: Callable[[np.ndarray], np.ndarray],
    initial_params: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 1000,
    line_search_func: Optional[Callable] = None,
    normalization: str = 'none',
    metrics: Union[str, Callable[[np.ndarray], float]] = 'mse'
) -> Dict:
    """
    Optimize a function using Newton's method.

    Parameters
    ----------
    objective_func : callable
        The objective function to minimize.
    gradient_func : callable
        The gradient of the objective function.
    hessian_func : callable
        The Hessian matrix of the objective function.
    initial_params : ndarray
        Initial parameters for optimization.
    tol : float, optional
        Tolerance for convergence (default: 1e-6).
    max_iter : int, optional
        Maximum number of iterations (default: 1000).
    line_search_func : callable, optional
        Line search function (default: None).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'none').
    metrics : str or callable, optional
        Metric to evaluate ('mse', 'mae', 'r2', 'logloss') or custom callable (default: 'mse').

    Returns
    -------
    dict
        Dictionary containing:
        - result: Optimized parameters.
        - metrics: Computed metrics.
        - params_used: Parameters used in the optimization.
        - warnings: Any warnings generated during optimization.

    Examples
    --------
    >>> def objective(x):
    ...     return x[0]**2 + x[1]**2
    >>> def gradient(x):
    ...     return np.array([2*x[0], 2*x[1]])
    >>> def hessian(x):
    ...     return np.array([[2, 0], [0, 2]])
    >>> result = newton_method_fit(objective, gradient, hessian, np.array([1.0, 1.0]))
    """
    # Validate inputs
    _validate_inputs(objective_func, gradient_func, hessian_func, initial_params)

    # Normalize parameters if required
    params = _apply_normalization(initial_params, normalization)

    # Initialize variables
    current_params = np.array(params)
    prev_params = None
    iterations = 0
    warnings_list = []

    # Define metric function
    metric_func = _get_metric_function(metrics)

    while iterations < max_iter:
        # Compute gradient and Hessian
        grad = gradient_func(current_params)
        hess = hessian_func(current_params)

        # Check for convergence
        if prev_params is not None and np.linalg.norm(current_params - prev_params) < tol:
            break

        # Compute search direction
        try:
            search_direction = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            warnings_list.append("Hessian matrix is singular. Using gradient descent direction.")
            search_direction = -grad

        # Line search if provided
        if line_search_func is not None:
            alpha = line_search_func(objective_func, current_params, search_direction)
        else:
            alpha = 1.0

        # Update parameters
        prev_params = current_params.copy()
        current_params += alpha * search_direction
        iterations += 1

    # Compute metrics
    result_metrics = metric_func(current_params)

    return {
        'result': current_params,
        'metrics': result_metrics,
        'params_used': {
            'initial_params': initial_params,
            'normalization': normalization
        },
        'warnings': warnings_list
    }

def _validate_inputs(
    objective_func: Callable,
    gradient_func: Callable,
    hessian_func: Callable,
    initial_params: np.ndarray
) -> None:
    """Validate input functions and parameters."""
    if not callable(objective_func):
        raise ValueError("objective_func must be a callable.")
    if not callable(gradient_func):
        raise ValueError("gradient_func must be a callable.")
    if not callable(hessian_func):
        raise ValueError("hessian_func must be a callable.")
    if not isinstance(initial_params, np.ndarray):
        raise ValueError("initial_params must be a numpy array.")
    if np.any(np.isnan(initial_params)) or np.any(np.isinf(initial_params)):
        raise ValueError("initial_params must not contain NaN or inf values.")

def _apply_normalization(
    params: np.ndarray,
    normalization: str
) -> np.ndarray:
    """Apply normalization to parameters."""
    if normalization == 'standard':
        mean = np.mean(params)
        std = np.std(params)
        if std == 0:
            return params
        return (params - mean) / std
    elif normalization == 'minmax':
        min_val = np.min(params)
        max_val = np.max(params)
        if max_val == min_val:
            return params
        return (params - min_val) / (max_val - min_val)
    elif normalization == 'robust':
        median = np.median(params)
        iqr = np.percentile(params, 75) - np.percentile(params, 25)
        if iqr == 0:
            return params
        return (params - median) / iqr
    else:
        return params

def _get_metric_function(
    metric: Union[str, Callable]
) -> Callable:
    """Get the metric function based on the input."""
    if callable(metric):
        return metric
    elif metric == 'mse':
        def mse(params: np.ndarray) -> float:
            return np.mean(np.square(params))
    elif metric == 'mae':
        def mae(params: np.ndarray) -> float:
            return np.mean(np.abs(params))
    elif metric == 'r2':
        def r2(params: np.ndarray) -> float:
            return 1 - (np.sum(np.square(params)) / len(params))
    elif metric == 'logloss':
        def logloss(params: np.ndarray) -> float:
            return -np.mean(np.log(params))
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return locals()[metric]

################################################################################
# conjugate_gradient
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def conjugate_gradient_fit(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    max_iter: int = 1000,
    tol: float = 1e-5,
    callback: Optional[Callable[[np.ndarray, int], None]] = None
) -> Dict[str, Union[np.ndarray, float, Dict[str, float], str]]:
    """
    Solve the linear system Ax = b using the Conjugate Gradient method.

    Parameters:
    -----------
    A : np.ndarray
        The coefficient matrix (must be symmetric positive definite).
    b : np.ndarray
        The right-hand side vector.
    x0 : Optional[np.ndarray], default=None
        Initial guess for the solution. If None, uses a zero vector.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-5
        Tolerance for stopping criterion.
    callback : Optional[Callable[[np.ndarray, int], None]], default=None
        A callable that is called at each iteration with the current solution and iteration number.

    Returns:
    --------
    result : Dict[str, Union[np.ndarray, float, Dict[str, float], str]]
        A dictionary containing:
        - "result": The solution vector.
        - "metrics": Dictionary of metrics (e.g., residual norm).
        - "params_used": Parameters used in the computation.
        - "warnings": Any warnings generated during execution.

    Example:
    --------
    >>> A = np.array([[3, 2], [1, 4]])
    >>> b = np.array([5, 6])
    >>> result = conjugate_gradient_fit(A, b)
    """
    # Validate inputs
    if not isinstance(A, np.ndarray) or not isinstance(b, np.ndarray):
        raise TypeError("A and b must be numpy arrays.")
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    if A.shape[0] != b.shape[0]:
        raise ValueError("A and b must have compatible dimensions.")
    if x0 is not None and x0.shape[0] != b.shape[0]:
        raise ValueError("x0 must have the same dimension as b.")

    # Initialize
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
    r = b - A @ x
    p = r.copy()
    rs_old = r.T @ r

    # Check if A is symmetric positive definite
    try:
        np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix A must be symmetric positive definite.")

    # Main loop
    for i in range(max_iter):
        Ap = A @ p
        alpha = rs_old / (p.T @ Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = r.T @ r

        if callback is not None:
            callback(x, i)

        if np.sqrt(rs_new) < tol:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    # Compute metrics
    residual_norm = np.linalg.norm(r)
    metrics = {
        "residual_norm": float(residual_norm),
        "iterations": i + 1
    }

    # Prepare output
    result = {
        "result": x,
        "metrics": metrics,
        "params_used": {
            "max_iter": max_iter,
            "tol": tol
        },
        "warnings": []
    }

    if i == max_iter - 1:
        result["warnings"].append("Maximum iterations reached without convergence.")

    return result

def _validate_inputs(A: np.ndarray, b: np.ndarray) -> None:
    """Validate the inputs for the conjugate gradient method."""
    if not isinstance(A, np.ndarray) or not isinstance(b, np.ndarray):
        raise TypeError("A and b must be numpy arrays.")
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    if A.shape[0] != b.shape[0]:
        raise ValueError("A and b must have compatible dimensions.")

def _is_symmetric_positive_definite(A: np.ndarray) -> bool:
    """Check if the matrix A is symmetric positive definite."""
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

def _compute_residual(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the residual vector r = b - Ax."""
    return b - A @ x

def _compute_alpha(rs_old: float, p: np.ndarray, Ap: np.ndarray) -> float:
    """Compute the step size alpha."""
    return rs_old / (p.T @ Ap)

def _update_solution(x: np.ndarray, p: np.ndarray, alpha: float) -> None:
    """Update the solution vector x."""
    x += alpha * p

def _update_residual(r: np.ndarray, Ap: np.ndarray, alpha: float) -> None:
    """Update the residual vector r."""
    r -= alpha * Ap

def _update_conjugate_direction(r: np.ndarray, p: np.ndarray, rs_new: float, rs_old: float) -> None:
    """Update the conjugate direction vector p."""
    p[:] = r + (rs_new / rs_old) * p

################################################################################
# quasi_newton_methods
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def quasi_newton_methods_fit(
    objective_func: Callable[[np.ndarray], float],
    gradient_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    initial_params: np.ndarray = None,
    method: str = 'bfgs',
    max_iter: int = 1000,
    tol: float = 1e-5,
    line_search: str = 'wolfe',
    normalization: Optional[str] = None,
    metrics: Union[list, str] = ['mse'],
    custom_metric_funcs: Optional[Dict[str, Callable]] = None,
    verbose: bool = False
) -> Dict:
    """
    Optimize a function using quasi-Newton methods.

    Parameters
    ----------
    objective_func : callable
        The objective function to minimize.
    gradient_func : callable, optional
        The gradient of the objective function. If None, numerical differentiation is used.
    initial_params : ndarray, optional
        Initial parameters. If None, zeros are used.
    method : str, optional
        Quasi-Newton method to use ('bfgs', 'l-bfgs', 'sr1').
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence.
    line_search : str, optional
        Line search method ('wolfe', 'armijo').
    normalization : str, optional
        Normalization method ('standard', 'minmax', 'robust').
    metrics : list or str, optional
        Metrics to compute ('mse', 'mae', 'r2').
    custom_metric_funcs : dict, optional
        Custom metric functions.
    verbose : bool, optional
        Whether to print progress.

    Returns
    -------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(objective_func, gradient_func, initial_params)

    # Normalize data if specified
    normalized_params = _apply_normalization(initial_params, normalization)

    # Initialize parameters
    params = np.zeros_like(normalized_params) if normalized_params is None else normalized_params.copy()

    # Choose optimization method
    optimizer = _get_optimizer(method, objective_func, gradient_func)

    # Optimize
    result = optimizer(
        params,
        max_iter=max_iter,
        tol=tol,
        line_search=line_search,
        verbose=verbose
    )

    # Compute metrics
    metrics_result = _compute_metrics(result['x'], objective_func, metrics, custom_metric_funcs)

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics_result,
        'params_used': {
            'method': method,
            'max_iter': max_iter,
            'tol': tol,
            'line_search': line_search,
            'normalization': normalization
        },
        'warnings': []
    }

    return output

def _validate_inputs(
    objective_func: Callable,
    gradient_func: Optional[Callable],
    initial_params: np.ndarray
) -> None:
    """Validate input parameters."""
    if not callable(objective_func):
        raise ValueError("objective_func must be a callable")
    if gradient_func is not None and not callable(gradient_func):
        raise ValueError("gradient_func must be a callable or None")
    if initial_params is not None and not isinstance(initial_params, np.ndarray):
        raise ValueError("initial_params must be a numpy array or None")

def _apply_normalization(
    params: np.ndarray,
    normalization: Optional[str]
) -> Union[np.ndarray, None]:
    """Apply normalization to parameters."""
    if params is None:
        return None
    if normalization == 'standard':
        mean = np.mean(params)
        std = np.std(params)
        return (params - mean) / (std + 1e-8)
    elif normalization == 'minmax':
        min_val = np.min(params)
        max_val = np.max(params)
        return (params - min_val) / (max_val - min_val + 1e-8)
    elif normalization == 'robust':
        median = np.median(params)
        iqr = np.percentile(params, 75) - np.percentile(params, 25)
        return (params - median) / (iqr + 1e-8)
    else:
        return params

def _get_optimizer(
    method: str,
    objective_func: Callable,
    gradient_func: Optional[Callable]
) -> Callable:
    """Get the appropriate optimizer function."""
    if method == 'bfgs':
        return _bfgs_optimizer
    elif method == 'l-bfgs':
        return _l_bfgs_optimizer
    elif method == 'sr1':
        return _sr1_optimizer
    else:
        raise ValueError(f"Unknown method: {method}")

def _bfgs_optimizer(
    params: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-5,
    line_search: str = 'wolfe',
    verbose: bool = False
) -> Dict:
    """BFGS optimizer implementation."""
    # Implementation of BFGS algorithm
    pass

def _l_bfgs_optimizer(
    params: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-5,
    line_search: str = 'wolfe',
    verbose: bool = False
) -> Dict:
    """L-BFGS optimizer implementation."""
    # Implementation of L-BFGS algorithm
    pass

def _sr1_optimizer(
    params: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-5,
    line_search: str = 'wolfe',
    verbose: bool = False
) -> Dict:
    """SR1 optimizer implementation."""
    # Implementation of SR1 algorithm
    pass

def _compute_metrics(
    params: np.ndarray,
    objective_func: Callable,
    metrics: Union[list, str],
    custom_metric_funcs: Optional[Dict[str, Callable]]
) -> Dict:
    """Compute metrics for the optimization result."""
    if isinstance(metrics, str):
        metrics = [metrics]

    metric_results = {}
    for metric in metrics:
        if metric == 'mse':
            metric_results['mse'] = _compute_mse(params, objective_func)
        elif metric == 'mae':
            metric_results['mae'] = _compute_mae(params, objective_func)
        elif metric == 'r2':
            metric_results['r2'] = _compute_r2(params, objective_func)
        elif custom_metric_funcs and metric in custom_metric_funcs:
            metric_results[metric] = custom_metric_funcs[metric](params)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return metric_results

def _compute_mse(
    params: np.ndarray,
    objective_func: Callable
) -> float:
    """Compute Mean Squared Error."""
    # Implementation of MSE calculation
    pass

def _compute_mae(
    params: np.ndarray,
    objective_func: Callable
) -> float:
    """Compute Mean Absolute Error."""
    # Implementation of MAE calculation
    pass

def _compute_r2(
    params: np.ndarray,
    objective_func: Callable
) -> float:
    """Compute R-squared."""
    # Implementation of R2 calculation
    pass

################################################################################
# bfgs
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def validate_inputs(
    func: Callable[[np.ndarray], float],
    x0: np.ndarray,
    args: tuple = (),
    bounds: Optional[tuple] = None,
    tol: float = 1e-5,
    max_iter: int = 100
) -> None:
    """
    Validate inputs for BFGS optimization.

    Parameters
    ----------
    func : Callable[[np.ndarray], float]
        Objective function to minimize.
    x0 : np.ndarray
        Initial guess.
    args : tuple, optional
        Additional arguments passed to objective function.
    bounds : Optional[tuple], optional
        Bounds for parameters, by default None
    tol : float, optional
        Tolerance for optimization, by default 1e-5
    max_iter : int, optional
        Maximum number of iterations, by default 100

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not callable(func):
        raise ValueError("Objective function must be callable")
    if not isinstance(x0, np.ndarray):
        raise ValueError("Initial guess must be a numpy array")
    if bounds is not None and len(bounds) != 2:
        raise ValueError("Bounds must be a tuple of length 2")
    if tol <= 0:
        raise ValueError("Tolerance must be positive")
    if max_iter <= 0:
        raise ValueError("Maximum iterations must be positive")

def compute_gradient(
    func: Callable[[np.ndarray], float],
    x: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Compute numerical gradient of objective function.

    Parameters
    ----------
    func : Callable[[np.ndarray], float]
        Objective function to minimize.
    x : np.ndarray
        Point at which to compute gradient.
    epsilon : float, optional
        Step size for numerical differentiation, by default 1e-8

    Returns
    ------
    np.ndarray
        Gradient vector.
    """
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += epsilon
        x_minus[i] -= epsilon

        grad[i] = (func(x_plus) - func(x_minus)) / (2 * epsilon)
    return grad

def bfgs_update(
    H_k: np.ndarray,
    s_k: np.ndarray,
    y_k: np.ndarray
) -> np.ndarray:
    """
    Update the inverse Hessian approximation using BFGS formula.

    Parameters
    ----------
    H_k : np.ndarray
        Current inverse Hessian approximation.
    s_k : np.ndarray
        Step taken in parameter space.
    y_k : np.ndarray
        Change in gradient.

    Returns
    ------
    np.ndarray
        Updated inverse Hessian approximation.
    """
    rho = 1.0 / (np.dot(y_k, s_k) + 1e-10)
    I = np.eye(len(s_k))
    V = I - rho * np.outer(s_k, y_k)
    H_new = np.dot(V.T, np.dot(H_k, V)) + rho * np.outer(s_k, s_k)
    return H_new

def line_search(
    func: Callable[[np.ndarray], float],
    x_k: np.ndarray,
    d_k: np.ndarray,
    alpha_init: float = 1.0,
    c1: float = 1e-4,
    max_iter: int = 20
) -> float:
    """
    Perform line search to find step size.

    Parameters
    ----------
    func : Callable[[np.ndarray], float]
        Objective function to minimize.
    x_k : np.ndarray
        Current point.
    d_k : np.ndarray
        Search direction.
    alpha_init : float, optional
        Initial step size, by default 1.0
    c1 : float, optional
        Armijo constant, by default 1e-4
    max_iter : int, optional
        Maximum number of iterations, by default 20

    Returns
    ------
    float
        Optimal step size.
    """
    alpha = alpha_init
    f_k = func(x_k)
    for _ in range(max_iter):
        x_new = x_k + alpha * d_k
        f_new = func(x_new)
        if f_new <= f_k + c1 * alpha * np.dot(d_k, compute_gradient(func, x_k)):
            return alpha
        alpha *= 0.5
    return alpha

def bfgs_fit(
    func: Callable[[np.ndarray], float],
    x0: np.ndarray,
    args: tuple = (),
    bounds: Optional[tuple] = None,
    tol: float = 1e-5,
    max_iter: int = 100,
    epsilon: float = 1e-8
) -> Dict[str, Any]:
    """
    Minimize a scalar function of one or more variables using the BFGS algorithm.

    Parameters
    ----------
    func : Callable[[np.ndarray], float]
        Objective function to minimize.
    x0 : np.ndarray
        Initial guess.
    args : tuple, optional
        Additional arguments passed to objective function.
    bounds : Optional[tuple], optional
        Bounds for parameters, by default None
    tol : float, optional
        Tolerance for optimization, by default 1e-5
    max_iter : int, optional
        Maximum number of iterations, by default 100
    epsilon : float, optional
        Step size for numerical differentiation, by default 1e-8

    Returns
    ------
    Dict[str, Any]
        Dictionary containing optimization results.
    """
    validate_inputs(func, x0, args, bounds, tol, max_iter)

    x_k = x0.copy()
    n = len(x0)
    H_k = np.eye(n)  # Initial inverse Hessian approximation
    f_k = func(x_k, *args)
    grad_k = compute_gradient(func, x_k, epsilon)

    results = {
        "result": None,
        "metrics": {},
        "params_used": {
            "tol": tol,
            "max_iter": max_iter,
            "epsilon": epsilon
        },
        "warnings": []
    }

    for k in range(max_iter):
        # Compute search direction
        d_k = -np.dot(H_k, grad_k)

        # Line search
        alpha_k = line_search(func, x_k, d_k)
        x_new = x_k + alpha_k * d_k

        # Update parameters
        s_k = x_new - x_k
        grad_new = compute_gradient(func, x_new, epsilon)
        y_k = grad_new - grad_k

        # Update inverse Hessian
        H_k = bfgs_update(H_k, s_k, y_k)

        # Check convergence
        if np.linalg.norm(grad_new) < tol:
            results["result"] = x_new
            results["metrics"]["final_value"] = func(x_new, *args)
            results["metrics"]["iterations"] = k + 1
            return results

        # Update for next iteration
        x_k, f_k, grad_k = x_new, func(x_new, *args), grad_new

    results["result"] = x_k
    results["metrics"]["final_value"] = f_k
    results["metrics"]["iterations"] = max_iter
    results["warnings"].append("Maximum iterations reached")
    return results

# Example usage:
"""
def objective(x):
    return x[0]**2 + 10*x[1]**2

x0 = np.array([1.0, 1.0])
result = bfgs_fit(objective, x0)
print(result)
"""

################################################################################
# l_bfgs
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    loss_func: Callable[[np.ndarray, np.ndarray], float],
    grad_func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
) -> None:
    """Validate input data and functions."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be a 2D array and y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input data contains NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input data contains infinite values.")
    if grad_func is None:
        # Simple numerical gradient approximation
        def numerical_grad(X: np.ndarray, y: np.ndarray) -> np.ndarray:
            epsilon = 1e-8
            grad = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                X_plus = X.copy()
                X_plus[i] += epsilon
                grad[i] = (loss_func(X_plus, y) - loss_func(X, y)) / epsilon
            return grad
        grad_func = numerical_grad

def normalize_data(
    X: np.ndarray,
    y: np.ndarray,
    normalization: str = "standard",
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize input data."""
    if normalization == "none":
        return X, y
    elif normalization == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_normalized = (X - mean) / std
    elif normalization == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_normalized = (X - min_val) / (max_val - min_val + 1e-8)
    elif normalization == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_normalized = (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")
    return X_normalized, y

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Union[str, list[str], Callable[[np.ndarray, np.ndarray], float]],
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metric_results = {}
    if isinstance(metrics, str):
        metrics = [metrics]
    for metric in metrics:
        if callable(metric):
            metric_results["custom"] = metric(y_true, y_pred)
        elif metric == "mse":
            metric_results["mse"] = np.mean((y_true - y_pred) ** 2)
        elif metric == "mae":
            metric_results["mae"] = np.mean(np.abs(y_true - y_pred))
        elif metric == "r2":
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metric_results["r2"] = 1 - (ss_res / ss_tot)
        elif metric == "logloss":
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            metric_results["logloss"] = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    return metric_results

def l_bfgs_fit(
    X: np.ndarray,
    y: np.ndarray,
    loss_func: Callable[[np.ndarray, np.ndarray], float],
    grad_func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    normalization: str = "standard",
    metrics: Union[str, list[str], Callable[[np.ndarray, np.ndarray], float]] = "mse",
    max_iter: int = 100,
    tol: float = 1e-4,
    learning_rate: float = 0.01,
) -> Dict[str, Any]:
    """
    L-BFGS optimization algorithm.

    Parameters:
    -----------
    X : np.ndarray
        Input features (n_samples, n_features)
    y : np.ndarray
        Target values (n_samples,)
    loss_func : callable
        Loss function to minimize
    grad_func : callable, optional
        Gradient function (default: numerical approximation)
    normalization : str, optional
        Normalization method ("none", "standard", "minmax", "robust")
    metrics : str, list of str, or callable
        Metrics to compute ("mse", "mae", "r2", "logloss")
    max_iter : int, optional
        Maximum number of iterations (default: 100)
    tol : float, optional
        Tolerance for convergence (default: 1e-4)
    learning_rate : float, optional
        Learning rate for gradient descent (default: 0.01)

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    validate_inputs(X, y, loss_func, grad_func)

    # Normalize data
    X_norm, y_norm = normalize_data(X, y, normalization)

    # Initialize parameters (example: linear regression coefficients)
    params = np.zeros(X_norm.shape[1])

    # L-BFGS optimization
    for i in range(max_iter):
        grad = grad_func(X_norm, params) if grad_func else None
        params_new = params - learning_rate * grad

        # Check convergence
        if np.linalg.norm(params_new - params) < tol:
            break

        params = params_new

    # Compute predictions and metrics
    y_pred = loss_func(X_norm, params)
    metric_results = compute_metrics(y_norm, y_pred, metrics)

    return {
        "result": params,
        "metrics": metric_results,
        "params_used": {
            "normalization": normalization,
            "max_iter": max_iter,
            "tol": tol,
            "learning_rate": learning_rate
        },
        "warnings": []
    }

# Example usage:
"""
X = np.random.rand(100, 5)
y = np.random.rand(100)

def example_loss(X: np.ndarray, params: np.ndarray) -> float:
    return np.mean((X @ params - y) ** 2)

def example_grad(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    return 2 * X.T @ (X @ params - y) / len(y)

result = l_bfgs_fit(X, y, example_loss, example_grad)
"""

################################################################################
# simulated_annealing
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def simulated_annealing_fit(
    objective_function: Callable[[np.ndarray], float],
    initial_solution: np.ndarray,
    temperature: float = 1000.0,
    cooling_rate: float = 0.99,
    max_iterations: int = 1000,
    neighborhood_function: Callable[[np.ndarray], np.ndarray] = lambda x: x + np.random.normal(0, 1, size=x.shape),
    acceptance_criterion: Callable[[float, float, float], bool] = lambda old_cost, new_cost, temp: np.exp((old_cost - new_cost) / temp) > np.random.rand(),
    normalization: Optional[str] = None,
    metric: str = 'mse',
    tol: float = 1e-6,
    max_no_improvement: int = 50
) -> Dict[str, Any]:
    """
    Perform simulated annealing optimization.

    Parameters:
    - objective_function: Callable that takes a solution and returns the cost.
    - initial_solution: Initial solution vector.
    - temperature: Starting temperature for simulated annealing.
    - cooling_rate: Rate at which the temperature is reduced.
    - max_iterations: Maximum number of iterations.
    - neighborhood_function: Function to generate neighboring solutions.
    - acceptance_criterion: Function to decide whether to accept a new solution.
    - normalization: Optional normalization method ('none', 'standard', 'minmax', 'robust').
    - metric: Metric to evaluate the solution ('mse', 'mae', 'r2', 'logloss').
    - tol: Tolerance for convergence.
    - max_no_improvement: Maximum number of iterations without improvement before stopping.

    Returns:
    - Dictionary containing the result, metrics, parameters used, and warnings.
    """
    # Validate inputs
    validate_inputs(objective_function, initial_solution)

    current_solution = initial_solution.copy()
    best_solution = current_solution.copy()
    current_cost = objective_function(current_solution)
    best_cost = current_cost
    no_improvement_count = 0

    for iteration in range(max_iterations):
        # Generate a neighboring solution
        new_solution = neighborhood_function(current_solution)
        new_cost = objective_function(new_solution)

        # Decide whether to accept the new solution
        if acceptance_criterion(current_cost, new_cost, temperature):
            current_solution = new_solution
            current_cost = new_cost

            # Update best solution if improved
            if current_cost < best_cost:
                best_solution = current_solution.copy()
                best_cost = current_cost
                no_improvement_count = 0
            else:
                no_improvement_count += 1

        # Check for convergence
        if no_improvement_count >= max_no_improvement or temperature < tol:
            break

        # Cool down the temperature
        temperature *= cooling_rate

    # Calculate metrics
    metrics = calculate_metrics(best_solution, objective_function, metric)

    return {
        "result": best_solution,
        "metrics": metrics,
        "params_used": {
            "temperature": temperature,
            "cooling_rate": cooling_rate,
            "max_iterations": max_iterations,
            "normalization": normalization,
            "metric": metric
        },
        "warnings": []
    }

def validate_inputs(objective_function: Callable, initial_solution: np.ndarray) -> None:
    """Validate the inputs for simulated annealing."""
    if not callable(objective_function):
        raise ValueError("objective_function must be a callable.")
    if not isinstance(initial_solution, np.ndarray):
        raise ValueError("initial_solution must be a numpy array.")
    if np.any(np.isnan(initial_solution)) or np.any(np.isinf(initial_solution)):
        raise ValueError("initial_solution contains NaN or Inf values.")

def calculate_metrics(
    solution: np.ndarray,
    objective_function: Callable[[np.ndarray], float],
    metric: str
) -> Dict[str, float]:
    """Calculate the metrics for the solution."""
    cost = objective_function(solution)
    metrics = {}

    if metric == 'mse':
        metrics['mse'] = cost
    elif metric == 'mae':
        metrics['mae'] = cost
    elif metric == 'r2':
        metrics['r2'] = 1 - cost
    elif metric == 'logloss':
        metrics['logloss'] = cost
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

# Example usage
if __name__ == "__main__":
    # Define a simple objective function (e.g., quadratic)
    def objective_function(x: np.ndarray) -> float:
        return np.sum(x**2)

    # Initial solution
    initial_solution = np.array([5.0, 3.0])

    # Run simulated annealing
    result = simulated_annealing_fit(
        objective_function=objective_function,
        initial_solution=initial_solution
    )

    print(result)

################################################################################
# genetic_algorithms
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def genetic_algorithms_fit(
    fitness_func: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    population_size: int = 100,
    generations: int = 50,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.7,
    selection_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
    mutation_func: Callable[[np.ndarray, float], np.ndarray] = None,
    crossover_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = None,
    normalization: str = 'none',
    metric: Callable[[np.ndarray, np.ndarray], float] = None,
    tolerance: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, Any]:
    """
    Optimize a function using genetic algorithms.

    Parameters:
    -----------
    fitness_func : callable
        Function to optimize. Takes a numpy array and returns a float.
    bounds : numpy.ndarray
        Array of shape (n_parameters, 2) defining the bounds for each parameter.
    population_size : int, optional
        Size of the population (default: 100).
    generations : int, optional
        Number of generations to run (default: 50).
    mutation_rate : float, optional
        Probability of mutation (default: 0.1).
    crossover_rate : float, optional
        Probability of crossover (default: 0.7).
    selection_func : callable, optional
        Function for selecting parents (default: tournament selection).
    mutation_func : callable, optional
        Function for mutating individuals (default: Gaussian mutation).
    crossover_func : callable, optional
        Function for crossing over individuals (default: single-point crossover).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'none').
    metric : callable, optional
        Metric function to evaluate the solution (default: None).
    tolerance : float, optional
        Tolerance for convergence (default: 1e-6).
    max_iter : int, optional
        Maximum number of iterations (default: 1000).

    Returns:
    --------
    dict
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    # Default functions
    if selection_func is None:
        selection_func = _tournament_selection
    if mutation_func is None:
        mutation_func = _gaussian_mutation
    if crossover_func is None:
        crossover_func = _single_point_crossover

    # Validate inputs
    _validate_inputs(fitness_func, bounds, population_size, generations,
                     mutation_rate, crossover_rate)

    # Initialize population
    population = _initialize_population(bounds, population_size)
    best_solution = None
    best_fitness = -np.inf

    for generation in range(generations):
        # Evaluate fitness
        fitness = np.array([fitness_func(ind) for ind in population])

        # Track best solution
        current_best_idx = np.argmax(fitness)
        if fitness[current_best_idx] > best_fitness:
            best_fitness = fitness[current_best_idx]
            best_solution = population[current_best_idx].copy()

        # Check convergence
        if np.max(fitness) - best_fitness < tolerance:
            break

        # Selection
        parents = selection_func(population, fitness)

        # Crossover
        offspring = np.zeros_like(population)
        for i in range(0, population_size, 2):
            if np.random.rand() < crossover_rate:
                offspring[i:i+2] = crossover_func(parents[i], parents[i+1], crossover_rate)
            else:
                offspring[i:i+2] = parents[i:i+2].copy()

        # Mutation
        for i in range(population_size):
            if np.random.rand() < mutation_rate:
                offspring[i] = mutation_func(offspring[i], mutation_rate)

        # Replace population
        population = offspring

    # Calculate metrics if provided
    metrics = {}
    if metric is not None:
        metrics['final_metric'] = metric(best_solution, np.array([best_fitness]))

    return {
        'result': best_solution,
        'metrics': metrics,
        'params_used': {
            'population_size': population_size,
            'generations': generations,
            'mutation_rate': mutation_rate,
            'crossover_rate': crossover_rate
        },
        'warnings': []
    }

def _validate_inputs(
    fitness_func: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    population_size: int,
    generations: int,
    mutation_rate: float,
    crossover_rate: float
) -> None:
    """Validate the inputs for genetic algorithms."""
    if not callable(fitness_func):
        raise ValueError("fitness_func must be a callable")
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must be a 2D array of shape (n_parameters, 2)")
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if generations <= 0:
        raise ValueError("generations must be positive")
    if not (0 <= mutation_rate <= 1):
        raise ValueError("mutation_rate must be between 0 and 1")
    if not (0 <= crossover_rate <= 1):
        raise ValueError("crossover_rate must be between 0 and 1")

def _initialize_population(
    bounds: np.ndarray,
    population_size: int
) -> np.ndarray:
    """Initialize a random population within the given bounds."""
    n_parameters = bounds.shape[0]
    population = np.random.rand(population_size, n_parameters)
    for i in range(n_parameters):
        population[:, i] = population[:, i] * (bounds[i, 1] - bounds[i, 0]) + bounds[i, 0]
    return population

def _tournament_selection(
    population: np.ndarray,
    fitness: np.ndarray,
    tournament_size: int = 3
) -> np.ndarray:
    """Tournament selection for genetic algorithms."""
    n_parents = population.shape[0]
    parents = np.zeros_like(population)
    for i in range(n_parents):
        contestants = np.random.choice(len(population), tournament_size, replace=False)
        winner = contestants[np.argmax(fitness[contestants])]
        parents[i] = population[winner].copy()
    return parents

def _gaussian_mutation(
    individual: np.ndarray,
    mutation_rate: float
) -> np.ndarray:
    """Gaussian mutation for genetic algorithms."""
    mutation = np.random.normal(0, 1, size=individual.shape)
    return individual + mutation_rate * mutation

def _single_point_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    crossover_rate: float
) -> np.ndarray:
    """Single-point crossover for genetic algorithms."""
    n_parameters = parent1.shape[0]
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, n_parameters)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return np.array([child1, child2])
    else:
        return np.array([parent1, parent2])

# Example usage
if __name__ == "__main__":
    # Define a simple fitness function (e.g., sphere function)
    def sphere_function(x: np.ndarray) -> float:
        return -np.sum(x**2)

    # Define bounds for the parameters
    bounds = np.array([[-5, 5], [-5, 5]])

    # Run genetic algorithms
    result = genetic_algorithms_fit(
        fitness_func=sphere_function,
        bounds=bounds,
        population_size=50,
        generations=30
    )

    print("Best solution:", result['result'])
    print("Metrics:", result['metrics'])

################################################################################
# particle_swarm_optimization
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def particle_swarm_optimization_fit(
    objective_function: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_particles: int = 30,
    max_iter: int = 100,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    tol: float = 1e-6,
    metric: str = 'mse',
    normalization: Optional[str] = None,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Particle Swarm Optimization (PSO) algorithm for optimization problems.

    Parameters:
    -----------
    objective_function : callable
        The function to be minimized.
    bounds : np.ndarray
        Array of shape (n_dimensions, 2) representing the lower and upper bounds for each dimension.
    n_particles : int, optional
        Number of particles in the swarm (default: 30).
    max_iter : int, optional
        Maximum number of iterations (default: 100).
    w : float, optional
        Inertia weight (default: 0.7).
    c1 : float, optional
        Cognitive coefficient (default: 1.5).
    c2 : float, optional
        Social coefficient (default: 1.5).
    tol : float, optional
        Tolerance for convergence (default: 1e-6).
    metric : str, optional
        Metric to evaluate the performance (default: 'mse').
    normalization : str, optional
        Type of normalization to apply (default: None).
    random_state : int, optional
        Random seed for reproducibility (default: None).

    Returns:
    --------
    dict
        A dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(objective_function, bounds, n_particles, max_iter, w, c1, c2, tol, metric, normalization)

    # Initialize random state
    rng = np.random.RandomState(random_state)

    # Initialize particles
    n_dimensions = bounds.shape[0]
    particles = _initialize_particles(n_particles, n_dimensions, bounds, rng)

    # Initialize velocities
    velocities = np.zeros_like(particles)

    # Initialize personal best positions and values
    personal_best_positions = particles.copy()
    personal_best_values = np.array([objective_function(particle) for particle in particles])

    # Initialize global best position and value
    global_best_index = np.argmin(personal_best_values)
    global_best_position = personal_best_positions[global_best_index]
    global_best_value = personal_best_values[global_best_index]

    # Initialize metrics
    metrics = {
        'best_value': [],
        'average_value': []
    }

    # Main optimization loop
    for _ in range(max_iter):
        # Update velocities and positions
        velocities, particles = _update_particles(
            particles, velocities, personal_best_positions,
            global_best_position, w, c1, c2, bounds, rng
        )

        # Evaluate objective function for each particle
        current_values = np.array([objective_function(particle) for particle in particles])

        # Update personal best positions and values
        improved_indices = current_values < personal_best_values
        personal_best_positions[improved_indices] = particles[improved_indices]
        personal_best_values[improved_indices] = current_values[improved_indices]

        # Update global best position and value
        new_global_best_index = np.argmin(personal_best_values)
        if personal_best_values[new_global_best_index] < global_best_value:
            global_best_position = personal_best_positions[new_global_best_index]
            global_best_value = personal_best_values[new_global_best_index]

        # Store metrics
        metrics['best_value'].append(global_best_value)
        metrics['average_value'].append(np.mean(current_values))

        # Check for convergence
        if np.abs(global_best_value - personal_best_values[new_global_best_index]) < tol:
            break

    # Prepare results
    result = {
        'position': global_best_position,
        'value': global_best_value
    }

    params_used = {
        'n_particles': n_particles,
        'max_iter': max_iter,
        'w': w,
        'c1': c1,
        'c2': c2,
        'tol': tol,
        'metric': metric,
        'normalization': normalization
    }

    warnings = []

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

def _validate_inputs(
    objective_function: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_particles: int,
    max_iter: int,
    w: float,
    c1: float,
    c2: float,
    tol: float,
    metric: str,
    normalization: Optional[str]
) -> None:
    """Validate the inputs for the PSO algorithm."""
    if not callable(objective_function):
        raise ValueError("objective_function must be a callable.")
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must be a 2D array of shape (n_dimensions, 2).")
    if n_particles <= 0:
        raise ValueError("n_particles must be a positive integer.")
    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")
    if not (0 <= w <= 1):
        raise ValueError("w must be between 0 and 1.")
    if c1 < 0 or c2 < 0:
        raise ValueError("c1 and c2 must be non-negative.")
    if tol < 0:
        raise ValueError("tol must be non-negative.")
    if metric not in ['mse', 'mae', 'r2', 'logloss']:
        raise ValueError("metric must be one of 'mse', 'mae', 'r2', 'logloss'.")
    if normalization not in [None, 'none', 'standard', 'minmax', 'robust']:
        raise ValueError("normalization must be one of None, 'none', 'standard', 'minmax', 'robust'.")

def _initialize_particles(
    n_particles: int,
    n_dimensions: int,
    bounds: np.ndarray,
    rng: np.random.RandomState
) -> np.ndarray:
    """Initialize particles within the given bounds."""
    particles = np.zeros((n_particles, n_dimensions))
    for i in range(n_dimensions):
        particles[:, i] = rng.uniform(bounds[i, 0], bounds[i, 1], n_particles)
    return particles

def _update_particles(
    particles: np.ndarray,
    velocities: np.ndarray,
    personal_best_positions: np.ndarray,
    global_best_position: np.ndarray,
    w: float,
    c1: float,
    c2: float,
    bounds: np.ndarray,
    rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray]:
    """Update the velocities and positions of the particles."""
    n_particles, n_dimensions = particles.shape

    # Update velocities
    r1 = rng.random((n_particles, n_dimensions))
    r2 = rng.random((n_particles, n_dimensions))

    cognitive_component = c1 * r1 * (personal_best_positions - particles)
    social_component = c2 * r2 * (global_best_position - particles)

    velocities = w * velocities + cognitive_component + social_component

    # Update positions
    particles += velocities

    # Ensure particles are within bounds
    for i in range(n_dimensions):
        particles[:, i] = np.clip(particles[:, i], bounds[i, 0], bounds[i, 1])

    return velocities, particles

################################################################################
# hill_climbing
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(
    objective_func: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    bounds: Optional[Union[tuple, list]] = None
) -> None:
    """
    Validate the inputs for hill climbing optimization.

    Parameters
    ----------
    objective_func : callable
        The objective function to minimize.
    initial_params : numpy.ndarray
        Initial parameters for the optimization.
    bounds : tuple or list, optional
        Bounds for each parameter (min, max) pairs.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not callable(objective_func):
        raise ValueError("objective_func must be a callable function.")

    if not isinstance(initial_params, np.ndarray):
        raise ValueError("initial_params must be a numpy array.")

    if bounds is not None:
        if len(bounds) != len(initial_params):
            raise ValueError("Length of bounds must match length of initial_params.")
        for bound in bounds:
            if len(bound) != 2:
                raise ValueError("Each bound must be a tuple of (min, max).")

def compute_neighbor(
    current_params: np.ndarray,
    step_size: float = 0.1
) -> np.ndarray:
    """
    Compute a neighbor by adding random noise to current parameters.

    Parameters
    ----------
    current_params : numpy.ndarray
        Current parameters.
    step_size : float, optional
        Step size for the random noise.

    Returns
    ------
    numpy.ndarray
        Neighbor parameters.
    """
    return current_params + np.random.normal(0, step_size, size=current_params.shape)

def hill_climbing_fit(
    objective_func: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    max_iterations: int = 1000,
    step_size: float = 0.1,
    tolerance: float = 1e-6,
    bounds: Optional[Union[tuple, list]] = None
) -> Dict[str, Any]:
    """
    Perform hill climbing optimization.

    Parameters
    ----------
    objective_func : callable
        The objective function to minimize.
    initial_params : numpy.ndarray
        Initial parameters for the optimization.
    max_iterations : int, optional
        Maximum number of iterations.
    step_size : float, optional
        Step size for the random noise.
    tolerance : float, optional
        Tolerance for convergence.
    bounds : tuple or list, optional
        Bounds for each parameter (min, max) pairs.

    Returns
    ------
    dict
        A dictionary containing the result, metrics, parameters used, and warnings.
    """
    validate_inputs(objective_func, initial_params, bounds)

    current_params = initial_params.copy()
    best_params = current_params.copy()
    best_value = objective_func(current_params)
    iterations = 0
    warnings_list = []

    for _ in range(max_iterations):
        neighbor_params = compute_neighbor(current_params, step_size)

        if bounds is not None:
            for i, (min_val, max_val) in enumerate(bounds):
                neighbor_params[i] = np.clip(neighbor_params[i], min_val, max_val)

        neighbor_value = objective_func(neighbor_params)

        if neighbor_value < best_value:
            best_value = neighbor_value
            best_params = neighbor_params.copy()
        else:
            current_params = best_params.copy()

        if np.linalg.norm(neighbor_params - current_params) < tolerance:
            warnings_list.append("Convergence reached.")
            break

        iterations += 1

    return {
        "result": best_value,
        "metrics": {"iterations": iterations},
        "params_used": {
            "initial_params": initial_params.tolist(),
            "best_params": best_params.tolist()
        },
        "warnings": warnings_list
    }

# Example usage:
if __name__ == "__main__":
    def example_objective(x: np.ndarray) -> float:
        return x[0]**2 + x[1]**2

    initial_params = np.array([5.0, 5.0])
    result = hill_climbing_fit(example_objective, initial_params)
    print(result)

################################################################################
# tabu_search
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_input(
    cost_function: Callable[[np.ndarray], float],
    initial_solution: np.ndarray,
    neighborhood_function: Callable[[np.ndarray], np.ndarray],
    tabu_list_size: int,
    max_iterations: int
) -> None:
    """
    Validate the input parameters for tabu search.

    Parameters
    ----------
    cost_function : Callable[[np.ndarray], float]
        The cost function to minimize.
    initial_solution : np.ndarray
        The starting point for the search.
    neighborhood_function : Callable[[np.ndarray], np.ndarray]
        Function to generate neighboring solutions.
    tabu_list_size : int
        Size of the tabu list.
    max_iterations : int
        Maximum number of iterations.

    Raises
    ------
    ValueError
        If any input parameter is invalid.
    """
    if not callable(cost_function):
        raise ValueError("cost_function must be a callable.")
    if not isinstance(initial_solution, np.ndarray):
        raise ValueError("initial_solution must be a numpy array.")
    if not callable(neighborhood_function):
        raise ValueError("neighborhood_function must be a callable.")
    if not isinstance(tabu_list_size, int) or tabu_list_size <= 0:
        raise ValueError("tabu_list_size must be a positive integer.")
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer.")

def compute_cost(
    cost_function: Callable[[np.ndarray], float],
    solution: np.ndarray
) -> float:
    """
    Compute the cost of a given solution.

    Parameters
    ----------
    cost_function : Callable[[np.ndarray], float]
        The cost function to minimize.
    solution : np.ndarray
        The solution for which to compute the cost.

    Returns
    ------
    float
        The computed cost.
    """
    return cost_function(solution)

def generate_neighborhood(
    neighborhood_function: Callable[[np.ndarray], np.ndarray],
    current_solution: np.ndarray
) -> np.ndarray:
    """
    Generate neighboring solutions.

    Parameters
    ----------
    neighborhood_function : Callable[[np.ndarray], np.ndarray]
        Function to generate neighboring solutions.
    current_solution : np.ndarray
        The current solution.

    Returns
    ------
    np.ndarray
        The generated neighboring solutions.
    """
    return neighborhood_function(current_solution)

def update_tabu_list(
    tabu_list: list,
    new_solution: np.ndarray,
    tabu_list_size: int
) -> list:
    """
    Update the tabu list with a new solution.

    Parameters
    ----------
    tabu_list : list
        The current tabu list.
    new_solution : np.ndarray
        The new solution to add to the tabu list.
    tabu_list_size : int
        Size of the tabu list.

    Returns
    ------
    list
        The updated tabu list.
    """
    if len(tabu_list) >= tabu_list_size:
        tabu_list.pop(0)
    tabu_list.append(new_solution.tolist())
    return tabu_list

def is_tabu(
    solution: np.ndarray,
    tabu_list: list
) -> bool:
    """
    Check if a solution is in the tabu list.

    Parameters
    ----------
    solution : np.ndarray
        The solution to check.
    tabu_list : list
        The current tabu list.

    Returns
    ------
    bool
        True if the solution is in the tabu list, False otherwise.
    """
    return solution.tolist() in tabu_list

def tabu_search_fit(
    cost_function: Callable[[np.ndarray], float],
    initial_solution: np.ndarray,
    neighborhood_function: Callable[[np.ndarray], np.ndarray],
    tabu_list_size: int = 10,
    max_iterations: int = 100,
    aspiration_criterion: Optional[Callable[[np.ndarray, float], bool]] = None
) -> Dict[str, Any]:
    """
    Perform tabu search optimization.

    Parameters
    ----------
    cost_function : Callable[[np.ndarray], float]
        The cost function to minimize.
    initial_solution : np.ndarray
        The starting point for the search.
    neighborhood_function : Callable[[np.ndarray], np.ndarray]
        Function to generate neighboring solutions.
    tabu_list_size : int, optional
        Size of the tabu list (default is 10).
    max_iterations : int, optional
        Maximum number of iterations (default is 100).
    aspiration_criterion : Optional[Callable[[np.ndarray, float], bool]], optional
        Function to override the tabu status of a solution (default is None).

    Returns
    ------
    Dict[str, Any]
        A dictionary containing the result, metrics, parameters used, and warnings.
    """
    validate_input(cost_function, initial_solution, neighborhood_function, tabu_list_size, max_iterations)

    current_solution = initial_solution.copy()
    best_solution = current_solution.copy()
    best_cost = compute_cost(cost_function, best_solution)
    tabu_list = []
    iteration = 0
    warnings = []

    while iteration < max_iterations:
        neighbors = generate_neighborhood(neighborhood_function, current_solution)
        best_neighbor = None
        best_neighbor_cost = float('inf')

        for neighbor in neighbors:
            if is_tabu(neighbor, tabu_list):
                if aspiration_criterion and aspiration_criterion(neighbor, compute_cost(cost_function, neighbor)):
                    continue
                else:
                    continue

            current_neighbor_cost = compute_cost(cost_function, neighbor)
            if current_neighbor_cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = current_neighbor_cost

        if best_neighbor is not None:
            current_solution = best_neighbor
            tabu_list = update_tabu_list(tabu_list, current_solution, tabu_list_size)

            if best_neighbor_cost < best_cost:
                best_solution = current_solution.copy()
                best_cost = best_neighbor_cost

        iteration += 1

    result = {
        "result": best_solution,
        "metrics": {"best_cost": best_cost},
        "params_used": {
            "tabu_list_size": tabu_list_size,
            "max_iterations": max_iterations
        },
        "warnings": warnings
    }

    return result

# Example usage:
"""
def example_cost_function(solution: np.ndarray) -> float:
    return np.sum(solution**2)

def example_neighborhood_function(solution: np.ndarray) -> np.ndarray:
    return [solution + np.random.normal(0, 1, solution.shape) for _ in range(5)]

initial_solution = np.array([0.0, 0.0])
result = tabu_search_fit(
    cost_function=example_cost_function,
    initial_solution=initial_solution,
    neighborhood_function=example_neighborhood_function
)
print(result)
"""

################################################################################
# linear_programming
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def linear_programming_fit(
    c: np.ndarray,
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    A_eq: Optional[np.ndarray] = None,
    b_eq: Optional[np.ndarray] = None,
    bounds: Optional[tuple] = None,
    method: str = 'simplex',
    solver_options: Optional[Dict] = None,
    normalize: bool = False
) -> Dict:
    """
    Solve a linear programming problem.

    Parameters:
    -----------
    c : np.ndarray
        Coefficients of the objective function.
    A : np.ndarray, optional
        Inequality constraints matrix.
    b : np.ndarray, optional
        Inequality constraints vector.
    A_eq : np.ndarray, optional
        Equality constraints matrix.
    b_eq : np.ndarray, optional
        Equality constraints vector.
    bounds : tuple, optional
        Bounds on variables (min, max).
    method : str, default='simplex'
        Solver method ('simplex', 'interior-point').
    solver_options : dict, optional
        Options for the solver.
    normalize : bool, default=False
        Whether to normalize the constraints.

    Returns:
    --------
    dict
        A dictionary containing the solution, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(c, A, b, A_eq, b_eq, bounds)

    # Normalize constraints if required
    if normalize:
        A, b = _normalize_constraints(A, b)
        A_eq, b_eq = _normalize_constraints(A_eq, b_eq)

    # Choose solver
    if method == 'simplex':
        result = _solve_simplex(c, A, b, A_eq, b_eq, bounds, solver_options)
    elif method == 'interior-point':
        result = _solve_interior_point(c, A, b, A_eq, b_eq, bounds, solver_options)
    else:
        raise ValueError("Unsupported method. Choose 'simplex' or 'interior-point'.")

    # Calculate metrics
    metrics = _calculate_metrics(result)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'method': method,
            'normalize': normalize
        },
        'warnings': []
    }

def _validate_inputs(
    c: np.ndarray,
    A: Optional[np.ndarray],
    b: Optional[np.ndarray],
    A_eq: Optional[np.ndarray],
    b_eq: Optional[np.ndarray],
    bounds: Optional[tuple]
) -> None:
    """
    Validate the inputs for linear programming.

    Parameters:
    -----------
    c : np.ndarray
        Coefficients of the objective function.
    A : np.ndarray, optional
        Inequality constraints matrix.
    b : np.ndarray, optional
        Inequality constraints vector.
    A_eq : np.ndarray, optional
        Equality constraints matrix.
    b_eq : np.ndarray, optional
        Equality constraints vector.
    bounds : tuple, optional
        Bounds on variables (min, max).
    """
    if not isinstance(c, np.ndarray):
        raise TypeError("c must be a numpy array.")
    if A is not None and (not isinstance(A, np.ndarray) or b is None):
        raise ValueError("A must be a numpy array and b must be provided.")
    if A_eq is not None and (not isinstance(A_eq, np.ndarray) or b_eq is None):
        raise ValueError("A_eq must be a numpy array and b_eq must be provided.")
    if bounds is not None:
        if len(bounds) != 2 or not all(isinstance(x, (int, float)) for x in bounds):
            raise ValueError("bounds must be a tuple of two numbers (min, max).")

def _normalize_constraints(
    A: Optional[np.ndarray],
    b: Optional[np.ndarray]
) -> tuple:
    """
    Normalize the constraints.

    Parameters:
    -----------
    A : np.ndarray, optional
        Inequality constraints matrix.
    b : np.ndarray, optional
        Inequality constraints vector.

    Returns:
    --------
    tuple
        Normalized A and b.
    """
    if A is None or b is None:
        return A, b
    norms = np.linalg.norm(A, axis=1)
    A_normalized = A / norms[:, np.newaxis]
    b_normalized = b / norms
    return A_normalized, b_normalized

def _solve_simplex(
    c: np.ndarray,
    A: Optional[np.ndarray],
    b: Optional[np.ndarray],
    A_eq: Optional[np.ndarray],
    b_eq: Optional[np.ndarray],
    bounds: Optional[tuple],
    solver_options: Optional[Dict]
) -> Dict:
    """
    Solve the linear programming problem using the simplex method.

    Parameters:
    -----------
    c : np.ndarray
        Coefficients of the objective function.
    A : np.ndarray, optional
        Inequality constraints matrix.
    b : np.ndarray, optional
        Inequality constraints vector.
    A_eq : np.ndarray, optional
        Equality constraints matrix.
    b_eq : np.ndarray, optional
        Equality constraints vector.
    bounds : tuple, optional
        Bounds on variables (min, max).
    solver_options : dict, optional
        Options for the solver.

    Returns:
    --------
    dict
        The solution from the simplex method.
    """
    # Placeholder for actual simplex implementation
    return {'x': np.zeros_like(c), 'fun': 0.0}

def _solve_interior_point(
    c: np.ndarray,
    A: Optional[np.ndarray],
    b: Optional[np.ndarray],
    A_eq: Optional[np.ndarray],
    b_eq: Optional[np.ndarray],
    bounds: Optional[tuple],
    solver_options: Optional[Dict]
) -> Dict:
    """
    Solve the linear programming problem using the interior-point method.

    Parameters:
    -----------
    c : np.ndarray
        Coefficients of the objective function.
    A : np.ndarray, optional
        Inequality constraints matrix.
    b : np.ndarray, optional
        Inequality constraints vector.
    A_eq : np.ndarray, optional
        Equality constraints matrix.
    b_eq : np.ndarray, optional
        Equality constraints vector.
    bounds : tuple, optional
        Bounds on variables (min, max).
    solver_options : dict, optional
        Options for the solver.

    Returns:
    --------
    dict
        The solution from the interior-point method.
    """
    # Placeholder for actual interior-point implementation
    return {'x': np.zeros_like(c), 'fun': 0.0}

def _calculate_metrics(
    result: Dict
) -> Dict:
    """
    Calculate metrics for the linear programming solution.

    Parameters:
    -----------
    result : dict
        The solution from the solver.

    Returns:
    --------
    dict
        Metrics for the solution.
    """
    return {
        'objective_value': result['fun'],
        'constraint_violation': 0.0
    }

# Example usage:
"""
c = np.array([-1, -2])
A = np.array([[1, 0], [0, 1]])
b = np.array([1, 2])
result = linear_programming_fit(c, A, b)
print(result)
"""

################################################################################
# quadratic_programming
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def quadratic_programming_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    solver: str = 'closed_form',
    normalization: Optional[str] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    verbose: bool = False
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Solve a quadratic programming problem with various options.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    solver : str, optional
        Solver to use ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to evaluate ('mse', 'mae', 'r2', 'logloss') or custom callable.
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet').
    alpha : float, optional
        Regularization strength.
    l1_ratio : float, optional
        ElasticNet mixing parameter (0 <= l1_ratio <= 1).
    tol : float, optional
        Tolerance for stopping criteria.
    max_iter : int, optional
        Maximum number of iterations.
    learning_rate : float, optional
        Learning rate for gradient-based solvers.
    verbose : bool, optional
        Whether to print progress information.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': Optimized parameters
        - 'metrics': Computed metrics
        - 'params_used': Parameters used in the optimization
        - 'warnings': Any warnings encountered

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = quadratic_programming_fit(X, y, solver='gradient_descent', metric='mse')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_norm, y_norm = _apply_normalization(X, y, normalization)

    # Choose solver
    if solver == 'closed_form':
        result = _solve_closed_form(X_norm, y_norm, regularization, alpha)
    elif solver == 'gradient_descent':
        result = _solve_gradient_descent(X_norm, y_norm, regularization, alpha,
                                        l1_ratio, tol, max_iter, learning_rate)
    elif solver == 'newton':
        result = _solve_newton(X_norm, y_norm, regularization, alpha,
                              tol, max_iter)
    elif solver == 'coordinate_descent':
        result = _solve_coordinate_descent(X_norm, y_norm, regularization, alpha,
                                          l1_ratio, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_norm, y_norm, result, metric)

    # Prepare output
    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'solver': solver,
            'normalization': normalization,
            'metric': metric if isinstance(metric, str) else 'custom',
            'regularization': regularization,
            'alpha': alpha,
            'l1_ratio': l1_ratio
        },
        'warnings': _check_warnings(X_norm, y_norm, result)
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

def _apply_normalization(X: np.ndarray, y: np.ndarray,
                        normalization: Optional[str]) -> tuple:
    """Apply specified normalization to data."""
    if normalization is None or normalization == 'none':
        return X, y

    X_norm = X.copy()
    y_norm = y.copy()

    if normalization == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_norm = (X - X_mean) / X_std
    elif normalization == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
    elif normalization == 'robust':
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - X_median) / (X_iqr + 1e-8)

    return X_norm, y_norm

def _solve_closed_form(X: np.ndarray, y: np.ndarray,
                       regularization: Optional[str], alpha: float) -> np.ndarray:
    """Solve quadratic programming problem with closed-form solution."""
    if regularization is None or regularization == 'none':
        XtX = np.dot(X.T, X)
        if not np.allclose(XtX, XtX.T):
            raise ValueError("Matrix is not symmetric")
        if np.linalg.cond(XtX) > 1e6:
            raise ValueError("Matrix is singular or nearly singular")
        Xty = np.dot(X.T, y)
        return np.linalg.solve(XtX, Xty)

    elif regularization == 'l2':
        I = np.eye(X.shape[1])
        return np.linalg.solve(X.T @ X + alpha * I, X.T @ y)

    elif regularization == 'l1':
        # For L1, we need to use a different approach
        from scipy.optimize import minimize
        def objective(coef):
            return 0.5 * np.sum((y - X @ coef)**2) + alpha * np.linalg.norm(coef, 1)
        res = minimize(objective, np.zeros(X.shape[1]), method='L-BFGS-B')
        return res.x

    elif regularization == 'elasticnet':
        from scipy.optimize import minimize
        def objective(coef):
            l1_part = alpha * l1_ratio * np.linalg.norm(coef, 1)
            l2_part = alpha * (1 - l1_ratio) * np.linalg.norm(coef, 2)**2
            return 0.5 * np.sum((y - X @ coef)**2) + l1_part + l2_part
        res = minimize(objective, np.zeros(X.shape[1]), method='L-BFGS-B')
        return res.x

    else:
        raise ValueError(f"Unknown regularization: {regularization}")

def _solve_gradient_descent(X: np.ndarray, y: np.ndarray,
                           regularization: Optional[str], alpha: float,
                           l1_ratio: float, tol: float, max_iter: int,
                           learning_rate: float) -> np.ndarray:
    """Solve quadratic programming problem with gradient descent."""
    n_features = X.shape[1]
    coef = np.zeros(n_features)

    for _ in range(max_iter):
        gradient = -X.T @ (y - X @ coef) / y.size

        if regularization == 'l2':
            gradient += 2 * alpha * coef
        elif regularization == 'l1':
            gradient += alpha * np.sign(coef)
        elif regularization == 'elasticnet':
            gradient += alpha * (l1_ratio * np.sign(coef) + (1 - l1_ratio) * 2 * coef)

        coef_new = coef - learning_rate * gradient

        if np.linalg.norm(coef_new - coef, ord=np.inf) < tol:
            break

        coef = coef_new

    return coef

def _solve_newton(X: np.ndarray, y: np.ndarray,
                  regularization: Optional[str], alpha: float,
                  tol: float, max_iter: int) -> np.ndarray:
    """Solve quadratic programming problem with Newton's method."""
    n_features = X.shape[1]
    coef = np.zeros(n_features)

    for _ in range(max_iter):
        gradient = -X.T @ (y - X @ coef) / y.size

        if regularization == 'l2':
            gradient += 2 * alpha * coef
            hessian = X.T @ X / y.size + 2 * alpha * np.eye(n_features)
        elif regularization == 'l1':
            gradient += alpha * np.sign(coef)
            hessian = X.T @ X / y.size
        elif regularization == 'elasticnet':
            gradient += alpha * (l1_ratio * np.sign(coef) + (1 - l1_ratio) * 2 * coef)
            hessian = X.T @ X / y.size + 2 * alpha * (1 - l1_ratio) * np.eye(n_features)
        else:
            hessian = X.T @ X / y.size

        if not np.allclose(hessian, hessian.T):
            raise ValueError("Hessian matrix is not symmetric")

        if np.linalg.cond(hessian) > 1e6:
            raise ValueError("Hessian matrix is singular or nearly singular")

        coef_new = coef - np.linalg.solve(hessian, gradient)

        if np.linalg.norm(coef_new - coef, ord=np.inf) < tol:
            break

        coef = coef_new

    return coef

def _solve_coordinate_descent(X: np.ndarray, y: np.ndarray,
                             regularization: Optional[str], alpha: float,
                             l1_ratio: float, tol: float, max_iter: int) -> np.ndarray:
    """Solve quadratic programming problem with coordinate descent."""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residual = y - np.dot(X, coef) + coef[j] * X_j

            if regularization == 'l2':
                numerator = np.dot(X_j, residual)
                denominator = np.dot(X_j, X_j) + 2 * alpha
            elif regularization == 'l1':
                numerator = np.dot(X_j, residual)
                denominator = 1
            elif regularization == 'elasticnet':
                numerator = np.dot(X_j, residual)
                denominator = 1
            else:
                numerator = np.dot(X_j, residual)
                denominator = np.dot(X_j, X_j)

            if regularization in ['l1', 'elasticnet']:
                coef_new = np.sign(numerator) * np.maximum(np.abs(numerator) - alpha * l1_ratio, 0) / denominator
            else:
                coef_new = numerator / denominator

            if np.abs(coef_new - coef[j]) < tol:
                continue

            coef[j] = coef_new

    return coef

def _compute_metrics(X: np.ndarray, y: np.ndarray,
                     coef: np.ndarray,
                     metric: Union[str, Callable]) -> Dict[str, float]:
    """Compute specified metrics."""
    y_pred = X @ coef

    if isinstance(metric, str):
        if metric == 'mse':
            return {'mse': np.mean((y - y_pred)**2)}
        elif metric == 'mae':
            return {'mae': np.mean(np.abs(y - y_pred))}
        elif metric == 'r2':
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            return {'r2': 1 - ss_res / (ss_tot + 1e-8)}
        elif metric == 'logloss':
            y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
            return {'logloss': -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))}
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        return {'custom': metric(y, y_pred)}

def _check_warnings(X: np.ndarray, y: np.ndarray,
                    coef: np.ndarray) -> Dict[str, str]:
    """Check for potential warnings."""
    warnings = {}

    if np.any(np.isnan(coef)):
        warnings['nan_coefficients'] = "Some coefficients are NaN"

    if np.any(np.isinf(coef)):
        warnings['inf_coefficients'] = "Some coefficients are infinite"

    if np.linalg.norm(coef) > 1e6:
        warnings['large_coefficients'] = "Some coefficients are very large"

    return warnings

################################################################################
# integer_programming
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List, Any

def integer_programming_fit(
    objective: Callable[[np.ndarray], float],
    constraints: List[Callable[[np.ndarray], bool]],
    bounds: Optional[List[tuple]] = None,
    initial_guess: Optional[np.ndarray] = None,
    solver: str = 'branch_and_bound',
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Solve an integer programming problem.

    Parameters:
    - objective: Callable that computes the objective function value.
    - constraints: List of callables representing constraints (must return boolean).
    - bounds: Optional list of tuples specifying variable bounds.
    - initial_guess: Optional initial guess for the solution.
    - solver: Solver method ('branch_and_bound', 'cutting_plane', etc.).
    - max_iter: Maximum number of iterations.
    - tol: Tolerance for convergence.
    - verbose: Whether to print progress information.

    Returns:
    - Dictionary containing 'result', 'metrics', 'params_used', and 'warnings'.
    """
    # Validate inputs
    validate_inputs(objective, constraints, bounds, initial_guess)

    # Initialize parameters
    params_used = {
        'solver': solver,
        'max_iter': max_iter,
        'tol': tol
    }

    # Solve the problem based on the chosen solver
    if solver == 'branch_and_bound':
        result, metrics = branch_and_bound_solver(
            objective, constraints, bounds, initial_guess,
            max_iter, tol, verbose
        )
    elif solver == 'cutting_plane':
        result, metrics = cutting_plane_solver(
            objective, constraints, bounds, initial_guess,
            max_iter, tol, verbose
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Prepare the output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': []
    }

    return output

def validate_inputs(
    objective: Callable[[np.ndarray], float],
    constraints: List[Callable[[np.ndarray], bool]],
    bounds: Optional[List[tuple]] = None,
    initial_guess: Optional[np.ndarray] = None
) -> None:
    """
    Validate the inputs for integer programming.

    Parameters:
    - objective: Callable that computes the objective function value.
    - constraints: List of callables representing constraints (must return boolean).
    - bounds: Optional list of tuples specifying variable bounds.
    - initial_guess: Optional initial guess for the solution.

    Raises:
    - ValueError if inputs are invalid.
    """
    if not callable(objective):
        raise ValueError("Objective must be a callable.")
    for constraint in constraints:
        if not callable(constraint):
            raise ValueError("All constraints must be callables.")
    if bounds is not None:
        for bound in bounds:
            if len(bound) != 2 or bound[0] > bound[1]:
                raise ValueError("Bounds must be tuples of (min, max) with min <= max.")
    if initial_guess is not None and bounds is not None:
        for i, val in enumerate(initial_guess):
            if not (bounds[i][0] <= val <= bounds[i][1]):
                raise ValueError("Initial guess violates bounds.")

def branch_and_bound_solver(
    objective: Callable[[np.ndarray], float],
    constraints: List[Callable[[np.ndarray], bool]],
    bounds: Optional[List[tuple]] = None,
    initial_guess: Optional[np.ndarray] = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False
) -> tuple[np.ndarray, Dict[str, float]]:
    """
    Solve the integer programming problem using branch and bound.

    Parameters:
    - objective: Callable that computes the objective function value.
    - constraints: List of callables representing constraints (must return boolean).
    - bounds: Optional list of tuples specifying variable bounds.
    - initial_guess: Optional initial guess for the solution.
    - max_iter: Maximum number of iterations.
    - tol: Tolerance for convergence.
    - verbose: Whether to print progress information.

    Returns:
    - Tuple of (result, metrics).
    """
    # Initialize variables
    best_solution = None
    best_value = float('inf')
    metrics = {'iterations': 0, 'converged': False}

    # Main optimization loop
    for _ in range(max_iter):
        metrics['iterations'] += 1

        # Implement branch and bound logic here
        # This is a placeholder for the actual implementation
        current_solution = np.random.randint(0, 2, size=10) if initial_guess is None else initial_guess.copy()
        current_value = objective(current_solution)

        # Check constraints
        feasible = all(constraint(current_solution) for constraint in constraints)
        if feasible and current_value < best_value:
            best_value = current_value
            best_solution = current_solution.copy()

        # Check for convergence
        if abs(best_value - current_value) < tol:
            metrics['converged'] = True
            break

    return best_solution, metrics

def cutting_plane_solver(
    objective: Callable[[np.ndarray], float],
    constraints: List[Callable[[np.ndarray], bool]],
    bounds: Optional[List[tuple]] = None,
    initial_guess: Optional[np.ndarray] = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False
) -> tuple[np.ndarray, Dict[str, float]]:
    """
    Solve the integer programming problem using cutting plane method.

    Parameters:
    - objective: Callable that computes the objective function value.
    - constraints: List of callables representing constraints (must return boolean).
    - bounds: Optional list of tuples specifying variable bounds.
    - initial_guess: Optional initial guess for the solution.
    - max_iter: Maximum number of iterations.
    - tol: Tolerance for convergence.
    - verbose: Whether to print progress information.

    Returns:
    - Tuple of (result, metrics).
    """
    # Initialize variables
    best_solution = None
    best_value = float('inf')
    metrics = {'iterations': 0, 'converged': False}

    # Main optimization loop
    for _ in range(max_iter):
        metrics['iterations'] += 1

        # Implement cutting plane logic here
        # This is a placeholder for the actual implementation
        current_solution = np.random.randint(0, 2, size=10) if initial_guess is None else initial_guess.copy()
        current_value = objective(current_solution)

        # Check constraints
        feasible = all(constraint(current_solution) for constraint in constraints)
        if feasible and current_value < best_value:
            best_value = current_value
            best_solution = current_solution.copy()

        # Check for convergence
        if abs(best_value - current_value) < tol:
            metrics['converged'] = True
            break

    return best_solution, metrics

################################################################################
# convex_optimization
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def convex_optimization_fit(
    objective: Callable[[np.ndarray], float],
    gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    hessian: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    initial_params: np.ndarray = np.zeros(1),
    solver: str = 'gradient_descent',
    max_iter: int = 1000,
    tol: float = 1e-6,
    learning_rate: float = 0.01,
    regularization: Optional[str] = None,
    reg_param: float = 1.0,
    normalization: str = 'none',
    metric: str = 'mse',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    verbose: bool = False
) -> Dict[str, Union[np.ndarray, float, Dict[str, float], str]]:
    """
    Perform convex optimization with various configurable options.

    Parameters
    ----------
    objective : callable
        The objective function to minimize.
    gradient : callable, optional
        The gradient of the objective function. If None, finite differences will be used.
    hessian : callable, optional
        The Hessian of the objective function. Only used with Newton method.
    initial_params : ndarray, optional
        Initial parameters for the optimization. Default is zeros array.
    solver : str, optional
        Optimization algorithm to use. Options: 'gradient_descent', 'newton', 'coordinate_descent'.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence.
    learning_rate : float, optional
        Learning rate for gradient descent.
    regularization : str, optional
        Type of regularization. Options: 'l1', 'l2', 'elasticnet'.
    reg_param : float, optional
        Regularization parameter.
    normalization : str, optional
        Type of normalization. Options: 'none', 'standard', 'minmax'.
    metric : str, optional
        Metric to evaluate. Options: 'mse', 'mae', 'r2'.
    custom_metric : callable, optional
        Custom metric function if needed.
    verbose : bool, optional
        Whether to print progress information.

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': optimized parameters
        - 'metrics': dictionary of computed metrics
        - 'params_used': dictionary of used parameters
        - 'warnings': any warnings encountered

    Example
    -------
    >>> def quadratic(x):
    ...     return np.sum(x**2)
    ...
    >>> result = convex_optimization_fit(quadratic, solver='gradient_descent')
    """
    # Validate inputs
    _validate_inputs(objective, initial_params)

    # Normalize data if needed
    params = _apply_normalization(initial_params, normalization)

    # Initialize solver-specific parameters
    solver_params = {
        'max_iter': max_iter,
        'tol': tol,
        'learning_rate': learning_rate
    }

    # Choose solver
    if solver == 'gradient_descent':
        result = _gradient_descent(objective, gradient, params, **solver_params)
    elif solver == 'newton':
        if hessian is None:
            raise ValueError("Hessian must be provided for Newton method")
        result = _newton_method(objective, gradient, hessian, params, **solver_params)
    elif solver == 'coordinate_descent':
        result = _coordinate_descent(objective, params, **solver_params)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if needed
    if regularization is not None:
        result = _apply_regularization(result, regularization, reg_param)

    # Calculate metrics
    metrics = _calculate_metrics(result, metric, custom_metric)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'solver': solver,
            'max_iter': max_iter,
            'tol': tol,
            'learning_rate': learning_rate,
            'regularization': regularization,
            'reg_param': reg_param,
            'normalization': normalization
        },
        'warnings': []
    }

def _validate_inputs(objective: Callable, params: np.ndarray) -> None:
    """Validate input parameters."""
    if not callable(objective):
        raise TypeError("Objective must be a callable function")
    if not isinstance(params, np.ndarray):
        raise TypeError("Initial parameters must be a numpy array")
    if np.any(np.isnan(params)) or np.any(np.isinf(params)):
        raise ValueError("Initial parameters contain NaN or Inf values")

def _apply_normalization(params: np.ndarray, normalization: str) -> np.ndarray:
    """Apply specified normalization to parameters."""
    if normalization == 'standard':
        mean = np.mean(params)
        std = np.std(params)
        if std == 0:
            return params
        return (params - mean) / std
    elif normalization == 'minmax':
        min_val = np.min(params)
        max_val = np.max(params)
        if max_val == min_val:
            return params
        return (params - min_val) / (max_val - min_val)
    elif normalization == 'robust':
        median = np.median(params)
        iqr = np.percentile(params, 75) - np.percentile(params, 25)
        if iqr == 0:
            return params
        return (params - median) / iqr
    else:
        return params

def _gradient_descent(
    objective: Callable,
    gradient: Optional[Callable],
    params: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-6,
    learning_rate: float = 0.01
) -> np.ndarray:
    """Perform gradient descent optimization."""
    for i in range(max_iter):
        current_value = objective(params)

        if gradient is None:
            grad = _finite_difference_gradient(objective, params)
        else:
            grad = gradient(params)

        new_params = params - learning_rate * grad
        new_value = objective(new_params)

        if np.linalg.norm(new_params - params) < tol:
            break

        params = new_params

    return params

def _newton_method(
    objective: Callable,
    gradient: Callable,
    hessian: Callable,
    params: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> np.ndarray:
    """Perform Newton method optimization."""
    for _ in range(max_iter):
        grad = gradient(params)
        hess = hessian(params)

        try:
            delta = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            delta = -np.linalg.pinv(hess) @ grad

        new_params = params + delta
        if np.linalg.norm(delta) < tol:
            break

        params = new_params

    return params

def _coordinate_descent(
    objective: Callable,
    params: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> np.ndarray:
    """Perform coordinate descent optimization."""
    for _ in range(max_iter):
        old_params = params.copy()
        for i in range(len(params)):
            # Create a copy of parameters
            test_params = params.copy()

            # Find the best value for this coordinate
            values = np.linspace(params[i] - 1, params[i] + 1, 100)
            obj_values = np.array([objective(test_params) for test_params in _create_test_array(test_params, values, i)])

            params[i] = values[np.argmin(obj_values)]

        if np.linalg.norm(params - old_params) < tol:
            break

    return params

def _create_test_array(base: np.ndarray, values: np.ndarray, index: int) -> np.ndarray:
    """Helper function for coordinate descent."""
    arrays = []
    for val in values:
        temp = base.copy()
        temp[index] = val
        arrays.append(temp)
    return np.array(arrays)

def _finite_difference_gradient(objective: Callable, params: np.ndarray) -> np.ndarray:
    """Compute gradient using finite differences."""
    epsilon = 1e-8
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += epsilon
        params_minus = params.copy()
        params_minus[i] -= epsilon

        grad[i] = (objective(params_plus) - objective(params_minus)) / (2 * epsilon)
    return grad

def _apply_regularization(
    params: np.ndarray,
    regularization: str,
    reg_param: float
) -> np.ndarray:
    """Apply specified regularization to parameters."""
    if regularization == 'l1':
        return np.sign(params) * np.maximum(np.abs(params) - reg_param, 0)
    elif regularization == 'l2':
        return params / (1 + reg_param * np.linalg.norm(params))
    elif regularization == 'elasticnet':
        l1 = np.sign(params) * np.maximum(np.abs(params) - reg_param, 0)
        l2 = params / (1 + reg_param * np.linalg.norm(params))
        return 0.5 * l1 + 0.5 * l2
    else:
        return params

def _calculate_metrics(
    result: np.ndarray,
    metric: str = 'mse',
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """Calculate specified metrics."""
    metrics = {}

    if custom_metric is not None:
        metrics['custom'] = custom_metric(result, result)  # Example usage

    if metric == 'mse':
        metrics['mse'] = np.mean((result - result) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(result - result))
    elif metric == 'r2':
        metrics['r2'] = 1 - np.sum((result - result) ** 2) / np.sum(result ** 2)

    return metrics

################################################################################
# non_convex_optimization
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def non_convex_optimization_fit(
    objective_func: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    solver: str = 'gradient_descent',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    normalization: str = 'none',
    max_iter: int = 1000,
    tol: float = 1e-6,
    learning_rate: float = 0.01,
    regularization: Optional[str] = None,
    l1_ratio: float = 0.5,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Optimize a non-convex objective function using various solvers and metrics.

    Parameters:
    - objective_func: Callable that takes parameters and returns the objective value.
    - initial_params: Initial guess for the parameters.
    - solver: Solver to use ('gradient_descent', 'newton', 'coordinate_descent').
    - metric: Metric to evaluate the optimization ('mse', 'mae', 'r2', or custom callable).
    - normalization: Normalization method ('none', 'standard', 'minmax', 'robust').
    - max_iter: Maximum number of iterations.
    - tol: Tolerance for convergence.
    - learning_rate: Learning rate for gradient descent.
    - regularization: Regularization type ('none', 'l1', 'l2', 'elasticnet').
    - l1_ratio: Ratio for elastic net regularization.
    - verbose: Whether to print progress.

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    validate_inputs(objective_func, initial_params)

    # Normalize data if required
    normalized_params = apply_normalization(initial_params, normalization)

    # Initialize solver parameters
    params_used = {
        'solver': solver,
        'metric': metric,
        'normalization': normalization,
        'max_iter': max_iter,
        'tol': tol,
        'learning_rate': learning_rate,
        'regularization': regularization,
        'l1_ratio': l1_ratio
    }

    # Choose solver and optimize
    if solver == 'gradient_descent':
        optimized_params = gradient_descent(
            objective_func, normalized_params, max_iter, tol, learning_rate,
            regularization, l1_ratio
        )
    elif solver == 'newton':
        optimized_params = newton_method(
            objective_func, normalized_params, max_iter, tol
        )
    elif solver == 'coordinate_descent':
        optimized_params = coordinate_descent(
            objective_func, normalized_params, max_iter, tol
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    result = objective_func(optimized_params)
    metrics = calculate_metrics(objective_func, optimized_params, metric)

    # Denormalize parameters if required
    denormalized_params = apply_denormalization(optimized_params, normalization)

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'optimized_params': denormalized_params,
        'warnings': check_warnings(optimized_params, metrics)
    }

    return output

def validate_inputs(
    objective_func: Callable[[np.ndarray], float],
    initial_params: np.ndarray
) -> None:
    """Validate the inputs for non-convex optimization."""
    if not callable(objective_func):
        raise TypeError("objective_func must be a callable.")
    if not isinstance(initial_params, np.ndarray):
        raise TypeError("initial_params must be a numpy array.")
    if np.any(np.isnan(initial_params)) or np.any(np.isinf(initial_params)):
        raise ValueError("initial_params must not contain NaN or Inf values.")

def apply_normalization(
    params: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply normalization to the parameters."""
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
        raise ValueError(f"Unknown normalization method: {method}")

def apply_denormalization(
    params: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply denormalization to the parameters."""
    if method == 'none':
        return params
    elif method == 'standard':
        mean = np.mean(params)
        std = np.std(params)
        return params * (std + 1e-8) + mean
    elif method == 'minmax':
        min_val = np.min(params)
        max_val = np.max(params)
        return params * (max_val - min_val + 1e-8) + min_val
    elif method == 'robust':
        median = np.median(params)
        iqr = np.percentile(params, 75) - np.percentile(params, 25)
        return params * (iqr + 1e-8) + median
    else:
        raise ValueError(f"Unknown denormalization method: {method}")

def gradient_descent(
    objective_func: Callable[[np.ndarray], float],
    params: np.ndarray,
    max_iter: int,
    tol: float,
    learning_rate: float,
    regularization: Optional[str],
    l1_ratio: float
) -> np.ndarray:
    """Perform gradient descent optimization."""
    for _ in range(max_iter):
        grad = compute_gradient(objective_func, params)
        if regularization == 'l1':
            grad += l1_ratio * np.sign(params)
        elif regularization == 'l2':
            grad += 2 * l1_ratio * params
        elif regularization == 'elasticnet':
            grad += l1_ratio * (np.sign(params) + 2 * params)
        params -= learning_rate * grad
        if np.linalg.norm(grad) < tol:
            break
    return params

def newton_method(
    objective_func: Callable[[np.ndarray], float],
    params: np.ndarray,
    max_iter: int,
    tol: float
) -> np.ndarray:
    """Perform Newton's method optimization."""
    for _ in range(max_iter):
        grad = compute_gradient(objective_func, params)
        hessian = compute_hessian(objective_func, params)
        update = np.linalg.solve(hessian, -grad)
        params += update
        if np.linalg.norm(update) < tol:
            break
    return params

def coordinate_descent(
    objective_func: Callable[[np.ndarray], float],
    params: np.ndarray,
    max_iter: int,
    tol: float
) -> np.ndarray:
    """Perform coordinate descent optimization."""
    for _ in range(max_iter):
        old_params = params.copy()
        for i in range(len(params)):
            other_params = np.delete(params, i)
            params[i] = minimize_coordinate(objective_func, other_params, i)
        if np.linalg.norm(params - old_params) < tol:
            break
    return params

def compute_gradient(
    objective_func: Callable[[np.ndarray], float],
    params: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """Compute the gradient of the objective function."""
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += epsilon
        params_minus = params.copy()
        params_minus[i] -= epsilon
        grad[i] = (objective_func(params_plus) - objective_func(params_minus)) / (2 * epsilon)
    return grad

def compute_hessian(
    objective_func: Callable[[np.ndarray], float],
    params: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """Compute the Hessian matrix of the objective function."""
    n = len(params)
    hessian = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            params_plus = params.copy()
            params_plus[i] += epsilon
            params_plus[j] += epsilon
            params_minus_i = params.copy()
            params_minus_i[i] -= epsilon
            params_minus_j = params.copy()
            params_minus_j[j] -= epsilon
            hessian[i, j] = (objective_func(params_plus) - objective_func(params_minus_i) -
                             objective_func(params_minus_j) + objective_func(params)) / (epsilon ** 2)
    return hessian

def minimize_coordinate(
    objective_func: Callable[[np.ndarray], float],
    other_params: np.ndarray,
    coord_index: int
) -> float:
    """Minimize the objective function with respect to a single coordinate."""
    # This is a placeholder for actual minimization logic
    return 0.0

def calculate_metrics(
    objective_func: Callable[[np.ndarray], float],
    params: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Calculate the metrics for the optimization result."""
    if callable(metric):
        return {'custom_metric': metric(params, params)}
    elif metric == 'mse':
        return {'mse': np.mean((params - params) ** 2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(params - params))}
    elif metric == 'r2':
        return {'r2': 1.0}  # Placeholder for R-squared
    else:
        raise ValueError(f"Unknown metric: {metric}")

def check_warnings(
    params: np.ndarray,
    metrics: Dict[str, float]
) -> list:
    """Check for warnings in the optimization result."""
    warnings = []
    if np.any(np.isnan(params)) or np.any(np.isinf(params)):
        warnings.append("Optimized parameters contain NaN or Inf values.")
    if any(v is None for v in metrics.values()):
        warnings.append("Some metrics could not be computed.")
    return warnings

################################################################################
# constrained_optimization
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def constrained_optimization_fit(
    objective_func: Callable[[np.ndarray], float],
    constraints: Optional[list] = None,
    bounds: Optional[tuple] = None,
    initial_guess: np.ndarray = None,
    solver: str = 'gradient_descent',
    normalization: str = 'none',
    metric: Union[str, Callable[[np.ndarray], float]] = 'mse',
    max_iter: int = 1000,
    tol: float = 1e-6,
    learning_rate: float = 0.01,
    regularization: str = 'none',
    alpha: float = 1.0,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Perform constrained optimization with various configurable options.

    Parameters:
    -----------
    objective_func : callable
        The objective function to minimize.
    constraints : list of tuples, optional
        List of (constraint_func, constraint_type) where constraint_type is 'eq' or 'ineq'.
    bounds : tuple of arrays, optional
        Bounds for the variables (min, max).
    initial_guess : np.ndarray, optional
        Initial guess for the parameters.
    solver : str, optional
        Solver to use ('gradient_descent', 'newton', 'coordinate_descent').
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to evaluate ('mse', 'mae', 'r2', custom callable).
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence.
    learning_rate : float, optional
        Learning rate for gradient descent.
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet').
    alpha : float, optional
        Regularization strength.
    verbose : bool, optional
        Whether to print progress.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(objective_func, constraints, bounds, initial_guess)

    # Normalize data if needed
    normalized_guess = _apply_normalization(initial_guess, normalization)

    # Choose solver and optimize
    if solver == 'gradient_descent':
        result = _gradient_descent(
            objective_func, normalized_guess, constraints, bounds,
            max_iter, tol, learning_rate, regularization, alpha
        )
    elif solver == 'newton':
        result = _newton_method(
            objective_func, normalized_guess, constraints, bounds,
            max_iter, tol
        )
    elif solver == 'coordinate_descent':
        result = _coordinate_descent(
            objective_func, normalized_guess, constraints, bounds,
            max_iter, tol
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(result['params'], metric)

    # Denormalize parameters if needed
    denormalized_params = _denormalize_parameters(result['params'], normalization)

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': denormalized_params,
        'warnings': _check_warnings(result)
    }

    return output

def _validate_inputs(
    objective_func: Callable[[np.ndarray], float],
    constraints: Optional[list] = None,
    bounds: Optional[tuple] = None,
    initial_guess: np.ndarray = None
) -> None:
    """Validate input parameters."""
    if not callable(objective_func):
        raise TypeError("objective_func must be a callable")
    if constraints is not None:
        for constraint in constraints:
            if len(constraint) != 2 or not callable(constraint[0]):
                raise ValueError("Constraints must be a list of (callable, type) tuples")
    if bounds is not None:
        if len(bounds) != 2 or bounds[0].shape != bounds[1].shape:
            raise ValueError("Bounds must be a tuple of two arrays with the same shape")
    if initial_guess is not None and np.any(np.isnan(initial_guess)):
        raise ValueError("Initial guess contains NaN values")

def _apply_normalization(
    x: np.ndarray,
    method: str = 'none'
) -> np.ndarray:
    """Apply normalization to the input array."""
    if method == 'none':
        return x
    elif method == 'standard':
        mean = np.mean(x)
        std = np.std(x)
        return (x - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(x)
        max_val = np.max(x)
        return (x - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(x)
        iqr = np.percentile(x, 75) - np.percentile(x, 25)
        return (x - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _denormalize_parameters(
    x: np.ndarray,
    method: str = 'none'
) -> np.ndarray:
    """Denormalize parameters to their original scale."""
    if method == 'none':
        return x
    elif method in ['standard', 'minmax', 'robust']:
        # In a real implementation, you would need to store the normalization parameters
        # For simplicity, we assume identity denormalization here
        return x
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _gradient_descent(
    objective_func: Callable[[np.ndarray], float],
    initial_guess: np.ndarray,
    constraints: Optional[list] = None,
    bounds: Optional[tuple] = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    learning_rate: float = 0.01,
    regularization: str = 'none',
    alpha: float = 1.0
) -> Dict[str, Any]:
    """Perform gradient descent optimization."""
    params = initial_guess.copy()
    for i in range(max_iter):
        grad = _compute_gradient(objective_func, params)
        if regularization == 'l1':
            grad += alpha * np.sign(params)
        elif regularization == 'l2':
            grad += 2 * alpha * params
        elif regularization == 'elasticnet':
            grad += alpha * (np.sign(params) + 2 * params)

        # Apply constraints and bounds
        if constraints is not None:
            grad = _apply_constraints(grad, constraints)
        if bounds is not None:
            params = np.clip(params - learning_rate * grad, bounds[0], bounds[1])
        else:
            params -= learning_rate * grad

        # Check convergence
        if np.linalg.norm(grad) < tol:
            break

    return {
        'params': params,
        'iterations': i + 1,
        'converged': np.linalg.norm(grad) < tol
    }

def _compute_gradient(
    objective_func: Callable[[np.ndarray], float],
    params: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """Compute the gradient of the objective function."""
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += epsilon
        params_minus = params.copy()
        params_minus[i] -= epsilon
        grad[i] = (objective_func(params_plus) - objective_func(params_minus)) / (2 * epsilon)
    return grad

def _apply_constraints(
    grad: np.ndarray,
    constraints: list
) -> np.ndarray:
    """Apply constraints to the gradient."""
    for constraint_func, constraint_type in constraints:
        if constraint_type == 'eq':
            grad -= constraint_func(grad)
    return grad

def _newton_method(
    objective_func: Callable[[np.ndarray], float],
    initial_guess: np.ndarray,
    constraints: Optional[list] = None,
    bounds: Optional[tuple] = None,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> Dict[str, Any]:
    """Perform Newton's method optimization."""
    params = initial_guess.copy()
    for i in range(max_iter):
        grad = _compute_gradient(objective_func, params)
        hessian = _compute_hessian(objective_func, params)

        # Solve the linear system for the step
        step = np.linalg.solve(hessian, -grad)

        # Apply constraints and bounds
        if constraints is not None:
            step = _apply_constraints(step, constraints)
        if bounds is not None:
            params = np.clip(params + step, bounds[0], bounds[1])
        else:
            params += step

        # Check convergence
        if np.linalg.norm(grad) < tol:
            break

    return {
        'params': params,
        'iterations': i + 1,
        'converged': np.linalg.norm(grad) < tol
    }

def _compute_hessian(
    objective_func: Callable[[np.ndarray], float],
    params: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """Compute the Hessian matrix of the objective function."""
    n = len(params)
    hessian = np.zeros((n, n))
    for i in range(n):
        params_plus_i = params.copy()
        params_plus_i[i] += epsilon
        grad_plus_i = _compute_gradient(objective_func, params_plus_i)

        params_minus_i = params.copy()
        params_minus_i[i] -= epsilon
        grad_minus_i = _compute_gradient(objective_func, params_minus_i)

        hessian[:, i] = (grad_plus_i - grad_minus_i) / (2 * epsilon)
    return hessian

def _coordinate_descent(
    objective_func: Callable[[np.ndarray], float],
    initial_guess: np.ndarray,
    constraints: Optional[list] = None,
    bounds: Optional[tuple] = None,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> Dict[str, Any]:
    """Perform coordinate descent optimization."""
    params = initial_guess.copy()
    for i in range(max_iter):
        prev_params = params.copy()
        for j in range(len(params)):
            # Optimize one coordinate at a time
            other_params = params.copy()
            other_params[j] = 0

            def subproblem(x):
                other_params[j] = x
                return objective_func(other_params)

            params[j] = _optimize_scalar(subproblem, params[j], bounds[j][0], bounds[j][1]) if bounds is not None else _optimize_scalar(subproblem, params[j])

        # Check convergence
        if np.linalg.norm(params - prev_params) < tol:
            break

    return {
        'params': params,
        'iterations': i + 1,
        'converged': np.linalg.norm(params - prev_params) < tol
    }

def _optimize_scalar(
    func: Callable[[float], float],
    initial_guess: float,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None
) -> float:
    """Optimize a scalar function."""
    # Simple golden section search implementation
    invphi = (np.sqrt(5) - 1) / 2  # 1/phi
    invphi2 = (3 - np.sqrt(5)) / 2  # 1/phi^2

    x = initial_guess
    a, b = lower_bound if lower_bound is not None else -1000, upper_bound if upper_bound is not None else 1000
    h = b - a

    # Keep track of the best seen so far
    xmin = (a + b) / 2

    # Main loop
    while h > 1e-6:
        c = a + invphi2 * h
        d = a + invphi * h

        if func(c) < func(xmin):
            xmin = c
        else:
            a = c

        if func(d) < func(xmin):
            xmin = d
        else:
            b = d

        h *= invphi

    return xmin

def _calculate_metrics(
    params: np.ndarray,
    metric: Union[str, Callable[[np.ndarray], float]]
) -> Dict[str, float]:
    """Calculate metrics for the optimization result."""
    if callable(metric):
        return {'custom_metric': metric(params)}
    elif metric == 'mse':
        # In a real implementation, you would need the true values
        return {'mse': 0.0}
    elif metric == 'mae':
        # In a real implementation, you would need the true values
        return {'mae': 0.0}
    elif metric == 'r2':
        # In a real implementation, you would need the true values
        return {'r2': 0.0}
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _check_warnings(
    result: Dict[str, Any]
) -> list:
    """Check for warnings in the optimization result."""
    warnings = []
    if not result['converged']:
        warnings.append("Optimization did not converge")
    return warnings

################################################################################
# unconstrained_optimization
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def unconstrained_optimization_fit(
    objective_func: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    solver: str = 'gradient_descent',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    normalization: str = 'none',
    max_iter: int = 1000,
    tol: float = 1e-6,
    learning_rate: float = 0.01,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Perform unconstrained optimization using various solvers and metrics.

    Parameters:
    -----------
    objective_func : callable
        The objective function to minimize.
    initial_params : np.ndarray
        Initial guess for the parameters.
    solver : str, optional
        Solver to use ('gradient_descent', 'newton', 'coordinate_descent').
    metric : str or callable, optional
        Metric to evaluate ('mse', 'mae', 'r2', custom callable).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence.
    learning_rate : float, optional
        Learning rate for gradient descent.
    verbose : bool, optional
        Whether to print progress.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    validate_inputs(objective_func, initial_params)

    # Normalize parameters if needed
    params = apply_normalization(initial_params, normalization)

    # Choose solver
    if solver == 'gradient_descent':
        result = gradient_descent(objective_func, params, max_iter, tol, learning_rate, verbose)
    elif solver == 'newton':
        result = newton_method(objective_func, params, max_iter, tol, verbose)
    elif solver == 'coordinate_descent':
        result = coordinate_descent(objective_func, params, max_iter, tol, verbose)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = calculate_metrics(result['params'], metric)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {'solver': solver, 'normalization': normalization},
        'warnings': []
    }

def validate_inputs(objective_func: Callable, params: np.ndarray) -> None:
    """Validate the inputs for unconstrained optimization."""
    if not callable(objective_func):
        raise TypeError("objective_func must be a callable")
    if not isinstance(params, np.ndarray):
        raise TypeError("initial_params must be a numpy array")
    if np.any(np.isnan(params)) or np.any(np.isinf(params)):
        raise ValueError("initial_params must not contain NaN or inf values")

def apply_normalization(params: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to the parameters."""
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
        raise ValueError(f"Unknown normalization method: {method}")

def gradient_descent(
    objective_func: Callable[[np.ndarray], float],
    params: np.ndarray,
    max_iter: int,
    tol: float,
    learning_rate: float,
    verbose: bool
) -> Dict[str, Any]:
    """Perform gradient descent optimization."""
    for i in range(max_iter):
        grad = compute_gradient(objective_func, params)
        new_params = params - learning_rate * grad
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params
        if verbose:
            print(f"Iteration {i}, Objective: {objective_func(params)}")
    return {'params': params, 'iterations': i}

def newton_method(
    objective_func: Callable[[np.ndarray], float],
    params: np.ndarray,
    max_iter: int,
    tol: float,
    verbose: bool
) -> Dict[str, Any]:
    """Perform Newton's method optimization."""
    for i in range(max_iter):
        grad = compute_gradient(objective_func, params)
        hessian = compute_hessian(objective_func, params)
        new_params = params - np.linalg.solve(hessian, grad)
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params
        if verbose:
            print(f"Iteration {i}, Objective: {objective_func(params)}")
    return {'params': params, 'iterations': i}

def coordinate_descent(
    objective_func: Callable[[np.ndarray], float],
    params: np.ndarray,
    max_iter: int,
    tol: float,
    verbose: bool
) -> Dict[str, Any]:
    """Perform coordinate descent optimization."""
    for i in range(max_iter):
        old_params = params.copy()
        for j in range(len(params)):
            other_params = np.delete(params, j)
            params[j] = minimize_coordinate(objective_func, j, other_params)
        if np.linalg.norm(params - old_params) < tol:
            break
        if verbose:
            print(f"Iteration {i}, Objective: {objective_func(params)}")
    return {'params': params, 'iterations': i}

def compute_gradient(objective_func: Callable[[np.ndarray], float], params: np.ndarray) -> np.ndarray:
    """Compute the gradient of the objective function."""
    eps = 1e-8
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        params_minus = params.copy()
        params_minus[i] -= eps
        grad[i] = (objective_func(params_plus) - objective_func(params_minus)) / (2 * eps)
    return grad

def compute_hessian(objective_func: Callable[[np.ndarray], float], params: np.ndarray) -> np.ndarray:
    """Compute the Hessian matrix of the objective function."""
    eps = 1e-8
    n = len(params)
    hessian = np.zeros((n, n))
    for i in range(n):
        params_plus_i = params.copy()
        params_plus_i[i] += eps
        grad_plus_i = compute_gradient(objective_func, params_plus_i)
        params_minus_i = params.copy()
        params_minus_i[i] -= eps
        grad_minus_i = compute_gradient(objective_func, params_minus_i)
        hessian[:, i] = (grad_plus_i - grad_minus_i) / (2 * eps)
    return hessian

def minimize_coordinate(
    objective_func: Callable[[np.ndarray], float],
    coord_index: int,
    other_params: np.ndarray
) -> float:
    """Minimize the objective function along a single coordinate."""
    def coord_objective(x):
        params = np.insert(other_params, coord_index, x)
        return objective_func(params)

    # Simple line search for the coordinate
    x = 0.0
    step = 1.0
    for _ in range(100):
        x_new = x - step * compute_gradient(coord_objective, np.array([x]))[0]
        if abs(x_new - x) < 1e-6:
            break
        x = x_new
    return x

def calculate_metrics(params: np.ndarray, metric: Union[str, Callable]) -> Dict[str, float]:
    """Calculate the metrics for the optimized parameters."""
    if callable(metric):
        return {'custom_metric': metric(params, params)}
    elif metric == 'mse':
        return {'mse': np.mean((params - params) ** 2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(params - params))}
    elif metric == 'r2':
        return {'r2': 1.0}  # Placeholder for R-squared
    else:
        raise ValueError(f"Unknown metric: {metric}")

################################################################################
# global_optimization
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def global_optimization_fit(
    objective_function: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_iter: int = 1000,
    population_size: int = 50,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.7,
    normalization: str = 'none',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'genetic_algorithm',
    tol: float = 1e-6,
    max_evals: int = 10000,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform global optimization using various solvers.

    Parameters:
    -----------
    objective_function : callable
        The function to minimize. Must take a numpy array as input and return a float.
    bounds : np.ndarray
        Array of shape (n_params, 2) defining the lower and upper bounds for each parameter.
    n_iter : int, optional
        Number of iterations (default: 1000).
    population_size : int, optional
        Size of the population for genetic algorithm (default: 50).
    mutation_rate : float, optional
        Mutation rate for genetic algorithm (default: 0.1).
    crossover_rate : float, optional
        Crossover rate for genetic algorithm (default: 0.7).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'none').
    metric : str or callable, optional
        Metric to evaluate the solution ('mse', 'mae', 'r2', 'logloss') or custom callable (default: 'mse').
    solver : str, optional
        Solver to use ('genetic_algorithm', 'simulated_annealing', 'particle_swarm') (default: 'genetic_algorithm').
    tol : float, optional
        Tolerance for convergence (default: 1e-6).
    max_evals : int, optional
        Maximum number of evaluations (default: 10000).
    random_state : int, optional
        Random seed for reproducibility (default: None).

    Returns:
    --------
    dict
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(objective_function, bounds)

    # Initialize random state
    rng = np.random.RandomState(random_state)

    # Normalize bounds if required
    normalized_bounds = _normalize_bounds(bounds, normalization)

    # Choose solver
    if solver == 'genetic_algorithm':
        result = _genetic_algorithm(
            objective_function, normalized_bounds, n_iter, population_size,
            mutation_rate, crossover_rate, tol, max_evals, rng
        )
    elif solver == 'simulated_annealing':
        result = _simulated_annealing(
            objective_function, normalized_bounds, n_iter, tol, max_evals, rng
        )
    elif solver == 'particle_swarm':
        result = _particle_swarm(
            objective_function, normalized_bounds, n_iter, population_size,
            tol, max_evals, rng
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(result['params'], metric)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'n_iter': n_iter,
            'population_size': population_size,
            'mutation_rate': mutation_rate,
            'crossover_rate': crossover_rate,
            'normalization': normalization,
            'metric': metric,
            'solver': solver,
            'tol': tol,
            'max_evals': max_evals
        },
        'warnings': []
    }

def _validate_inputs(objective_function: Callable, bounds: np.ndarray) -> None:
    """Validate the inputs for global optimization."""
    if not callable(objective_function):
        raise TypeError("objective_function must be a callable")
    if not isinstance(bounds, np.ndarray) or bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must be a numpy array of shape (n_params, 2)")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Lower bound must be less than upper bound for each parameter")

def _normalize_bounds(bounds: np.ndarray, method: str) -> np.ndarray:
    """Normalize the bounds based on the specified method."""
    if method == 'none':
        return bounds
    elif method == 'standard':
        mean = np.mean(bounds, axis=1)
        std = np.std(bounds, axis=1)
        return (bounds - mean[:, np.newaxis]) / std[:, np.newaxis]
    elif method == 'minmax':
        min_val = np.min(bounds, axis=1)
        max_val = np.max(bounds, axis=1)
        return (bounds - min_val[:, np.newaxis]) / (max_val[:, np.newaxis] - min_val[:, np.newaxis])
    elif method == 'robust':
        median = np.median(bounds, axis=1)
        iqr = np.subtract(*np.percentile(bounds, [75, 25], axis=1))
        return (bounds - median[:, np.newaxis]) / iqr[:, np.newaxis]
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _genetic_algorithm(
    objective_function: Callable,
    bounds: np.ndarray,
    n_iter: int,
    population_size: int,
    mutation_rate: float,
    crossover_rate: float,
    tol: float,
    max_evals: int,
    rng: np.random.RandomState
) -> Dict[str, Any]:
    """Genetic algorithm solver for global optimization."""
    n_params = bounds.shape[0]
    population = rng.uniform(bounds[:, 0], bounds[:, 1], (population_size, n_params))
    best_solution = None
    best_score = float('inf')
    evals = 0

    for _ in range(n_iter):
        scores = np.array([objective_function(ind) for ind in population])
        evals += population_size

        if np.min(scores) < best_score - tol:
            best_idx = np.argmin(scores)
            best_solution = population[best_idx]
            best_score = scores[best_idx]

        if evals >= max_evals:
            break

        # Selection
        selected = population[np.argsort(scores)[:population_size // 2]]

        # Crossover
        children = []
        for i in range(0, len(selected), 2):
            if rng.rand() < crossover_rate:
                parent1, parent2 = selected[i], selected[i+1]
                crossover_point = rng.randint(1, n_params)
                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                children.extend([child1, child2])
            else:
                children.extend([selected[i], selected[i+1]])

        # Mutation
        for i in range(len(children)):
            if rng.rand() < mutation_rate:
                mutation_point = rng.randint(n_params)
                children[i][mutation_point] = rng.uniform(bounds[mutation_point, 0], bounds[mutation_point, 1])

        population = np.vstack([selected, np.array(children)])

    return {
        'params': best_solution,
        'score': best_score,
        'evals': evals
    }

def _simulated_annealing(
    objective_function: Callable,
    bounds: np.ndarray,
    n_iter: int,
    tol: float,
    max_evals: int,
    rng: np.random.RandomState
) -> Dict[str, Any]:
    """Simulated annealing solver for global optimization."""
    n_params = bounds.shape[0]
    current_solution = rng.uniform(bounds[:, 0], bounds[:, 1])
    best_solution = current_solution.copy()
    best_score = objective_function(current_solution)
    evals = 1
    temperature = 1.0

    for _ in range(n_iter):
        # Generate a neighbor
        neighbor = current_solution + rng.normal(0, 0.1, n_params)
        neighbor = np.clip(neighbor, bounds[:, 0], bounds[:, 1])
        neighbor_score = objective_function(neighbor)
        evals += 1

        if evals >= max_evals:
            break

        # Acceptance probability
        delta = neighbor_score - best_score
        if delta < 0 or rng.rand() < np.exp(-delta / temperature):
            current_solution = neighbor
            if delta < 0:
                best_solution = neighbor.copy()
                best_score = neighbor_score

        # Cooling schedule
        temperature *= 0.99

    return {
        'params': best_solution,
        'score': best_score,
        'evals': evals
    }

def _particle_swarm(
    objective_function: Callable,
    bounds: np.ndarray,
    n_iter: int,
    population_size: int,
    tol: float,
    max_evals: int,
    rng: np.random.RandomState
) -> Dict[str, Any]:
    """Particle swarm optimization solver for global optimization."""
    n_params = bounds.shape[0]
    particles = rng.uniform(bounds[:, 0], bounds[:, 1], (population_size, n_params))
    velocities = rng.uniform(-0.1, 0.1, (population_size, n_params))
    personal_best = particles.copy()
    personal_best_scores = np.array([objective_function(p) for p in particles])
    global_best_idx = np.argmin(personal_best_scores)
    global_best = personal_best[global_best_idx]
    global_best_score = personal_best_scores[global_best_idx]
    evals = population_size

    for _ in range(n_iter):
        r1, r2 = rng.rand(2)
        velocities = 0.5 * velocities + r1 * (personal_best - particles) + r2 * (global_best - particles)
        particles += velocities
        particles = np.clip(particles, bounds[:, 0], bounds[:, 1])

        current_scores = np.array([objective_function(p) for p in particles])
        evals += population_size

        if evals >= max_evals:
            break

        improved = current_scores < personal_best_scores
        personal_best[improved] = particles[improved]
        personal_best_scores[improved] = current_scores[improved]

        new_global_idx = np.argmin(personal_best_scores)
        if personal_best_scores[new_global_idx] < global_best_score - tol:
            global_best = personal_best[new_global_idx]
            global_best_score = personal_best_scores[new_global_idx]

    return {
        'params': global_best,
        'score': global_best_score,
        'evals': evals
    }

def _calculate_metrics(params: np.ndarray, metric: Union[str, Callable]) -> Dict[str, float]:
    """Calculate metrics for the optimization result."""
    if callable(metric):
        return {'custom_metric': metric(params)}
    elif metric == 'mse':
        return {'mse': np.mean(np.square(params))}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(params))}
    elif metric == 'r2':
        return {'r2': 1 - np.sum(np.square(params)) / len(params)}
    elif metric == 'logloss':
        return {'logloss': -np.mean(np.log(params + 1e-10))}
    else:
        raise ValueError(f"Unknown metric: {metric}")

################################################################################
# local_optimization
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def local_optimization_fit(
    objective_func: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    solver: str = 'gradient_descent',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    normalization: str = 'none',
    regularization: Optional[str] = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    learning_rate: float = 0.01,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Perform local optimization of a given objective function.

    Parameters:
    -----------
    objective_func : callable
        The objective function to minimize. Must accept a numpy array as input.
    initial_params : np.ndarray
        Initial parameters for the optimization.
    solver : str, optional
        Optimization algorithm to use. Options: 'gradient_descent', 'newton'.
    metric : str or callable, optional
        Metric to evaluate the optimization. Options: 'mse', 'mae', custom callable.
    normalization : str, optional
        Normalization method. Options: 'none', 'standard', 'minmax'.
    regularization : str, optional
        Regularization method. Options: 'none', 'l1', 'l2'.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence.
    learning_rate : float, optional
        Learning rate for gradient descent.
    verbose : bool, optional
        Whether to print progress information.

    Returns:
    --------
    dict
        Dictionary containing optimization results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(objective_func, initial_params)

    # Normalize data if required
    normalized_params = _apply_normalization(initial_params, normalization)

    # Initialize solver parameters
    params_used = {
        'solver': solver,
        'metric': metric,
        'normalization': normalization,
        'regularization': regularization,
        'max_iter': max_iter,
        'tol': tol,
        'learning_rate': learning_rate
    }

    # Choose solver
    if solver == 'gradient_descent':
        optimized_params = _gradient_descent(
            objective_func, normalized_params,
            max_iter=max_iter, tol=tol,
            learning_rate=learning_rate
        )
    elif solver == 'newton':
        optimized_params = _newton_method(
            objective_func, normalized_params,
            max_iter=max_iter, tol=tol
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    result = objective_func(optimized_params)
    metrics = _calculate_metrics(result, metric)

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': []
    }

    return output

def _validate_inputs(
    objective_func: Callable[[np.ndarray], float],
    initial_params: np.ndarray
) -> None:
    """
    Validate the inputs for local optimization.

    Parameters:
    -----------
    objective_func : callable
        The objective function to validate.
    initial_params : np.ndarray
        Initial parameters to validate.

    Raises:
    -------
    ValueError
        If inputs are invalid.
    """
    if not callable(objective_func):
        raise ValueError("objective_func must be a callable")
    if not isinstance(initial_params, np.ndarray):
        raise ValueError("initial_params must be a numpy array")
    if np.any(np.isnan(initial_params)) or np.any(np.isinf(initial_params)):
        raise ValueError("initial_params must not contain NaN or Inf values")

def _apply_normalization(
    params: np.ndarray,
    method: str
) -> np.ndarray:
    """
    Apply normalization to parameters.

    Parameters:
    -----------
    params : np.ndarray
        Parameters to normalize.
    method : str
        Normalization method. Options: 'none', 'standard', 'minmax'.

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
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _gradient_descent(
    objective_func: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-6,
    learning_rate: float = 0.01
) -> np.ndarray:
    """
    Perform gradient descent optimization.

    Parameters:
    -----------
    objective_func : callable
        The objective function to minimize.
    initial_params : np.ndarray
        Initial parameters for the optimization.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence.
    learning_rate : float, optional
        Learning rate for gradient descent.

    Returns:
    --------
    np.ndarray
        Optimized parameters.
    """
    params = initial_params.copy()
    for _ in range(max_iter):
        grad = _compute_gradient(objective_func, params)
        new_params = params - learning_rate * grad
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params
    return params

def _newton_method(
    objective_func: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> np.ndarray:
    """
    Perform Newton's method optimization.

    Parameters:
    -----------
    objective_func : callable
        The objective function to minimize.
    initial_params : np.ndarray
        Initial parameters for the optimization.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence.

    Returns:
    --------
    np.ndarray
        Optimized parameters.
    """
    params = initial_params.copy()
    for _ in range(max_iter):
        grad = _compute_gradient(objective_func, params)
        hessian = _compute_hessian(objective_func, params)
        new_params = params - np.linalg.solve(hessian, grad)
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params
    return params

def _compute_gradient(
    objective_func: Callable[[np.ndarray], float],
    params: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Compute the gradient of the objective function.

    Parameters:
    -----------
    objective_func : callable
        The objective function to differentiate.
    params : np.ndarray
        Parameters at which to compute the gradient.
    epsilon : float, optional
        Small value for numerical differentiation.

    Returns:
    --------
    np.ndarray
        Gradient of the objective function.
    """
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += epsilon
        params_minus = params.copy()
        params_minus[i] -= epsilon
        grad[i] = (objective_func(params_plus) - objective_func(params_minus)) / (2 * epsilon)
    return grad

def _compute_hessian(
    objective_func: Callable[[np.ndarray], float],
    params: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Compute the Hessian matrix of the objective function.

    Parameters:
    -----------
    objective_func : callable
        The objective function to differentiate.
    params : np.ndarray
        Parameters at which to compute the Hessian.
    epsilon : float, optional
        Small value for numerical differentiation.

    Returns:
    --------
    np.ndarray
        Hessian matrix of the objective function.
    """
    n = len(params)
    hessian = np.zeros((n, n))
    for i in range(n):
        params_plus_i = params.copy()
        params_plus_i[i] += epsilon
        grad_plus_i = _compute_gradient(objective_func, params_plus_i)

        params_minus_i = params.copy()
        params_minus_i[i] -= epsilon
        grad_minus_i = _compute_gradient(objective_func, params_minus_i)

        hessian[:, i] = (grad_plus_i - grad_minus_i) / (2 * epsilon)
    return hessian

def _calculate_metrics(
    result: float,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """
    Calculate metrics for the optimization result.

    Parameters:
    -----------
    result : float
        The result of the objective function.
    metric : str or callable
        Metric to calculate. Options: 'mse', 'mae', custom callable.

    Returns:
    --------
    dict
        Dictionary containing calculated metrics.
    """
    if callable(metric):
        return {'custom_metric': metric(result)}
    elif metric == 'mse':
        return {'mse': result}
    elif metric == 'mae':
        return {'mae': result}
    else:
        raise ValueError(f"Unknown metric: {metric}")

################################################################################
# multi_objective_optimization
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def multi_objective_optimization_fit(
    objective_functions: list,
    constraints: Optional[list] = None,
    bounds: Optional[np.ndarray] = None,
    normalization: str = 'standard',
    metrics: list = ['mse'],
    solver: str = 'gradient_descent',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for multi-objective optimization.

    Parameters:
    - objective_functions: List of callable functions representing the objectives to minimize.
    - constraints: List of callable functions representing constraints (optional).
    - bounds: Array-like, shape (n_parameters, 2), bounds for each parameter (optional).
    - normalization: Type of normalization to apply ('none', 'standard', 'minmax', 'robust').
    - metrics: List of metric names to compute ('mse', 'mae', 'r2', 'logloss') or custom callable.
    - solver: Solver to use ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    - tol: Tolerance for convergence.
    - max_iter: Maximum number of iterations.
    - custom_metric: Custom metric function (optional).
    - **kwargs: Additional solver-specific parameters.

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    validate_inputs(objective_functions, constraints, bounds)

    # Normalize objectives if required
    normalized_objectives = apply_normalization(objective_functions, normalization)

    # Choose solver and optimize
    if solver == 'gradient_descent':
        result = gradient_descent_optimize(
            normalized_objectives,
            constraints=constraints,
            bounds=bounds,
            tol=tol,
            max_iter=max_iter,
            **kwargs
        )
    elif solver == 'newton':
        result = newton_optimize(
            normalized_objectives,
            constraints=constraints,
            bounds=bounds,
            tol=tol,
            max_iter=max_iter,
            **kwargs
        )
    elif solver == 'coordinate_descent':
        result = coordinate_descent_optimize(
            normalized_objectives,
            constraints=constraints,
            bounds=bounds,
            tol=tol,
            max_iter=max_iter,
            **kwargs
        )
    else:
        raise ValueError("Unsupported solver")

    # Compute metrics
    metrics_result = compute_metrics(result['params'], objective_functions, metrics, custom_metric)

    return {
        'result': result,
        'metrics': metrics_result,
        'params_used': kwargs,
        'warnings': []
    }

def validate_inputs(
    objective_functions: list,
    constraints: Optional[list] = None,
    bounds: Optional[np.ndarray] = None
) -> None:
    """
    Validate input parameters.

    Parameters:
    - objective_functions: List of callable functions.
    - constraints: List of callable functions (optional).
    - bounds: Array-like, shape (n_parameters, 2) (optional).

    Raises:
    - ValueError: If inputs are invalid.
    """
    if not all(callable(f) for f in objective_functions):
        raise ValueError("All objective functions must be callable")
    if constraints is not None and not all(callable(c) for c in constraints):
        raise ValueError("All constraints must be callable")
    if bounds is not None and (len(bounds.shape) != 2 or bounds.shape[1] != 2):
        raise ValueError("Bounds must be a 2D array with shape (n_parameters, 2)")

def apply_normalization(
    objective_functions: list,
    normalization: str = 'standard'
) -> list:
    """
    Apply normalization to objective functions.

    Parameters:
    - objective_functions: List of callable functions.
    - normalization: Type of normalization to apply.

    Returns:
    - List of normalized objective functions.
    """
    if normalization == 'none':
        return objective_functions
    elif normalization == 'standard':
        return [standardize_objective(f) for f in objective_functions]
    elif normalization == 'minmax':
        return [minmax_normalize_objective(f) for f in objective_functions]
    elif normalization == 'robust':
        return [robust_normalize_objective(f) for f in objective_functions]
    else:
        raise ValueError("Unsupported normalization method")

def standardize_objective(
    objective_function: Callable
) -> Callable:
    """
    Standardize an objective function.

    Parameters:
    - objective_function: Callable function to standardize.

    Returns:
    - Standardized callable function.
    """
    # Implementation of standardization
    def standardized_func(*args, **kwargs):
        return objective_function(*args, **kwargs) / np.std(objective_function(np.random.rand(10)))
    return standardized_func

def minmax_normalize_objective(
    objective_function: Callable
) -> Callable:
    """
    Min-max normalize an objective function.

    Parameters:
    - objective_function: Callable function to normalize.

    Returns:
    - Normalized callable function.
    """
    # Implementation of min-max normalization
    def normalized_func(*args, **kwargs):
        return (objective_function(*args, **kwargs) - np.min(objective_function(np.random.rand(10)))) / \
               (np.max(objective_function(np.random.rand(10))) - np.min(objective_function(np.random.rand(10))))
    return normalized_func

def robust_normalize_objective(
    objective_function: Callable
) -> Callable:
    """
    Robustly normalize an objective function.

    Parameters:
    - objective_function: Callable function to normalize.

    Returns:
    - Normalized callable function.
    """
    # Implementation of robust normalization
    def normalized_func(*args, **kwargs):
        values = objective_function(np.random.rand(10))
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        return (objective_function(*args, **kwargs) - np.median(values)) / iqr
    return normalized_func

def gradient_descent_optimize(
    objective_functions: list,
    constraints: Optional[list] = None,
    bounds: Optional[np.ndarray] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    learning_rate: float = 0.01
) -> Dict[str, Any]:
    """
    Optimize using gradient descent.

    Parameters:
    - objective_functions: List of callable functions.
    - constraints: List of callable functions (optional).
    - bounds: Array-like, shape (n_parameters, 2) (optional).
    - tol: Tolerance for convergence.
    - max_iter: Maximum number of iterations.
    - learning_rate: Learning rate for gradient descent.

    Returns:
    - Dictionary containing optimization results.
    """
    # Implementation of gradient descent
    params = np.random.rand(len(objective_functions))
    for _ in range(max_iter):
        gradients = [compute_gradient(f, params) for f in objective_functions]
        params -= learning_rate * np.mean(gradients, axis=0)
        if np.linalg.norm(gradients) < tol:
            break
    return {'params': params, 'converged': True}

def compute_gradient(
    objective_function: Callable,
    params: np.ndarray
) -> np.ndarray:
    """
    Compute the gradient of an objective function.

    Parameters:
    - objective_function: Callable function.
    - params: Current parameters.

    Returns:
    - Gradient as a numpy array.
    """
    # Implementation of gradient computation
    epsilon = 1e-8
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += epsilon
        params_minus = params.copy()
        params_minus[i] -= epsilon
        grad[i] = (objective_function(*params_plus) - objective_function(*params_minus)) / (2 * epsilon)
    return grad

def compute_metrics(
    params: np.ndarray,
    objective_functions: list,
    metrics: list,
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """
    Compute metrics for the optimization results.

    Parameters:
    - params: Optimized parameters.
    - objective_functions: List of callable functions.
    - metrics: List of metric names or custom callable.
    - custom_metric: Custom metric function (optional).

    Returns:
    - Dictionary of computed metrics.
    """
    results = {}
    for metric in metrics:
        if callable(metric):
            results['custom_metric'] = metric(params, objective_functions)
        elif metric == 'mse':
            results['mse'] = mean_squared_error(params, objective_functions)
        elif metric == 'mae':
            results['mae'] = mean_absolute_error(params, objective_functions)
        elif metric == 'r2':
            results['r2'] = r_squared(params, objective_functions)
        elif metric == 'logloss':
            results['logloss'] = log_loss(params, objective_functions)
    if custom_metric is not None:
        results['custom_metric'] = custom_metric(params, objective_functions)
    return results

def mean_squared_error(
    params: np.ndarray,
    objective_functions: list
) -> float:
    """
    Compute mean squared error.

    Parameters:
    - params: Optimized parameters.
    - objective_functions: List of callable functions.

    Returns:
    - Mean squared error.
    """
    return np.mean([f(*params) ** 2 for f in objective_functions])

def mean_absolute_error(
    params: np.ndarray,
    objective_functions: list
) -> float:
    """
    Compute mean absolute error.

    Parameters:
    - params: Optimized parameters.
    - objective_functions: List of callable functions.

    Returns:
    - Mean absolute error.
    """
    return np.mean([np.abs(f(*params)) for f in objective_functions])

def r_squared(
    params: np.ndarray,
    objective_functions: list
) -> float:
    """
    Compute R-squared.

    Parameters:
    - params: Optimized parameters.
    - objective_functions: List of callable functions.

    Returns:
    - R-squared value.
    """
    # Implementation of R-squared
    return 1.0 - np.mean([f(*params) ** 2 for f in objective_functions]) / np.var(objective_functions)

def log_loss(
    params: np.ndarray,
    objective_functions: list
) -> float:
    """
    Compute log loss.

    Parameters:
    - params: Optimized parameters.
    - objective_functions: List of callable functions.

    Returns:
    - Log loss value.
    """
    # Implementation of log loss
    return -np.mean([f(*params) * np.log(f(*params)) for f in objective_functions])

# Example usage
if __name__ == "__main__":
    # Define example objective functions
    def f1(x):
        return x[0]**2 + x[1]**2

    def f2(x):
        return (x[0] - 1)**2 + x[1]**2

    # Run optimization
    result = multi_objective_optimization_fit(
        objective_functions=[f1, f2],
        solver='gradient_descent',
        normalization='standard'
    )

################################################################################
# hyperparameter_tuning
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def hyperparameter_tuning_fit(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict[str, Any],
    scoring: Union[str, Callable] = 'mse',
    cv: int = 5,
    normalize: Optional[str] = None,
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning for a given model.

    Parameters:
    -----------
    model : Callable
        The model to tune. Should accept parameters and return a fitted model.
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    param_grid : Dict[str, Any]
        Dictionary with parameters names as keys and lists of parameter settings to try.
    scoring : Union[str, Callable], optional
        Scoring method to evaluate the predictions on the test set.
    cv : int, optional
        Number of folds for cross-validation.
    normalize : Optional[str], optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    solver : str, optional
        Solver to use: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'.
    regularization : Optional[str], optional
        Regularization method: 'none', 'l1', 'l2', or 'elasticnet'.
    tol : float, optional
        Tolerance for stopping criteria.
    max_iter : int, optional
        Maximum number of iterations.
    random_state : Optional[int], optional
        Random seed for reproducibility.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the results of hyperparameter tuning.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_normalized = _apply_normalization(X, normalize)

    # Perform hyperparameter tuning
    best_params, best_score = _grid_search(
        model=model,
        X=X_normalized,
        y=y,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state
    )

    # Fit the best model on the entire dataset
    best_model = _fit_best_model(
        model=model,
        X=X_normalized,
        y=y,
        params=best_params,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter
    )

    # Calculate metrics on the best model
    metrics = _calculate_metrics(best_model, X_normalized, y, scoring)

    return {
        "result": best_model,
        "metrics": metrics,
        "params_used": best_params,
        "warnings": []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate the input data."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input data contains NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input data contains infinite values.")

def _apply_normalization(X: np.ndarray, method: Optional[str]) -> np.ndarray:
    """Apply normalization to the feature matrix."""
    if method is None:
        return X
    elif method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _grid_search(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict[str, Any],
    scoring: Union[str, Callable],
    cv: int,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    random_state: Optional[int]
) -> tuple[Dict[str, Any], float]:
    """Perform grid search over the parameter grid."""
    best_score = -np.inf
    best_params = {}

    # Generate all parameter combinations
    from itertools import product
    param_combinations = [dict(zip(param_grid.keys(), p)) for p in product(*param_grid.values())]

    for params in param_combinations:
        # Perform cross-validation
        scores = _cross_validate(
            model=model,
            X=X,
            y=y,
            params=params,
            scoring=scoring,
            cv=cv,
            solver=solver,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state
        )
        mean_score = np.mean(scores)

        # Update best parameters if current score is better
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    return best_params, best_score

def _cross_validate(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any],
    scoring: Union[str, Callable],
    cv: int,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    random_state: Optional[int]
) -> np.ndarray:
    """Perform cross-validation."""
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit the model
        fitted_model = _fit_best_model(
            model=model,
            X=X_train,
            y=y_train,
            params=params,
            solver=solver,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter
        )

        # Calculate the score
        score = _calculate_metric(fitted_model, X_test, y_test, scoring)
        scores.append(score)

    return np.array(scores)

def _fit_best_model(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any],
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Any:
    """Fit the best model with given parameters."""
    # Update model parameters
    model_params = params.copy()
    model_params.update({
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'max_iter': max_iter
    })

    # Fit the model
    return model(X, y, **model_params)

def _calculate_metrics(model: Any, X: np.ndarray, y: np.ndarray, scoring: Union[str, Callable]) -> Dict[str, float]:
    """Calculate metrics for the given model."""
    y_pred = model.predict(X)
    return {'score': _calculate_metric(model, X, y, scoring)}

def _calculate_metric(model: Any, X: np.ndarray, y: np.ndarray, scoring: Union[str, Callable]) -> float:
    """Calculate the specified metric."""
    y_pred = model.predict(X)

    if scoring == 'mse':
        return np.mean((y - y_pred) ** 2)
    elif scoring == 'mae':
        return np.mean(np.abs(y - y_pred))
    elif scoring == 'r2':
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    elif scoring == 'logloss':
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    elif callable(scoring):
        return scoring(y, y_pred)
    else:
        raise ValueError(f"Unknown scoring method: {scoring}")

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
    surrogate_model: Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], float]] = None,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Callable[[np.ndarray, np.ndarray], float] = None,
    solver: str = 'lbfgs',
    tol: float = 1e-5,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform Bayesian optimization to minimize an objective function.

    Parameters:
    -----------
    objective_function : callable
        The objective function to minimize.
    bounds : np.ndarray
        Array of shape (n_features, 2) defining the search space.
    n_iter : int, optional
        Number of iterations to run (default: 100).
    acquisition_function : callable, optional
        Acquisition function to use (default: None).
    surrogate_model : callable, optional
        Surrogate model to fit (default: None).
    normalizer : callable, optional
        Normalization function for the input space (default: None).
    metric : callable, optional
        Metric to evaluate the optimization result (default: None).
    solver : str, optional
        Solver to use for optimization ('lbfgs', 'cg', 'newton') (default: 'lbfgs').
    tol : float, optional
        Tolerance for optimization convergence (default: 1e-5).
    random_state : int, optional
        Random seed for reproducibility (default: None).

    Returns:
    --------
    dict
        Dictionary containing the optimization result, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(objective_function, bounds)

    # Initialize random state if provided
    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    # Initialize surrogate model and acquisition function
    surrogate_model = _initialize_surrogate_model(surrogate_model)
    acquisition_function = _initialize_acquisition_function(acquisition_function)

    # Initialize optimization parameters
    best_params = None
    best_value = np.inf

    # Optimization loop
    for _ in range(n_iter):
        # Sample next point to evaluate
        next_point = _sample_next_point(bounds, surrogate_model, acquisition_function, rng)

        # Evaluate objective function
        value = objective_function(next_point)

        # Update best parameters and value
        if value < best_value:
            best_params = next_point.copy()
            best_value = value

        # Update surrogate model
        _update_surrogate_model(surrogate_model, best_params, best_value)

    # Compute metrics if provided
    metrics = _compute_metrics(best_params, best_value, metric)

    # Return results
    return {
        'result': {'params': best_params, 'value': best_value},
        'metrics': metrics,
        'params_used': {
            'n_iter': n_iter,
            'solver': solver,
            'tol': tol
        },
        'warnings': []
    }

def _validate_inputs(objective_function: Callable, bounds: np.ndarray) -> None:
    """
    Validate the inputs for Bayesian optimization.

    Parameters:
    -----------
    objective_function : callable
        The objective function to minimize.
    bounds : np.ndarray
        Array of shape (n_features, 2) defining the search space.
    """
    if not callable(objective_function):
        raise ValueError("objective_function must be a callable.")
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must be a 2D array of shape (n_features, 2).")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Lower bound must be less than upper bound for each feature.")

def _initialize_surrogate_model(surrogate_model: Optional[Callable]) -> Callable:
    """
    Initialize the surrogate model.

    Parameters:
    -----------
    surrogate_model : callable, optional
        Surrogate model to fit (default: None).

    Returns:
    --------
    callable
        The initialized surrogate model.
    """
    if surrogate_model is None:
        # Default to a simple Gaussian Process surrogate model
        def default_surrogate_model(X: np.ndarray, y: np.ndarray) -> Callable:
            def model(x: np.ndarray) -> float:
                return np.mean(y)
            return model
        surrogate_model = default_surrogate_model
    return surrogate_model

def _initialize_acquisition_function(acquisition_function: Optional[Callable]) -> Callable:
    """
    Initialize the acquisition function.

    Parameters:
    -----------
    acquisition_function : callable, optional
        Acquisition function to use (default: None).

    Returns:
    --------
    callable
        The initialized acquisition function.
    """
    if acquisition_function is None:
        # Default to Expected Improvement
        def default_acquisition_function(X: np.ndarray, y: np.ndarray, model: Callable) -> float:
            return -np.mean(y)
        acquisition_function = default_acquisition_function
    return acquisition_function

def _sample_next_point(
    bounds: np.ndarray,
    surrogate_model: Callable,
    acquisition_function: Callable,
    rng: np.random.RandomState
) -> np.ndarray:
    """
    Sample the next point to evaluate.

    Parameters:
    -----------
    bounds : np.ndarray
        Array of shape (n_features, 2) defining the search space.
    surrogate_model : callable
        The surrogate model to use.
    acquisition_function : callable
        The acquisition function to use.
    rng : np.random.RandomState
        Random number generator.

    Returns:
    --------
    np.ndarray
        The next point to evaluate.
    """
    # Sample a random point within the bounds
    return rng.uniform(bounds[:, 0], bounds[:, 1])

def _update_surrogate_model(
    surrogate_model: Callable,
    X_new: np.ndarray,
    y_new: float
) -> None:
    """
    Update the surrogate model with new observations.

    Parameters:
    -----------
    surrogate_model : callable
        The surrogate model to update.
    X_new : np.ndarray
        New input point.
    y_new : float
        New output value.
    """
    pass  # Placeholder for actual surrogate model update logic

def _compute_metrics(
    best_params: np.ndarray,
    best_value: float,
    metric: Optional[Callable]
) -> Dict[str, Any]:
    """
    Compute metrics for the optimization result.

    Parameters:
    -----------
    best_params : np.ndarray
        The best parameters found.
    best_value : float
        The best value found.
    metric : callable, optional
        Metric to evaluate the optimization result (default: None).

    Returns:
    --------
    dict
        Dictionary containing the computed metrics.
    """
    if metric is None:
        return {}
    return {'custom_metric': metric(best_params, best_value)}

################################################################################
# grid_search
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
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def normalize_data(X: np.ndarray, y: np.ndarray,
                  normalizer: Callable[[np.ndarray], np.ndarray] = None) -> tuple:
    """Normalize input data using provided normalizer."""
    if normalizer is None:
        return X, y
    X_norm = normalizer(X)
    if len(y.shape) == 1:
        y_norm = normalizer(y.reshape(-1, 1)).flatten()
    else:
        y_norm = normalizer(y)
    return X_norm, y_norm

def compute_metric(y_true: np.ndarray, y_pred: np.ndarray,
                  metric_func: Callable[[np.ndarray, np.ndarray], float]) -> float:
    """Compute evaluation metric between true and predicted values."""
    return metric_func(y_true, y_pred)

def grid_search_fit(X: np.ndarray, y: np.ndarray,
                   param_grid: Dict[str, Any],
                   metric_func: Callable[[np.ndarray, np.ndarray], float] = None,
                   normalizer: Callable[[np.ndarray], np.ndarray] = None,
                   solver_func: Callable[..., Any] = None) -> Dict[str, Any]:
    """
    Perform grid search over parameter space.

    Parameters:
    -----------
    X : np.ndarray
        Input features (n_samples, n_features)
    y : np.ndarray
        Target values (n_samples,)
    param_grid : dict
        Dictionary of parameter grids to search over
    metric_func : callable, optional
        Function to compute evaluation metric (default: MSE)
    normalizer : callable, optional
        Function to normalize data (default: None)
    solver_func : callable, optional
        Solver function to optimize parameters (default: None)

    Returns:
    --------
    dict
        Dictionary containing best results, metrics, parameters used and warnings

    Example:
    --------
    >>> from sklearn.metrics import mean_squared_error
    >>> def standard_normalizer(X): return (X - X.mean(axis=0)) / X.std(axis=0)
    >>> param_grid = {'alpha': [0.1, 1.0, 10.0], 'max_iter': [100, 200]}
    >>> result = grid_search_fit(X_train, y_train, param_grid,
    ...                         metric_func=mean_squared_error,
    ...                         normalizer=standard_normalizer)
    """
    # Validate inputs
    validate_inputs(X, y)

    # Set default metric if not provided
    if metric_func is None:
        def mse(y_true, y_pred):
            return np.mean((y_true - y_pred) ** 2)
        metric_func = mse

    # Normalize data
    X_norm, y_norm = normalize_data(X, y, normalizer)

    # Initialize results storage
    best_score = np.inf
    best_params = {}
    best_result = None
    warnings_list = []

    # Generate parameter combinations
    from itertools import product
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    param_combinations = product(*param_values)

    # Perform grid search
    for params in param_combinations:
        try:
            # Create parameter dictionary
            current_params = dict(zip(param_names, params))

            # Run solver with current parameters
            if solver_func is None:
                raise ValueError("No solver function provided")
            result = solver_func(X_norm, y_norm, **current_params)

            # Compute metric
            if 'predict' not in result:
                warnings_list.append("Solver result missing predict function")
                continue

            y_pred = result['predict'](X_norm)
            current_score = compute_metric(y_norm, y_pred, metric_func)

            # Update best result if current is better
            if current_score < best_score:
                best_score = current_score
                best_params = current_params.copy()
                best_result = result

        except Exception as e:
            warnings_list.append(f"Error with params {current_params}: {str(e)}")
            continue

    # Prepare output
    output = {
        "result": best_result,
        "metrics": {"best_score": best_score},
        "params_used": best_params,
        "warnings": warnings_list
    }

    return output

def _closed_form_solver(X: np.ndarray, y: np.ndarray,
                       **kwargs) -> Dict[str, Any]:
    """Closed form solution for linear regression."""
    X_tX = X.T @ X
    if np.linalg.det(X_tX) == 0:
        raise ValueError("Matrix is singular")
    coef = np.linalg.inv(X_tX) @ X.T @ y

    def predict(X_new: np.ndarray) -> np.ndarray:
        return X_new @ coef

    return {
        "coef": coef,
        "predict": predict
    }

def _gradient_descent_solver(X: np.ndarray, y: np.ndarray,
                           learning_rate: float = 0.01,
                           max_iter: int = 1000,
                           tol: float = 1e-4) -> Dict[str, Any]:
    """Gradient descent solver."""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)
    prev_loss = np.inf

    for _ in range(max_iter):
        gradients = 2/n_samples * X.T @ (X @ coef - y)
        coef -= learning_rate * gradients
        current_loss = np.mean((X @ coef - y) ** 2)

        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    def predict(X_new: np.ndarray) -> np.ndarray:
        return X_new @ coef

    return {
        "coef": coef,
        "predict": predict
    }

def _newton_solver(X: np.ndarray, y: np.ndarray,
                  max_iter: int = 100) -> Dict[str, Any]:
    """Newton's method solver."""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)

    for _ in range(max_iter):
        gradients = 2/n_samples * X.T @ (X @ coef - y)
        hessian = 2/n_samples * X.T @ X
        coef -= np.linalg.inv(hessian) @ gradients

    def predict(X_new: np.ndarray) -> np.ndarray:
        return X_new @ coef

    return {
        "coef": coef,
        "predict": predict
    }

################################################################################
# random_search
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def random_search_fit(
    objective_function: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_iterations: int = 100,
    random_state: Optional[int] = None,
    metric: Union[str, Callable[[np.ndarray], float]] = 'minimize',
    normalization: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform random search optimization.

    Parameters:
    -----------
    objective_function : callable
        Function to minimize or maximize. Must accept a numpy array as input.
    bounds : np.ndarray
        Array of shape (n_params, 2) defining the search space for each parameter.
    n_iterations : int, optional
        Number of random samples to evaluate (default: 100).
    random_state : int, optional
        Random seed for reproducibility (default: None).
    metric : str or callable, optional
        Optimization direction ('minimize' or 'maximize') or custom metric function (default: 'minimize').
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax') (default: None).
    **kwargs : dict
        Additional keyword arguments passed to the objective function.

    Returns:
    --------
    result : dict
        Dictionary containing optimization results, metrics, parameters used and warnings.
    """
    # Initialize random state
    rng = np.random.RandomState(random_state)

    # Validate inputs
    _validate_inputs(objective_function, bounds, n_iterations, metric)

    # Normalize bounds if required
    normalized_bounds = _apply_normalization(bounds, normalization)

    # Initialize best parameters and score
    best_params = None
    best_score = float('inf') if metric == 'minimize' else float('-inf')

    # Perform random search
    for _ in range(n_iterations):
        # Generate random parameters within bounds
        params = rng.uniform(low=normalized_bounds[:, 0], high=normalized_bounds[:, 1])

        # Evaluate objective function
        score = objective_function(params, **kwargs)

        # Update best parameters if needed
        if (metric == 'minimize' and score < best_score) or \
           (metric == 'maximize' and score > best_score):
            best_params = params.copy()
            best_score = score

    # Prepare results
    result = {
        'result': best_params,
        'metrics': {'best_score': best_score},
        'params_used': {
            'n_iterations': n_iterations,
            'random_state': random_state,
            'metric': metric,
            'normalization': normalization
        },
        'warnings': []
    }

    return result

def _validate_inputs(
    objective_function: Callable,
    bounds: np.ndarray,
    n_iterations: int,
    metric: Union[str, Callable]
) -> None:
    """
    Validate input parameters for random search.

    Parameters:
    -----------
    objective_function : callable
        Function to validate.
    bounds : np.ndarray
        Parameter bounds to validate.
    n_iterations : int
        Number of iterations to validate.
    metric : str or callable
        Optimization direction to validate.

    Raises:
    -------
    ValueError
        If any input parameter is invalid.
    """
    if not callable(objective_function):
        raise ValueError("objective_function must be a callable")

    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must be a 2D array of shape (n_params, 2)")

    if n_iterations <= 0:
        raise ValueError("n_iterations must be positive")

    if metric not in ['minimize', 'maximize'] and not callable(metric):
        raise ValueError("metric must be either 'minimize', 'maximize' or a callable")

def _apply_normalization(
    bounds: np.ndarray,
    normalization: Optional[str]
) -> np.ndarray:
    """
    Apply normalization to parameter bounds.

    Parameters:
    -----------
    bounds : np.ndarray
        Original parameter bounds.
    normalization : str, optional
        Normalization method to apply.

    Returns:
    --------
    np.ndarray
        Normalized bounds.
    """
    if normalization is None or normalization == 'none':
        return bounds.copy()

    normalized_bounds = np.zeros_like(bounds)

    if normalization == 'standard':
        # Standardize to mean=0, std=1
        for i in range(bounds.shape[0]):
            low, high = bounds[i]
            mean = (low + high) / 2
            std = (high - low) / 4
            normalized_bounds[i] = [mean - std, mean + std]

    elif normalization == 'minmax':
        # Scale to [0, 1]
        for i in range(bounds.shape[0]):
            low, high = bounds[i]
            normalized_bounds[i] = [0, 1]

    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    return normalized_bounds

################################################################################
# adam
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    loss_func: Callable[[np.ndarray, np.ndarray], float],
    gradient_func: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> None:
    """Validate input data and functions."""
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

def default_loss_func(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Default mean squared error loss function."""
    return np.mean((y_true - y_pred) ** 2)

def default_gradient_func(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Default gradient function for MSE loss."""
    predictions = X @ weights
    error = predictions - y
    gradient = 2 * X.T @ error / len(y)
    return gradient

def adam_fit(
    X: np.ndarray,
    y: np.ndarray,
    loss_func: Callable[[np.ndarray, np.ndarray], float] = default_loss_func,
    gradient_func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = default_gradient_func,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Perform optimization using the Adam algorithm.

    Parameters:
    - X: Input features (n_samples, n_features)
    - y: Target values (n_samples,)
    - loss_func: Loss function to minimize
    - gradient_func: Gradient of the loss function
    - learning_rate: Learning rate for optimization
    - beta1: Exponential decay rate for the first moment estimates
    - beta2: Exponential decay rate for the second moment estimates
    - epsilon: Small constant to prevent division by zero
    - max_iter: Maximum number of iterations
    - tol: Tolerance for stopping criterion
    - verbose: Whether to print progress

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate inputs
    validate_inputs(X, y, loss_func, gradient_func)

    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    m = np.zeros_like(weights)
    v = np.zeros_like(weights)

    best_loss = float('inf')
    best_weights = None
    history = []

    for t in range(1, max_iter + 1):
        # Compute predictions and loss
        y_pred = X @ weights
        current_loss = loss_func(y, y_pred)

        # Update best weights if current loss is better
        if current_loss < best_loss - tol:
            best_loss = current_loss
            best_weights = weights.copy()

        # Compute gradient
        grad = gradient_func(X, y, weights)

        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * grad

        # Update biased second moment estimate
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - beta1 ** t)

        # Compute bias-corrected second moment estimate
        v_hat = v / (1 - beta2 ** t)

        # Update weights
        weights -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        # Store history
        history.append(current_loss)

        if verbose and t % 100 == 0:
            print(f"Iteration {t}, Loss: {current_loss}")

        # Check for convergence
        if t > 1 and abs(history[-2] - history[-1]) < tol:
            if verbose:
                print(f"Converged at iteration {t}")
            break

    # Calculate final metrics
    y_pred = X @ best_weights
    final_loss = loss_func(y, y_pred)

    # Prepare output dictionary
    result = {
        "result": best_weights,
        "metrics": {
            "final_loss": final_loss,
            "history": history
        },
        "params_used": {
            "learning_rate": learning_rate,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
            "max_iter": max_iter,
            "tol": tol
        },
        "warnings": []
    }

    return result

# Example usage:
"""
X = np.random.rand(100, 5)
y = np.random.rand(100)

result = adam_fit(X, y, learning_rate=0.1, max_iter=500)
print(result['result'])
"""

################################################################################
# rmsprop
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    rho: float = 0.9,
    epsilon: float = 1e-8,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> None:
    """Validate input parameters for RMSprop."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not (0 < rho < 1):
        raise ValueError("rho must be in (0, 1)")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol <= 0:
        raise ValueError("tol must be positive")

def compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    loss_func: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """Compute the gradient of the loss function."""
    n_samples = X.shape[0]
    predictions = np.dot(X, weights)
    gradient = (1 / n_samples) * np.dot(X.T, (predictions - y))
    return gradient

def rmsprop_fit(
    X: np.ndarray,
    y: np.ndarray,
    loss_func: Callable[[np.ndarray, np.ndarray], float] = lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
    learning_rate: float = 0.01,
    rho: float = 0.9,
    epsilon: float = 1e-8,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Perform RMSprop optimization to minimize the given loss function.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    loss_func : Callable[[np.ndarray, np.ndarray], float]
        Loss function to minimize.
    learning_rate : float
        Learning rate for the optimizer.
    rho : float
        Decay rate for the moving average of squared gradients.
    epsilon : float
        Small constant to avoid division by zero.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for stopping criterion.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], str]]
        Dictionary containing the optimized weights, metrics, parameters used, and warnings.
    """
    validate_inputs(X, y, learning_rate, rho, epsilon, max_iter, tol)

    n_features = X.shape[1]
    weights = np.zeros(n_features)
    squared_gradients = np.zeros_like(weights)

    for i in range(max_iter):
        gradient = compute_gradient(X, y, weights, loss_func)
        squared_gradients = rho * squared_gradients + (1 - rho) * gradient ** 2
        weights -= learning_rate * gradient / (np.sqrt(squared_gradients) + epsilon)

        if i % 100 == 0:
            current_loss = loss_func(y, np.dot(X, weights))
            if i > 0 and abs(prev_loss - current_loss) < tol:
                break
            prev_loss = current_loss

    metrics = {
        'loss': loss_func(y, np.dot(X, weights)),
        'iterations': i + 1
    }

    return {
        'result': weights,
        'metrics': metrics,
        'params_used': {
            'learning_rate': learning_rate,
            'rho': rho,
            'epsilon': epsilon,
            'max_iter': max_iter,
            'tol': tol
        },
        'warnings': []
    }

# Example usage:
"""
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([1, 2, 3])
result = rmsprop_fit(X, y)
print(result)
"""

################################################################################
# adagrad
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    epsilon: float = 1e-8,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> None:
    """
    Validate input data and parameters for Adagrad optimization.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    learning_rate : float, optional
        Learning rate for gradient descent.
    epsilon : float, optional
        Small constant to avoid division by zero.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for stopping criterion.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("X and y must be numpy arrays.")
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if learning_rate <= 0:
        raise ValueError("Learning rate must be positive.")
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive.")
    if max_iter <= 0:
        raise ValueError("Max iterations must be positive.")
    if tol <= 0:
        raise ValueError("Tolerance must be positive.")

def compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray
) -> np.ndarray:
    """
    Compute the gradient of the loss function.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    weights : np.ndarray
        Current weights.

    Returns
    ------
    np.ndarray
        Gradient of the loss function.
    """
    predictions = X.dot(weights)
    errors = predictions - y
    gradient = X.T.dot(errors) / len(y)
    return gradient

def adagrad_update(
    weights: np.ndarray,
    gradient: np.ndarray,
    learning_rate: float,
    epsilon: float,
    G: np.ndarray
) -> tuple:
    """
    Perform Adagrad update.

    Parameters
    ----------
    weights : np.ndarray
        Current weights.
    gradient : np.ndarray
        Gradient of the loss function.
    learning_rate : float
        Learning rate.
    epsilon : float
        Small constant to avoid division by zero.
    G : np.ndarray
        Sum of squared gradients.

    Returns
    ------
    tuple
        Updated weights and sum of squared gradients.
    """
    G += gradient ** 2
    adjusted_gradient = learning_rate * gradient / (np.sqrt(G) + epsilon)
    weights -= adjusted_gradient
    return weights, G

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics for the model.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.

    Returns
    ------
    Dict[str, float]
        Dictionary of metrics.
    """
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    return {"mse": mse, "mae": mae}

def adagrad_fit(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    epsilon: float = 1e-8,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, float], list]]:
    """
    Perform Adagrad optimization to fit a linear model.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    learning_rate : float, optional
        Learning rate for gradient descent.
    epsilon : float, optional
        Small constant to avoid division by zero.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for stopping criterion.

    Returns
    ------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, float], list]]
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    validate_inputs(X, y, learning_rate, epsilon, max_iter, tol)

    n_features = X.shape[1]
    weights = np.zeros(n_features)
    G = np.zeros(n_features)
    previous_loss = float('inf')

    for iteration in range(max_iter):
        gradient = compute_gradient(X, y, weights)
        weights, G = adagrad_update(weights, gradient, learning_rate, epsilon, G)
        predictions = X.dot(weights)
        current_loss = np.mean((predictions - y) ** 2)

        if abs(previous_loss - current_loss) < tol:
            break
        previous_loss = current_loss

    metrics = compute_metrics(y, predictions)
    result = {
        "result": weights,
        "metrics": metrics,
        "params_used": {
            "learning_rate": learning_rate,
            "epsilon": epsilon,
            "max_iter": iteration + 1,
            "tol": tol
        },
        "warnings": []
    }

    return result

# Example usage:
# X = np.random.rand(100, 5)
# y = np.random.rand(100)
# result = adagrad_fit(X, y)

################################################################################
# adadelta
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Union[float, int]],
    loss_func: Callable[[np.ndarray, np.ndarray], float],
    grad_func: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> None:
    """Validate input data and parameters."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or Inf values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or Inf values")
    if 'rho' in params and (params['rho'] <= 0 or params['rho'] >= 1):
        raise ValueError("rho must be in (0, 1)")
    if 'epsilon' in params and params['epsilon'] <= 0:
        raise ValueError("epsilon must be positive")
    if 'max_iter' in params and params['max_iter'] <= 0:
        raise ValueError("max_iter must be positive")
    if 'tol' in params and params['tol'] <= 0:
        raise ValueError("tol must be positive")

def compute_gradients(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray
) -> np.ndarray:
    """Compute gradients for the given parameters."""
    return X.T @ (X @ params - y) / len(y)

def update_parameters(
    grad: np.ndarray,
    prev_grad_sq: np.ndarray,
    prev_param_update_sq: np.ndarray,
    rho: float,
    epsilon: float
) -> tuple:
    """Update parameters using Adadelta algorithm."""
    grad_sq = rho * prev_grad_sq + (1 - rho) * (grad ** 2)
    param_update = np.sqrt(prev_param_update_sq + epsilon) / np.sqrt(grad_sq + epsilon) * grad
    param_update_sq = rho * prev_param_update_sq + (1 - rho) * (param_update ** 2)
    return grad_sq, param_update, param_update_sq

def adadelta_fit(
    X: np.ndarray,
    y: np.ndarray,
    loss_func: Callable[[np.ndarray, np.ndarray], float],
    grad_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    params_init: Optional[np.ndarray] = None,
    rho: float = 0.95,
    epsilon: float = 1e-8,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Perform Adadelta optimization.

    Parameters
    ----------
    X : np.ndarray
        Input features (n_samples, n_features)
    y : np.ndarray
        Target values (n_samples,)
    loss_func : callable
        Loss function to minimize
    grad_func : callable
        Gradient of the loss function
    params_init : np.ndarray, optional
        Initial parameters (n_features,)
    rho : float, optional
        Decay rate for moving average (default: 0.95)
    epsilon : float, optional
        Small constant for numerical stability (default: 1e-8)
    max_iter : int, optional
        Maximum number of iterations (default: 1000)
    tol : float, optional
        Tolerance for stopping criterion (default: 1e-6)

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': optimized parameters
        - 'metrics': dictionary of metrics
        - 'params_used': dictionary of parameters used
        - 'warnings': list of warnings (if any)
    """
    # Validate inputs
    validate_inputs(X, y, locals(), loss_func, grad_func)

    # Initialize parameters
    if params_init is None:
        params = np.zeros(X.shape[1])
    else:
        params = params_init.copy()

    # Initialize Adadelta variables
    prev_grad_sq = np.zeros(X.shape[1])
    prev_param_update_sq = np.zeros(X.shape[1])

    # Optimization loop
    for _ in range(max_iter):
        grad = grad_func(X, y)
        prev_grad_sq, param_update, prev_param_update_sq = update_parameters(
            grad, prev_grad_sq, prev_param_update_sq, rho, epsilon
        )
        params -= param_update

    # Calculate final loss and metrics
    final_loss = loss_func(X @ params, y)
    r2_score = 1 - (final_loss / np.var(y))

    return {
        'result': params,
        'metrics': {
            'loss': final_loss,
            'r2_score': r2_score
        },
        'params_used': {
            'rho': rho,
            'epsilon': epsilon,
            'max_iter': max_iter,
            'tol': tol
        },
        'warnings': []
    }

# Example usage:
"""
X = np.random.rand(100, 5)
y = X @ np.array([1.0, -2.0, 3.0, -4.0, 5.0]) + np.random.normal(0, 0.1, 100)

def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return np.mean((y_pred - y_true) ** 2)

def mse_grad(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return X.T @ (X @ params - y) / len(y)

result = adadelta_fit(X, y, mse_loss, mse_grad)
"""

################################################################################
# momentum
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    momentum_factor: float = 0.9,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> None:
    """Validate input parameters for momentum optimization."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if learning_rate <= 0 or learning_rate >= 1:
        raise ValueError("learning_rate must be in (0, 1)")
    if momentum_factor < 0 or momentum_factor >= 1:
        raise ValueError("momentum_factor must be in [0, 1)")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol < 0:
        raise ValueError("tol must be non-negative")

def compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    metric: str = "mse"
) -> np.ndarray:
    """Compute gradient based on specified metric."""
    n_samples = X.shape[0]
    if metric == "mse":
        residuals = y - X @ weights
        gradient = -(2/n_samples) * X.T @ residuals
    elif metric == "mae":
        residuals = y - X @ weights
        gradient = -(1/n_samples) * np.sum(np.sign(residuals)[:, np.newaxis] * X, axis=0)
    elif callable(metric):
        gradient = metric(X, y, weights)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    return gradient

def normalize_data(
    X: np.ndarray,
    y: np.ndarray,
    normalization: str = "standard"
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize data based on specified method."""
    if normalization == "none":
        return X, y
    elif normalization == "standard":
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1  # Avoid division by zero
        X_normalized = (X - X_mean) / X_std
        y_mean = np.mean(y)
        y_std = np.std(y)
        if y_std == 0:
            y_std = 1
        y_normalized = (y - y_mean) / y_std
    elif normalization == "minmax":
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1
        X_normalized = (X - X_min) / X_range
        y_min = np.min(y)
        y_max = np.max(y)
        if y_max == y_min:
            y_normalized = np.zeros_like(y)
        else:
            y_normalized = (y - y_min) / (y_max - y_min)
    elif normalization == "robust":
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_iqr[X_iqr == 0] = 1
        X_normalized = (X - X_median) / X_iqr
        y_median = np.median(y)
        y_iqr = np.subtract(*np.percentile(y, [75, 25]))
        if y_iqr == 0:
            y_iqr = 1
        y_normalized = (y - y_median) / y_iqr
    else:
        raise ValueError(f"Unsupported normalization: {normalization}")
    return X_normalized, y_normalized

def momentum_fit(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    momentum_factor: float = 0.9,
    max_iter: int = 1000,
    tol: float = 1e-4,
    metric: Union[str, Callable] = "mse",
    normalization: str = "standard",
    initial_weights: Optional[np.ndarray] = None,
    verbose: bool = False
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Perform momentum optimization to find optimal weights.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    learning_rate : float, optional
        Learning rate for gradient descent, by default 0.01
    momentum_factor : float, optional
        Momentum factor, by default 0.9
    max_iter : int, optional
        Maximum number of iterations, by default 1000
    tol : float, optional
        Tolerance for stopping criterion, by default 1e-4
    metric : str or callable, optional
        Metric to optimize ("mse", "mae"), by default "mse"
    normalization : str, optional
        Normalization method ("none", "standard", "minmax", "robust"), by default "standard"
    initial_weights : np.ndarray, optional
        Initial weights, by default None (random initialization)
    verbose : bool, optional
        Whether to print progress, by default False

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - "result": optimized weights
        - "metrics": dictionary of computed metrics
        - "params_used": parameters used in the optimization
        - "warnings": any warnings encountered

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = momentum_fit(X, y, learning_rate=0.1, max_iter=500)
    """
    # Validate inputs
    validate_inputs(X, y, learning_rate, momentum_factor, max_iter, tol)

    # Normalize data
    X_norm, y_norm = normalize_data(X, y, normalization)

    # Initialize weights
    n_features = X.shape[1]
    if initial_weights is None:
        weights = np.random.randn(n_features)
    else:
        if len(initial_weights) != n_features:
            raise ValueError("initial_weights must have the same length as number of features")
        weights = initial_weights.copy()

    # Initialize velocity
    velocity = np.zeros_like(weights)

    # Store metrics
    metrics = {
        "final_loss": float('inf'),
        "iterations": 0,
        "convergence": False
    }

    # Optimization loop
    for i in range(max_iter):
        # Compute gradient
        grad = compute_gradient(X_norm, y_norm, weights, metric)

        # Update velocity and weights
        velocity = momentum_factor * velocity + learning_rate * grad
        new_weights = weights - velocity

        # Check for convergence
        if np.linalg.norm(new_weights - weights) < tol:
            metrics["convergence"] = True
            break

        # Update weights for next iteration
        weights = new_weights

    metrics["final_loss"] = np.mean((y_norm - X_norm @ weights) ** 2)
    metrics["iterations"] = i + 1

    return {
        "result": weights,
        "metrics": metrics,
        "params_used": {
            "learning_rate": learning_rate,
            "momentum_factor": momentum_factor,
            "max_iter": max_iter,
            "tol": tol,
            "metric": metric,
            "normalization": normalization
        },
        "warnings": []
    }

################################################################################
# nesterov_momentum
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def nesterov_momentum_fit(
    objective_func: Callable[[np.ndarray], float],
    gradient_func: Callable[[np.ndarray], np.ndarray],
    initial_params: np.ndarray,
    learning_rate: float = 0.01,
    momentum_factor: float = 0.9,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    normalization: str = 'none',
    metric_func: Callable[[np.ndarray, np.ndarray], float] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Optimize a function using Nesterov's accelerated gradient (momentum) method.

    Parameters:
    - objective_func: Callable[[np.ndarray], float]
        The objective function to minimize.
    - gradient_func: Callable[[np.ndarray], np.ndarray]
        The gradient of the objective function.
    - initial_params: np.ndarray
        Initial parameters for optimization.
    - learning_rate: float, optional (default=0.01)
        Learning rate for the optimizer.
    - momentum_factor: float, optional (default=0.9)
        Momentum factor for Nesterov's method.
    - max_iterations: int, optional (default=1000)
        Maximum number of iterations.
    - tolerance: float, optional (default=1e-6)
        Tolerance for convergence.
    - normalization: str, optional (default='none')
        Normalization method ('none', 'standard', 'minmax', 'robust').
    - metric_func: Callable[[np.ndarray, np.ndarray], float], optional
        Custom metric function to evaluate the optimization result.

    Returns:
    - Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]
        A dictionary containing:
        - 'result': Optimized parameters.
        - 'metrics': Dictionary of metrics (if applicable).
        - 'params_used': Parameters used during optimization.
        - 'warnings': Dictionary of warnings (if any).
    """
    # Validate inputs
    if not callable(objective_func):
        raise ValueError("objective_func must be a callable.")
    if not callable(gradient_func):
        raise ValueError("gradient_func must be a callable.")
    if not isinstance(initial_params, np.ndarray):
        raise ValueError("initial_params must be a numpy array.")
    if normalization not in ['none', 'standard', 'minmax', 'robust']:
        raise ValueError("normalization must be one of: 'none', 'standard', 'minmax', 'robust'.")

    # Initialize parameters
    params = initial_params.copy()
    velocity = np.zeros_like(params)
    prev_params = np.zeros_like(params)

    # Normalize initial parameters if required
    if normalization == 'standard':
        mean = np.mean(params)
        std = np.std(params)
        if std != 0:
            params = (params - mean) / std
    elif normalization == 'minmax':
        min_val = np.min(params)
        max_val = np.max(params)
        if max_val != min_val:
            params = (params - min_val) / (max_val - min_val)
    elif normalization == 'robust':
        median = np.median(params)
        iqr = np.percentile(params, 75) - np.percentile(params, 25)
        if iqr != 0:
            params = (params - median) / iqr

    # Optimization loop
    for iteration in range(max_iterations):
        # Nesterov's momentum update
        velocity = momentum_factor * velocity + learning_rate * gradient_func(params - momentum_factor * velocity)
        prev_params = params.copy()
        params = params - velocity

        # Check convergence
        if np.linalg.norm(params - prev_params) < tolerance:
            break

    # Evaluate metrics if provided
    metrics = {}
    if metric_func is not None:
        metrics['custom_metric'] = metric_func(params, prev_params)

    # Prepare output
    result_dict = {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'learning_rate': learning_rate,
            'momentum_factor': momentum_factor,
            'max_iterations': iteration + 1,
            'tolerance': tolerance,
            'normalization': normalization
        },
        'warnings': {}
    }

    return result_dict

def _validate_inputs(
    objective_func: Callable[[np.ndarray], float],
    gradient_func: Callable[[np.ndarray], np.ndarray],
    initial_params: np.ndarray
) -> None:
    """
    Validate the inputs for Nesterov's momentum optimization.

    Parameters:
    - objective_func: Callable[[np.ndarray], float]
        The objective function to minimize.
    - gradient_func: Callable[[np.ndarray], np.ndarray]
        The gradient of the objective function.
    - initial_params: np.ndarray
        Initial parameters for optimization.

    Raises:
    - ValueError: If any input is invalid.
    """
    if not callable(objective_func):
        raise ValueError("objective_func must be a callable.")
    if not callable(gradient_func):
        raise ValueError("gradient_func must be a callable.")
    if not isinstance(initial_params, np.ndarray):
        raise ValueError("initial_params must be a numpy array.")

def _apply_normalization(
    params: np.ndarray,
    normalization: str
) -> np.ndarray:
    """
    Apply normalization to the parameters.

    Parameters:
    - params: np.ndarray
        Parameters to normalize.
    - normalization: str
        Normalization method ('none', 'standard', 'minmax', 'robust').

    Returns:
    - np.ndarray: Normalized parameters.
    """
    if normalization == 'standard':
        mean = np.mean(params)
        std = np.std(params)
        if std != 0:
            params = (params - mean) / std
    elif normalization == 'minmax':
        min_val = np.min(params)
        max_val = np.max(params)
        if max_val != min_val:
            params = (params - min_val) / (max_val - min_val)
    elif normalization == 'robust':
        median = np.median(params)
        iqr = np.percentile(params, 75) - np.percentile(params, 25)
        if iqr != 0:
            params = (params - median) / iqr
    return params

def _compute_metrics(
    params: np.ndarray,
    prev_params: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float] = None
) -> Dict[str, float]:
    """
    Compute metrics for the optimization result.

    Parameters:
    - params: np.ndarray
        Current parameters.
    - prev_params: np.ndarray
        Previous parameters.
    - metric_func: Callable[[np.ndarray, np.ndarray], float], optional
        Custom metric function to evaluate the optimization result.

    Returns:
    - Dict[str, float]: Dictionary of metrics.
    """
    metrics = {}
    if metric_func is not None:
        metrics['custom_metric'] = metric_func(params, prev_params)
    return metrics
