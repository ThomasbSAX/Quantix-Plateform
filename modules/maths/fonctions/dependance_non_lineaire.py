"""
Quantix – Module dependance_non_lineaire
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# regression_logistique
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def regression_logistique_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalisation: str = 'standard',
    metrique: Union[str, Callable] = 'logloss',
    solveur: str = 'gradient_descent',
    reg_type: Optional[str] = None,
    reg_param: float = 0.0,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    custom_metric: Optional[Callable] = None,
    verbose: bool = False
) -> Dict:
    """
    Fit a logistic regression model with various customizable options.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,) or (n_samples, 1).
    normalisation : str
        Type of normalization: 'none', 'standard', 'minmax', or 'robust'.
    metrique : str or Callable
        Metric to evaluate the model: 'mse', 'mae', 'r2', 'logloss', or custom callable.
    solveur : str
        Solver to use: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'.
    reg_type : str, optional
        Type of regularization: 'none', 'l1', 'l2', or 'elasticnet'.
    reg_param : float
        Regularization parameter.
    tol : float
        Tolerance for stopping criteria.
    max_iter : int
        Maximum number of iterations.
    learning_rate : float
        Learning rate for gradient descent.
    custom_metric : Callable, optional
        Custom metric function.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    Dict
        Dictionary containing the results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, 100)
    >>> result = regression_logistique_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _apply_normalisation(X, normalisation)

    # Initialize parameters
    n_features = X_normalized.shape[1]
    coefficients = np.zeros(n_features)
    intercept = 0.0

    # Choose solver
    if solveur == 'closed_form':
        coefficients, intercept = _solve_closed_form(X_normalized, y)
    elif solveur == 'gradient_descent':
        coefficients, intercept = _solve_gradient_descent(
            X_normalized, y, reg_type, reg_param, tol, max_iter, learning_rate, verbose
        )
    elif solveur == 'newton':
        coefficients, intercept = _solve_newton(
            X_normalized, y, reg_type, reg_param, tol, max_iter, verbose
        )
    elif solveur == 'coordinate_descent':
        coefficients, intercept = _solve_coordinate_descent(
            X_normalized, y, reg_type, reg_param, tol, max_iter, verbose
        )
    else:
        raise ValueError("Invalid solver specified.")

    # Calculate metrics
    metrics = _calculate_metrics(X_normalized, y, coefficients, intercept, metrique, custom_metric)

    # Prepare output
    result = {
        'result': {
            'coefficients': coefficients,
            'intercept': intercept
        },
        'metrics': metrics,
        'params_used': {
            'normalisation': normalisation,
            'metrique': metrique,
            'solveur': solveur,
            'reg_type': reg_type,
            'reg_param': reg_param,
            'tol': tol,
            'max_iter': max_iter,
            'learning_rate': learning_rate
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1 and y.ndim != 2:
        raise ValueError("y must be a 1D or 2D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")

def _apply_normalisation(X: np.ndarray, normalisation: str) -> np.ndarray:
    """Apply specified normalization to the input data."""
    if normalisation == 'none':
        return X
    elif normalisation == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif normalisation == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif normalisation == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError("Invalid normalisation specified.")

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> tuple:
    """Solve logistic regression using closed-form solution."""
    # This is a placeholder; actual implementation would use iterative methods
    return np.zeros(X.shape[1]), 0.0

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    reg_type: Optional[str],
    reg_param: float,
    tol: float,
    max_iter: int,
    learning_rate: float,
    verbose: bool
) -> tuple:
    """Solve logistic regression using gradient descent."""
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)
    intercept = 0.0

    for _ in range(max_iter):
        # Compute predictions
        linear_model = np.dot(X, coefficients) + intercept
        predictions = 1 / (1 + np.exp(-linear_model))

        # Compute gradients
        gradient = np.dot(X.T, (predictions - y)) / n_samples

        # Add regularization
        if reg_type == 'l1':
            gradient += reg_param * np.sign(coefficients)
        elif reg_type == 'l2':
            gradient += 2 * reg_param * coefficients
        elif reg_type == 'elasticnet':
            gradient += reg_param * (np.sign(coefficients) + 2 * coefficients)

        # Update parameters
        coefficients -= learning_rate * gradient
        intercept -= learning_rate * np.mean(predictions - y)

        # Check convergence
        if np.linalg.norm(gradient) < tol:
            break

    return coefficients, intercept

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    reg_type: Optional[str],
    reg_param: float,
    tol: float,
    max_iter: int,
    verbose: bool
) -> tuple:
    """Solve logistic regression using Newton's method."""
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)
    intercept = 0.0

    for _ in range(max_iter):
        # Compute predictions
        linear_model = np.dot(X, coefficients) + intercept
        predictions = 1 / (1 + np.exp(-linear_model))

        # Compute gradients and Hessian
        gradient = np.dot(X.T, (predictions - y)) / n_samples
        hessian_diag = np.diag(np.dot(X.T, predictions * (1 - predictions)) / n_samples)

        # Add regularization
        if reg_type == 'l2':
            hessian_diag += 2 * reg_param * np.eye(n_features)

        # Update parameters
        coefficients -= np.linalg.solve(hessian_diag, gradient)
        intercept -= np.mean(predictions - y)

        # Check convergence
        if np.linalg.norm(gradient) < tol:
            break

    return coefficients, intercept

def _solve_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    reg_type: Optional[str],
    reg_param: float,
    tol: float,
    max_iter: int,
    verbose: bool
) -> tuple:
    """Solve logistic regression using coordinate descent."""
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)
    intercept = 0.0

    for _ in range(max_iter):
        for j in range(n_features):
            # Compute predictions
            linear_model = np.dot(X, coefficients) + intercept
            predictions = 1 / (1 + np.exp(-linear_model))

            # Compute gradient for feature j
            gradient_j = np.dot(X[:, j], (predictions - y)) / n_samples

            # Add regularization
            if reg_type == 'l1':
                gradient_j += reg_param * np.sign(coefficients[j])
            elif reg_type == 'l2':
                gradient_j += 2 * reg_param * coefficients[j]
            elif reg_type == 'elasticnet':
                gradient_j += reg_param * (np.sign(coefficients[j]) + 2 * coefficients[j])

            # Update coefficient j
            coefficients[j] -= gradient_j

        # Check convergence
        if np.linalg.norm(gradient_j) < tol:
            break

    return coefficients, intercept

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    metrique: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Calculate metrics for the logistic regression model."""
    linear_model = np.dot(X, coefficients) + intercept
    predictions = 1 / (1 + np.exp(-linear_model))

    metrics = {}

    if metrique == 'mse':
        metrics['mse'] = np.mean((predictions - y) ** 2)
    elif metrique == 'mae':
        metrics['mae'] = np.mean(np.abs(predictions - y))
    elif metrique == 'r2':
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)
    elif metrique == 'logloss':
        metrics['logloss'] = -np.mean(y * np.log(predictions + 1e-8) + (1 - y) * np.log(1 - predictions + 1e-8))
    elif callable(metrique):
        metrics['custom'] = metrique(y, predictions)
    if custom_metric is not None:
        metrics['custom'] = custom_metric(y, predictions)

    return metrics

################################################################################
# arbre_decision
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def arbre_decision_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_depth: int = 3,
    min_samples_split: int = 2,
    metric: Union[str, Callable] = "mse",
    distance: Union[str, Callable] = "euclidean",
    solver: str = "closed_form",
    normalizer: Optional[Callable] = None,
    regularization: str = "none",
    tol: float = 1e-4,
    max_iter: int = 100,
    random_state: Optional[int] = None
) -> Dict:
    """
    Fit a decision tree model to capture non-linear dependencies.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    max_depth : int, optional
        Maximum depth of the tree.
    min_samples_split : int, optional
        Minimum number of samples required to split a node.
    metric : str or callable, optional
        Metric to evaluate splits. Options: "mse", "mae", "gini", or custom callable.
    distance : str or callable, optional
        Distance metric for feature selection. Options: "euclidean", "manhattan", or custom.
    solver : str, optional
        Solver for optimization. Options: "closed_form", "gradient_descent".
    normalizer : callable, optional
        Normalization function for features.
    regularization : str, optional
        Regularization type. Options: "none", "l1", "l2".
    tol : float, optional
        Tolerance for stopping criteria.
    max_iter : int, optional
        Maximum number of iterations.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict
        Dictionary containing:
        - "result": Fitted tree structure.
        - "metrics": Evaluation metrics.
        - "params_used": Parameters used during fitting.
        - "warnings": Any warnings encountered.

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = arbre_decision_fit(X, y, max_depth=3, metric="mse")
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Initialize tree structure
    tree = {"depth": 0, "samples": len(X), "feature": None, "threshold": None, "left": None, "right": None}

    # Build tree recursively
    _build_tree(
        X_normalized, y,
        tree=tree,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        metric=metric,
        distance=distance,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state
    )

    # Calculate metrics
    metrics = _calculate_metrics(X_normalized, y, tree, metric)

    return {
        "result": tree,
        "metrics": metrics,
        "params_used": {
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "normalizer": normalizer.__name__ if normalizer else None,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to features."""
    if normalizer is None:
        return X
    return normalizer(X)

def _build_tree(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tree: Dict,
    max_depth: int,
    min_samples_split: int,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    solver: str,
    regularization: str,
    tol: float,
    max_iter: int,
    random_state: Optional[int]
) -> None:
    """Recursively build the decision tree."""
    if tree["depth"] >= max_depth or len(y) < min_samples_split:
        return

    # Find best split
    best_feature, best_threshold = _find_best_split(
        X, y,
        metric=metric,
        distance=distance
    )

    if best_feature is None:
        return

    # Split data
    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask

    # Create child nodes
    tree["feature"] = best_feature
    tree["threshold"] = best_threshold

    left_child = {"depth": tree["depth"] + 1, "samples": np.sum(left_mask)}
    right_child = {"depth": tree["depth"] + 1, "samples": np.sum(right_mask)}

    tree["left"] = left_child
    tree["right"] = right_child

    # Recursively build children
    _build_tree(
        X[left_mask], y[left_mask],
        tree=left_child,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        metric=metric,
        distance=distance,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state
    )

    _build_tree(
        X[right_mask], y[right_mask],
        tree=right_child,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        metric=metric,
        distance=distance,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state
    )

def _find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    metric: Union[str, Callable],
    distance: Union[str, Callable]
) -> tuple:
    """Find the best feature and threshold to split on."""
    best_feature = None
    best_threshold = None
    best_score = -np.inf

    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask

            if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
                continue

            score = _evaluate_split(y[left_mask], y[right_mask], metric)
            if score > best_score:
                best_score = score
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold

def _evaluate_split(
    y_left: np.ndarray,
    y_right: np.ndarray,
    metric: Union[str, Callable]
) -> float:
    """Evaluate the quality of a split using the specified metric."""
    if callable(metric):
        return metric(y_left, y_right)

    if metric == "mse":
        return -np.mean((y_left - np.mean(y_left))**2 + (y_right - np.mean(y_right))**2)
    elif metric == "mae":
        return -(np.mean(np.abs(y_left - np.median(y_left))) + np.mean(np.abs(y_right - np.median(y_right))))
    elif metric == "gini":
        return _calculate_gini_impurity(y_left, y_right)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _calculate_gini_impurity(
    y_left: np.ndarray,
    y_right: np.ndarray
) -> float:
    """Calculate Gini impurity for a split."""
    def gini(y):
        if len(y) == 0:
            return 0
        p = np.bincount(y.astype(int)) / len(y)
        return 1 - np.sum(p**2)

    n_left = len(y_left)
    n_right = len(y_right)
    total = n_left + n_right

    return (n_left / total) * gini(y_left) + (n_right / total) * gini(y_right)

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    tree: Dict,
    metric: Union[str, Callable]
) -> Dict:
    """Calculate evaluation metrics for the fitted tree."""
    y_pred = _predict(tree, X)

    if callable(metric):
        return {"custom_metric": metric(y, y_pred)}

    metrics = {}
    if metric == "mse":
        metrics["mse"] = np.mean((y - y_pred)**2)
    elif metric == "mae":
        metrics["mae"] = np.mean(np.abs(y - y_pred))
    elif metric == "r2":
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        metrics["r2"] = 1 - (ss_res / ss_tot)

    return metrics

def _predict(tree: Dict, X: np.ndarray) -> np.ndarray:
    """Predict using the fitted tree."""
    y_pred = np.zeros(len(X))

    for i in range(len(X)):
        node = tree
        while node["left"] is not None:
            if X[i, node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]

        y_pred[i] = np.mean(y)  # Default to mean for leaf nodes

    return y_pred

################################################################################
# foret_aleatoire
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def foret_aleatoire_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    metric: Union[str, Callable] = 'mse',
    normalization: str = 'none',
    random_state: Optional[int] = None,
    n_jobs: int = -1
) -> Dict[str, Any]:
    """
    Fit a random forest model to estimate non-linear dependencies.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    n_estimators : int, optional
        Number of trees in the forest (default: 100).
    max_depth : int, optional
        Maximum depth of the trees (default: None).
    min_samples_split : int, optional
        Minimum number of samples required to split a node (default: 2).
    min_samples_leaf : int, optional
        Minimum number of samples required at each leaf node (default: 1).
    metric : str or callable, optional
        Metric to evaluate the quality of splits ('mse', 'mae', etc.) or custom callable (default: 'mse').
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'none').
    random_state : int, optional
        Random seed for reproducibility (default: None).
    n_jobs : int, optional
        Number of jobs to run in parallel (-1 means using all processors) (default: -1).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the fitted model and related information.

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = foret_aleatoire_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_normalized = _apply_normalization(X, normalization)

    # Initialize random forest
    forest = _initialize_forest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        metric=metric,
        random_state=random_state
    )

    # Fit the forest in parallel if n_jobs > 1
    if n_jobs > 1:
        forest = _parallel_fit(forest, X_normalized, y, n_jobs=n_jobs)
    else:
        forest = _fit_forest(forest, X_normalized, y)

    # Calculate metrics
    metrics = _calculate_metrics(forest, X_normalized, y)

    # Prepare output
    result = {
        "result": forest,
        "metrics": metrics,
        "params_used": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "metric": metric,
            "normalization": normalization
        },
        "warnings": _check_warnings(forest)
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be a 2D array and y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values.")

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to the input features."""
    if method == 'standard':
        X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        X_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        X_normalized = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        X_normalized = X.copy()
    return X_normalized

def _initialize_forest(
    n_estimators: int,
    max_depth: Optional[int],
    min_samples_split: int,
    min_samples_leaf: int,
    metric: Union[str, Callable],
    random_state: Optional[int]
) -> Dict[str, Any]:
    """Initialize the random forest structure."""
    if isinstance(metric, str):
        metric_func = _get_metric_function(metric)
    else:
        metric_func = metric

    forest = {
        'trees': [],
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'metric_func': metric_func,
        'random_state': random_state
    }
    return forest

def _get_metric_function(metric: str) -> Callable:
    """Get the metric function based on the input string."""
    metrics = {
        'mse': _mse,
        'mae': _mae
    }
    if metric not in metrics:
        raise ValueError(f"Unsupported metric: {metric}")
    return metrics[metric]

def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def _fit_forest(forest: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Fit a single tree in the forest."""
    for _ in range(forest['n_estimators']):
        tree = _fit_single_tree(X, y, forest)
        forest['trees'].append(tree)
    return forest

def _fit_single_tree(X: np.ndarray, y: np.ndarray, forest_params: Dict[str, Any]) -> Dict[str, Any]:
    """Fit a single decision tree."""
    tree = {
        'structure': _build_tree(X, y, forest_params),
        'feature_importances': np.zeros(X.shape[1])
    }
    return tree

def _build_tree(X: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively build a decision tree."""
    node = _create_node(X, y, params)
    if _is_leaf(node, params):
        return node
    else:
        left_X, right_X, left_y, right_y = _split_data(X, y, node['split'])
        node['left'] = _build_tree(left_X, left_y, params)
        node['right'] = _build_tree(right_X, right_y, params)
    return node

def _create_node(X: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a node in the decision tree."""
    best_split = _find_best_split(X, y, params['metric_func'])
    return {
        'split': best_split,
        'value': np.mean(y) if best_split is None else None
    }

def _find_best_split(X: np.ndarray, y: np.ndarray, metric_func: Callable) -> Optional[Dict[str, Any]]:
    """Find the best split for a node."""
    best_split = None
    best_score = float('inf')
    n_features = X.shape[1]

    for feature_idx in np.random.choice(n_features, size=int(np.sqrt(n_features)), replace=False):
        for threshold in np.unique(X[:, feature_idx]):
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask

            if np.sum(left_mask) < params['min_samples_split'] or np.sum(right_mask) < params['min_samples_split']:
                continue

            left_y = y[left_mask]
            right_y = y[right_mask]

            score = metric_func(y, np.concatenate([np.full_like(left_y, np.mean(left_y)), np.full_like(right_y, np.mean(right_y))]))

            if score < best_score:
                best_score = score
                best_split = {
                    'feature_idx': feature_idx,
                    'threshold': threshold
                }

    return best_split

def _is_leaf(node: Dict[str, Any], params: Dict[str, Any]) -> bool:
    """Check if a node is a leaf."""
    return (node['split'] is None or
            len(node.get('left', [])) < params['min_samples_leaf'] or
            len(node.get('right', [])) < params['min_samples_leaf'])

def _split_data(X: np.ndarray, y: np.ndarray, split: Dict[str, Any]) -> tuple:
    """Split data based on a split rule."""
    left_mask = X[:, split['feature_idx']] <= split['threshold']
    right_mask = ~left_mask
    return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

def _parallel_fit(forest: Dict[str, Any], X: np.ndarray, y: np.ndarray, n_jobs: int) -> Dict[str, Any]:
    """Fit the forest in parallel."""
    from multiprocessing import Pool
    with Pool(n_jobs) as pool:
        forest['trees'] = pool.starmap(
            _fit_single_tree,
            [(X, y, forest) for _ in range(forest['n_estimators'])]
        )
    return forest

def _calculate_metrics(forest: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Calculate metrics for the fitted forest."""
    y_pred = _predict(forest, X)
    return {
        'mse': _mse(y, y_pred),
        'mae': _mae(y, y_pred)
    }

def _predict(forest: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Predict using the fitted forest."""
    predictions = np.array([_predict_single_tree(tree, X) for tree in forest['trees']])
    return np.mean(predictions, axis=0)

def _predict_single_tree(tree: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Predict using a single tree."""
    return _traverse_tree(tree['structure'], X)

def _traverse_tree(node: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Traverse the tree to make predictions."""
    if 'left' in node and 'right' in node:
        left_mask = X[:, node['split']['feature_idx']] <= node['split']['threshold']
        right_mask = ~left_mask
        return np.where(
            left_mask,
            _traverse_tree(node['left'], X[left_mask]),
            _traverse_tree(node['right'], X[right_mask])
        )
    else:
        return np.full(X.shape[0], node['value'])

def _check_warnings(forest: Dict[str, Any]) -> list:
    """Check for warnings in the fitted forest."""
    warnings = []
    if len(forest['trees']) == 0:
        warnings.append("No trees were fitted.")
    return warnings

################################################################################
# gradient_boosting
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def gradient_boosting_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    loss: str = 'mse',
    metric: Union[str, Callable] = 'mse',
    normalizer: Optional[Callable] = None,
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_depth: int = 3,
    min_samples_split: int = 2,
    random_state: Optional[int] = None
) -> Dict:
    """
    Fit a gradient boosting model to the data.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    n_estimators : int, default=100
        Number of boosting stages to be run.
    learning_rate : float, default=0.1
        Shrinkage factor for the contributions of each tree.
    loss : str, default='mse'
        Loss function to be optimized. Options: 'mse', 'mae', 'huber'.
    metric : Union[str, Callable], default='mse'
        Metric to evaluate the quality of predictions. Options: 'mse', 'mae', 'r2', custom callable.
    normalizer : Optional[Callable], default=None
        Normalization function to apply to the data. Options: None, standard, minmax, robust.
    solver : str, default='gradient_descent'
        Solver to use for optimization. Options: 'gradient_descent', 'newton'.
    regularization : Optional[str], default=None
        Regularization type. Options: None, 'l1', 'l2', 'elasticnet'.
    tol : float, default=1e-4
        Tolerance for the early stopping criterion.
    max_depth : int, default=3
        Maximum depth of the individual regression estimators.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    random_state : Optional[int], default=None
        Controls the randomness of the estimator.

    Returns:
    --------
    Dict
        A dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Initialize random state
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize data
    if normalizer is not None:
        X, y = _apply_normalization(X, y, normalizer)

    # Initialize model
    model = {
        'trees': [],
        'learning_rate': learning_rate,
        'loss': loss,
        'metric': metric,
        'regularization': regularization
    }

    # Initialize predictions with the mean of y
    F = np.full(y.shape, np.mean(y))

    for _ in range(n_estimators):
        # Compute residuals
        residuals = _compute_residuals(y, F, loss)

        # Fit a tree to the residuals
        tree = _fit_tree(X, residuals, max_depth, min_samples_split)

        # Update predictions
        F += learning_rate * tree.predict(X)

        # Check for early stopping
        if _check_early_stopping(F, y, metric, tol):
            break

    # Compute final metrics
    metrics = _compute_metrics(y, F, metric)

    return {
        'result': {'predictions': F},
        'metrics': metrics,
        'params_used': {
            'n_estimators': len(model['trees']),
            'learning_rate': learning_rate,
            'loss': loss,
            'metric': metric,
            'regularization': regularization
        },
        'warnings': []
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

def _apply_normalization(X: np.ndarray, y: np.ndarray, normalizer: Callable) -> tuple:
    """Apply normalization to the data."""
    X_normalized = normalizer(X)
    y_normalized = normalizer(y.reshape(-1, 1)).flatten()
    return X_normalized, y_normalized

def _compute_residuals(y: np.ndarray, F: np.ndarray, loss: str) -> np.ndarray:
    """Compute the residuals based on the specified loss function."""
    if loss == 'mse':
        return y - F
    elif loss == 'mae':
        return np.sign(y - F)
    elif loss == 'huber':
        delta = 1.0
        residuals = y - F
        return np.where(np.abs(residuals) <= delta, residuals, delta * np.sign(residuals))
    else:
        raise ValueError(f"Unsupported loss function: {loss}")

def _fit_tree(X: np.ndarray, y: np.ndarray, max_depth: int, min_samples_split: int) -> object:
    """Fit a decision tree to the data."""
    # Placeholder for actual tree fitting logic
    class Tree:
        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.zeros(X.shape[0])
    return Tree()

def _check_early_stopping(F: np.ndarray, y: np.ndarray, metric: Union[str, Callable], tol: float) -> bool:
    """Check if early stopping criteria is met."""
    current_metric = _compute_metrics(y, F, metric)
    return abs(current_metric['value']) < tol

def _compute_metrics(y: np.ndarray, F: np.ndarray, metric: Union[str, Callable]) -> Dict:
    """Compute the specified metrics."""
    if callable(metric):
        return {'value': metric(y, F)}
    elif metric == 'mse':
        return {'value': np.mean((y - F) ** 2)}
    elif metric == 'mae':
        return {'value': np.mean(np.abs(y - F))}
    elif metric == 'r2':
        ss_res = np.sum((y - F) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return {'value': 1 - (ss_res / ss_tot)}
    else:
        raise ValueError(f"Unsupported metric: {metric}")

################################################################################
# reseau_de_neurones
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def _normalize_data(X: np.ndarray, y: np.ndarray, normalization: str) -> tuple:
    """Normalize input data."""
    if normalization == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_normalized = (X - X_mean) / X_std
    elif normalization == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_normalized = (X - X_min) / (X_max - X_min)
    elif normalization == 'robust':
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_normalized = (X - X_median) / X_iqr
    else:
        X_normalized = X.copy()

    return X_normalized, y

def _initialize_weights(n_features: int, n_hidden: int) -> tuple:
    """Initialize weights for the neural network."""
    W1 = np.random.randn(n_features, n_hidden) * 0.01
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.randn(n_hidden, 1) * 0.01
    b2 = np.zeros((1, 1))
    return W1, b1, W2, b2

def _forward_propagation(X: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                         W2: np.ndarray, b2: np.ndarray) -> tuple:
    """Forward propagation through the neural network."""
    Z1 = np.dot(X, W1) + b1
    A1 = 1 / (1 + np.exp(-Z1))
    Z2 = np.dot(A1, W2) + b2
    A2 = Z2  # Linear activation for output layer
    return A1, A2

def _compute_loss(y: np.ndarray, y_pred: np.ndarray,
                  loss_function: str) -> float:
    """Compute the loss based on the specified function."""
    if loss_function == 'mse':
        return np.mean((y - y_pred) ** 2)
    elif loss_function == 'mae':
        return np.mean(np.abs(y - y_pred))
    else:
        raise ValueError("Unsupported loss function")

def _backward_propagation(X: np.ndarray, y: np.ndarray, A1: np.ndarray,
                          A2: np.ndarray, W2: np.ndarray) -> tuple:
    """Backward propagation to compute gradients."""
    m = X.shape[0]
    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * A1 * (1 - A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    return dW1, db1, dW2, db2

def _update_parameters(W1: np.ndarray, b1: np.ndarray,
                       W2: np.ndarray, b2: np.ndarray,
                       dW1: np.ndarray, db1: np.ndarray,
                       dW2: np.ndarray, db2: np.ndarray,
                       learning_rate: float) -> tuple:
    """Update the parameters using gradient descent."""
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

def _compute_metrics(y: np.ndarray, y_pred: np.ndarray,
                     metrics: list) -> Dict[str, float]:
    """Compute the specified metrics."""
    results = {}
    if 'mse' in metrics:
        results['mse'] = np.mean((y - y_pred) ** 2)
    if 'mae' in metrics:
        results['mae'] = np.mean(np.abs(y - y_pred))
    if 'r2' in metrics:
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        results['r2'] = 1 - (ss_res / ss_tot)
    return results

def reseau_de_neurones_fit(X: np.ndarray, y: np.ndarray,
                           n_hidden: int = 4,
                           normalization: str = 'standard',
                           loss_function: str = 'mse',
                           learning_rate: float = 0.01,
                           n_iterations: int = 1000,
                           metrics: list = ['mse'],
                           tol: float = 1e-4) -> Dict:
    """
    Fit a neural network model to the data.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    n_hidden : int, optional
        Number of hidden units, by default 4.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust'), by default 'standard'.
    loss_function : str, optional
        Loss function ('mse', 'mae'), by default 'mse'.
    learning_rate : float, optional
        Learning rate for gradient descent, by default 0.01.
    n_iterations : int, optional
        Number of iterations, by default 1000.
    metrics : list, optional
        List of metrics to compute ('mse', 'mae', 'r2'), by default ['mse'].
    tol : float, optional
        Tolerance for early stopping, by default 1e-4.

    Returns
    -------
    Dict
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    _validate_inputs(X, y)
    X_normalized, y = _normalize_data(X, y, normalization)
    W1, b1, W2, b2 = _initialize_weights(X.shape[1], n_hidden)
    prev_loss = float('inf')

    for i in range(n_iterations):
        A1, A2 = _forward_propagation(X_normalized, W1, b1, W2, b2)
        current_loss = _compute_loss(y, A2, loss_function)

        if abs(prev_loss - current_loss) < tol:
            break

        dW1, db1, dW2, db2 = _backward_propagation(X_normalized, y, A1, A2, W2)
        W1, b1, W2, b2 = _update_parameters(W1, b1, W2, b2,
                                            dW1, db1, dW2, db2,
                                            learning_rate)
        prev_loss = current_loss

    y_pred = _forward_propagation(X_normalized, W1, b1, W2, b2)[1]
    metrics_results = _compute_metrics(y, y_pred, metrics)

    return {
        'result': y_pred,
        'metrics': metrics_results,
        'params_used': {
            'n_hidden': n_hidden,
            'normalization': normalization,
            'loss_function': loss_function,
            'learning_rate': learning_rate,
            'n_iterations': i + 1
        },
        'warnings': []
    }

################################################################################
# svm_non_lineaire
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def svm_non_lineaire_fit(
    X: np.ndarray,
    y: np.ndarray,
    kernel: Union[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = 'rbf',
    C: float = 1.0,
    epsilon: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-3,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'smoothed_gradient_descent',
    **kwargs
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit a non-linear SVM model to the data.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,) or (n_samples, n_outputs).
    kernel : str or callable
        Kernel function to use. Can be 'linear', 'poly', 'rbf', 'sigmoid' or a custom callable.
    C : float
        Regularization parameter. The strength of the regularization is inversely proportional to C.
    epsilon : float
        Epsilon-tube for the epsilon-SVR.
    max_iter : int
        Maximum number of iterations to perform.
    tol : float
        Tolerance for stopping criteria.
    normalizer : callable or None
        Function to normalize the input features. If None, no normalization is applied.
    metric : str or callable
        Metric to evaluate the model. Can be 'mse', 'mae', 'r2' or a custom callable.
    solver : str
        Solver to use. Can be 'smoothed_gradient_descent', 'newton', or 'coordinate_descent'.
    **kwargs
        Additional keyword arguments passed to the solver.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        A dictionary containing the results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = svm_non_lineaire_fit(X, y, kernel='rbf', C=1.0)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if a normalizer is provided
    if normalizer is not None:
        X = normalizer(X)

    # Initialize kernel function
    kernel_func = _get_kernel_function(kernel)

    # Fit the model using the specified solver
    if solver == 'smoothed_gradient_descent':
        params = _svm_sgd(X, y, kernel_func, C, epsilon, max_iter, tol, **kwargs)
    elif solver == 'newton':
        params = _svm_newton(X, y, kernel_func, C, epsilon, max_iter, tol, **kwargs)
    elif solver == 'coordinate_descent':
        params = _svm_cd(X, y, kernel_func, C, epsilon, max_iter, tol, **kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X, y, params['alpha'], kernel_func, metric)

    # Prepare the result dictionary
    result = {
        'result': params['alpha'],
        'metrics': metrics,
        'params_used': {
            'kernel': kernel,
            'C': C,
            'epsilon': epsilon,
            'max_iter': max_iter,
            'tol': tol,
            'solver': solver
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate the input data."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1 and y.ndim != 2:
        raise ValueError("y must be a 1D or 2D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")

def _get_kernel_function(kernel: Union[str, Callable]) -> Callable:
    """Get the kernel function based on the input."""
    if callable(kernel):
        return kernel
    elif kernel == 'linear':
        return lambda x, y: np.dot(x, y.T)
    elif kernel == 'poly':
        return lambda x, y, degree=3: (np.dot(x, y.T) + 1) ** degree
    elif kernel == 'rbf':
        return lambda x, y, gamma=1.0: np.exp(-gamma * np.linalg.norm(x[:, np.newaxis] - y, axis=2) ** 2)
    elif kernel == 'sigmoid':
        return lambda x, y, gamma=1.0, coef0=0: np.tanh(gamma * np.dot(x, y.T) + coef0)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

def _svm_sgd(
    X: np.ndarray,
    y: np.ndarray,
    kernel_func: Callable,
    C: float,
    epsilon: float,
    max_iter: int,
    tol: float,
    **kwargs
) -> Dict[str, np.ndarray]:
    """Fit the SVM model using smoothed gradient descent."""
    n_samples = X.shape[0]
    alpha = np.zeros(n_samples)

    for _ in range(max_iter):
        for i in range(n_samples):
            # Compute the gradient
            grad = _compute_gradient(X, y, alpha, kernel_func, i)

            # Update alpha
            alpha[i] -= grad * C

            # Clip alpha to ensure it stays within bounds
            alpha[i] = np.clip(alpha[i], 0, C)

        # Check for convergence
        if np.linalg.norm(grad) < tol:
            break

    return {'alpha': alpha}

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    alpha: np.ndarray,
    kernel_func: Callable,
    i: int
) -> float:
    """Compute the gradient for the SVM objective."""
    K = kernel_func(X, X)
    yKalpha = np.dot(y * alpha, K[i])
    return 1 - y[i] * yKalpha

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    alpha: np.ndarray,
    kernel_func: Callable,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """Compute the metrics for the SVM model."""
    K = kernel_func(X, X)
    y_pred = np.dot(alpha * y, K)

    if callable(metric):
        return {'custom_metric': metric(y, y_pred)}
    elif metric == 'mse':
        return {'mse': np.mean((y - y_pred) ** 2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(y - y_pred))}
    elif metric == 'r2':
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return {'r2': 1 - ss_res / ss_tot}
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _svm_newton(
    X: np.ndarray,
    y: np.ndarray,
    kernel_func: Callable,
    C: float,
    epsilon: float,
    max_iter: int,
    tol: float,
    **kwargs
) -> Dict[str, np.ndarray]:
    """Fit the SVM model using Newton's method."""
    n_samples = X.shape[0]
    alpha = np.zeros(n_samples)

    for _ in range(max_iter):
        # Compute the gradient and Hessian
        grad = np.zeros(n_samples)
        hess = np.eye(n_samples)

        for i in range(n_samples):
            grad[i] = _compute_gradient(X, y, alpha, kernel_func, i)
            hess[i, i] = _compute_hessian(X, y, alpha, kernel_func, i)

        # Update alpha
        delta = np.linalg.solve(hess, grad)
        alpha -= delta * C

        # Clip alpha to ensure it stays within bounds
        alpha = np.clip(alpha, 0, C)

        # Check for convergence
        if np.linalg.norm(delta) < tol:
            break

    return {'alpha': alpha}

def _compute_hessian(
    X: np.ndarray,
    y: np.ndarray,
    alpha: np.ndarray,
    kernel_func: Callable,
    i: int
) -> float:
    """Compute the Hessian for the SVM objective."""
    K = kernel_func(X, X)
    return np.dot(y * alpha, K[i] ** 2)

def _svm_cd(
    X: np.ndarray,
    y: np.ndarray,
    kernel_func: Callable,
    C: float,
    epsilon: float,
    max_iter: int,
    tol: float,
    **kwargs
) -> Dict[str, np.ndarray]:
    """Fit the SVM model using coordinate descent."""
    n_samples = X.shape[0]
    alpha = np.zeros(n_samples)

    for _ in range(max_iter):
        for i in range(n_samples):
            # Compute the gradient
            grad = _compute_gradient(X, y, alpha, kernel_func, i)

            # Update alpha
            alpha[i] -= grad * C

            # Clip alpha to ensure it stays within bounds
            alpha[i] = np.clip(alpha[i], 0, C)

        # Check for convergence
        if np.linalg.norm(grad) < tol:
            break

    return {'alpha': alpha}

################################################################################
# kernel_methods
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or inf values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or inf values")

def _normalize_data(X: np.ndarray, y: np.ndarray,
                   normalization: str = 'standard') -> tuple:
    """Normalize input data."""
    if normalization == 'none':
        return X, y
    elif normalization == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1
        X_norm = (X - X_mean) / X_std
        y_mean = np.mean(y)
        y_std = np.std(y)
        if y_std == 0:
            y_std = 1
        y_norm = (y - y_mean) / y_std
        return X_norm, y_norm
    elif normalization == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1
        X_norm = (X - X_min) / X_range
        y_min = np.min(y)
        y_max = np.max(y)
        if y_max == y_min:
            y_norm = np.zeros_like(y)
        else:
            y_norm = (y - y_min) / (y_max - y_min)
        return X_norm, y_norm
    elif normalization == 'robust':
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_iqr[X_iqr == 0] = 1
        X_norm = (X - X_median) / X_iqr
        y_median = np.median(y)
        y_iqr = np.subtract(*np.percentile(y, [75, 25]))
        if y_iqr == 0:
            y_iqr = 1
        y_norm = (y - y_median) / y_iqr
        return X_norm, y_norm
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _compute_kernel(X: np.ndarray, kernel_func: Callable,
                   **kernel_params) -> np.ndarray:
    """Compute kernel matrix."""
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = kernel_func(X[i], X[j], **kernel_params)
    return K

def _solve_closed_form(K: np.ndarray, y: np.ndarray,
                      regularization: str = 'none',
                      alpha: float = 1.0) -> np.ndarray:
    """Solve kernel method using closed form solution."""
    if regularization == 'none':
        alpha = 0
    elif regularization == 'l1':
        # For simplicity, we'll use coordinate descent for L1
        raise NotImplementedError("L1 regularization not implemented yet")
    elif regularization == 'l2':
        alpha = alpha
    else:
        raise ValueError(f"Unknown regularization method: {regularization}")

    if alpha > 0:
        K_ridge = K + alpha * np.eye(K.shape[0])
    else:
        K_ridge = K

    try:
        alpha_hat = np.linalg.solve(K_ridge, y)
    except np.linalg.LinAlgError:
        alpha_hat = np.linalg.pinv(K_ridge) @ y

    return alpha_hat

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(y_true, y_pred)
        except Exception as e:
            metrics[f"{name}_error"] = str(e)
    return metrics

def kernel_methods_fit(X: np.ndarray, y: np.ndarray,
                      kernel_func: Callable = np.dot,
                      normalization: str = 'standard',
                      regularization: str = 'none',
                      alpha: float = 1.0,
                      metric_funcs: Optional[Dict[str, Callable]] = None,
                      **kernel_params) -> Dict:
    """
    Fit kernel method for non-linear dependence modeling.

    Parameters
    ----------
    X : np.ndarray
        Input features (n_samples, n_features)
    y : np.ndarray
        Target values (n_samples,)
    kernel_func : Callable
        Kernel function to use. Default is linear kernel (dot product).
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    regularization : str
        Regularization method ('none', 'l1', 'l2')
    alpha : float
        Regularization strength (for L1/L2)
    metric_funcs : dict
        Dictionary of metric functions to compute (name: function)
    kernel_params : dict
        Additional parameters for the kernel function

    Returns
    -------
    dict
        Dictionary containing results, metrics, parameters used, and warnings

    Example
    -------
    >>> def rbf_kernel(x1, x2, gamma=0.1):
    ...     return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)
    ...
    >>> def mse(y_true, y_pred):
    ...     return np.mean((y_true - y_pred)**2)
    ...
    >>> result = kernel_methods_fit(X, y,
    ...                           kernel_func=rbf_kernel,
    ...                           gamma=0.1,
    ...                           metric_funcs={'mse': mse})
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = _normalize_data(X, y, normalization=normalization)

    # Compute kernel matrix
    K = _compute_kernel(X_norm, kernel_func, **kernel_params)

    # Solve for coefficients
    alpha_hat = _solve_closed_form(K, y_norm,
                                 regularization=regularization,
                                 alpha=alpha)

    # Compute predictions
    y_pred = K @ alpha_hat

    # Compute metrics if provided
    metrics = {}
    warnings = []
    if metric_funcs is not None:
        try:
            metrics = _compute_metrics(y_norm, y_pred, metric_funcs)
        except Exception as e:
            warnings.append(f"Metric computation failed: {str(e)}")

    # Prepare output
    result = {
        'result': y_pred,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'kernel_func': kernel_func.__name__,
            'regularization': regularization,
            'alpha': alpha,
            **kernel_params
        },
        'warnings': warnings
    }

    return result

################################################################################
# activation_functions
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def activation_functions_fit(
    X: np.ndarray,
    y: np.ndarray,
    activation_func: Callable[[np.ndarray], np.ndarray],
    solver: str = 'gradient_descent',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    normalization: str = 'standard',
    regularization: Optional[str] = None,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Fit activation functions to data with various solvers and metrics.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    activation_func : Callable[[np.ndarray], np.ndarray]
        Activation function to apply.
    solver : str, optional
        Solver to use ('gradient_descent', 'newton', 'coordinate_descent').
    metric : str or Callable, optional
        Metric to optimize ('mse', 'mae', 'r2', 'logloss') or custom callable.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet').
    learning_rate : float, optional
        Learning rate for gradient descent.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for stopping criterion.
    random_state : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm = _normalize_data(X, normalization)
    y_norm = _normalize_data(y.reshape(-1, 1), normalization).flatten()

    # Initialize parameters
    params = _initialize_parameters(X_norm.shape[1], random_state)

    # Choose solver
    if solver == 'gradient_descent':
        params = _gradient_descent(
            X_norm, y_norm, activation_func, params,
            metric, regularization, learning_rate, max_iter, tol
        )
    elif solver == 'newton':
        params = _newton_method(
            X_norm, y_norm, activation_func, params,
            metric, regularization, max_iter, tol
        )
    elif solver == 'coordinate_descent':
        params = _coordinate_descent(
            X_norm, y_norm, activation_func, params,
            metric, regularization, max_iter, tol
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    y_pred = _predict(X_norm, params, activation_func)
    metrics = _compute_metrics(y_norm, y_pred, metric)

    return {
        'result': {'params': params},
        'metrics': metrics,
        'params_used': {
            'solver': solver,
            'metric': metric,
            'normalization': normalization,
            'regularization': regularization
        },
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

def _normalize_data(data: np.ndarray, method: str) -> np.ndarray:
    """Normalize data using specified method."""
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

def _initialize_parameters(n_features: int, random_state: Optional[int]) -> np.ndarray:
    """Initialize parameters for activation function."""
    if random_state is not None:
        np.random.seed(random_state)
    return np.random.randn(n_features + 1)  # +1 for bias

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    activation_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    regularization: Optional[str],
    learning_rate: float,
    max_iter: int,
    tol: float
) -> np.ndarray:
    """Gradient descent solver."""
    for _ in range(max_iter):
        y_pred = _predict(X, params, activation_func)
        grad = _compute_gradient(X, y, y_pred, activation_func, regularization)

        if np.linalg.norm(grad) < tol:
            break

        params -= learning_rate * grad

    return params

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    activation_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    regularization: Optional[str],
    max_iter: int,
    tol: float
) -> np.ndarray:
    """Newton method solver."""
    for _ in range(max_iter):
        y_pred = _predict(X, params, activation_func)
        grad = _compute_gradient(X, y, y_pred, activation_func, regularization)
        hess = _compute_hessian(X, y_pred, activation_func)

        if np.linalg.norm(grad) < tol:
            break

        params -= np.linalg.solve(hess, grad)

    return params

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    activation_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    regularization: Optional[str],
    max_iter: int,
    tol: float
) -> np.ndarray:
    """Coordinate descent solver."""
    for _ in range(max_iter):
        y_pred = _predict(X, params, activation_func)
        grad = _compute_gradient(X, y, y_pred, activation_func, regularization)

        if np.linalg.norm(grad) < tol:
            break

        for i in range(len(params)):
            X_i = X[:, i] if i < len(params) - 1 else np.ones(X.shape[0])
            params[i] = _update_parameter(params, i, X_i, y, activation_func, regularization)

    return params

def _update_parameter(
    params: np.ndarray,
    i: int,
    X_i: np.ndarray,
    y: np.ndarray,
    activation_func: Callable[[np.ndarray], np.ndarray],
    regularization: Optional[str]
) -> float:
    """Update single parameter in coordinate descent."""
    # Implementation depends on regularization type
    return 0.0

def _predict(X: np.ndarray, params: np.ndarray, activation_func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Make predictions using activation function."""
    linear = X @ params[:-1] + params[-1]
    return activation_func(linear)

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    activation_func: Callable[[np.ndarray], np.ndarray],
    regularization: Optional[str]
) -> np.ndarray:
    """Compute gradient of loss function."""
    # Implementation depends on activation function and regularization
    return np.zeros(X.shape[1] + 1)

def _compute_hessian(
    X: np.ndarray,
    y_pred: np.ndarray,
    activation_func: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Compute Hessian matrix."""
    # Implementation depends on activation function
    return np.eye(X.shape[1] + 1)

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute metrics."""
    if callable(metric):
        return {'custom': metric(y_true, y_pred)}

    metrics = {}
    if metric == 'mse' or not isinstance(metric, str):
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    if metric == 'mae' or not isinstance(metric, str):
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    if metric == 'r2' or not isinstance(metric, str):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - ss_res / (ss_tot + 1e-8)
    if metric == 'logloss' or not isinstance(metric, str):
        metrics['logloss'] = -np.mean(y_true * np.log(y_pred + 1e-8) +
                                    (1 - y_true) * np.log(1 - y_pred + 1e-8))

    return metrics

################################################################################
# loss_functions_non_lineaires
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def loss_functions_non_lineaires_fit(
    X: np.ndarray,
    y: np.ndarray,
    loss_func: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    normalization: str = 'none',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    weights: Optional[np.ndarray] = None
) -> Dict[str, Union[Dict, float]]:
    """
    Fit non-linear loss functions to data with configurable parameters.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    loss_func : str or callable, optional
        Loss function to minimize. Can be 'mse', 'mae', 'r2', 'logloss' or a custom callable.
    normalization : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'.
    regularization : str, optional
        Regularization type: 'none', 'l1', 'l2', or 'elasticnet'.
    tol : float, optional
        Tolerance for stopping criteria.
    max_iter : int, optional
        Maximum number of iterations.
    learning_rate : float, optional
        Learning rate for gradient-based solvers.
    custom_metric : callable, optional
        Custom metric function to evaluate the model.
    weights : np.ndarray, optional
        Weights for weighted loss calculation.

    Returns
    -------
    Dict[str, Union[Dict, float]]
        Dictionary containing:
        - 'result': Fitted parameters.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings encountered.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = loss_functions_non_lineaires_fit(X, y, loss_func='mse', solver='gradient_descent')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_norm, y_norm = _apply_normalization(X, y, normalization)

    # Prepare weights if provided
    if weights is not None:
        _validate_weights(weights, X.shape[0])

    # Initialize parameters
    params = _initialize_parameters(X_norm)

    # Select loss function
    if isinstance(loss_func, str):
        loss_func = _get_loss_function(loss_func)

    # Select solver
    if solver == 'gradient_descent':
        params = _gradient_descent(X_norm, y_norm, loss_func, params,
                                  tol=tol, max_iter=max_iter, learning_rate=learning_rate)
    elif solver == 'newton':
        params = _newton_method(X_norm, y_norm, loss_func, params,
                               tol=tol, max_iter=max_iter)
    elif solver == 'coordinate_descent':
        params = _coordinate_descent(X_norm, y_norm, loss_func, params,
                                    tol=tol, max_iter=max_iter)
    elif solver == 'closed_form':
        params = _closed_form_solution(X_norm, y_norm)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if required
    if regularization is not None:
        params = _apply_regularization(params, X_norm, y_norm, regularization)

    # Compute metrics
    metrics = _compute_metrics(X_norm, y_norm, params, loss_func, custom_metric)

    # Prepare output
    result = {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'loss_func': loss_func.__name__ if callable(loss_func) else loss_func,
            'normalization': normalization,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
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

def _validate_weights(weights: np.ndarray, n_samples: int) -> None:
    """Validate weights array."""
    if weights.shape[0] != n_samples:
        raise ValueError("Weights must have the same length as number of samples")
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative")

def _apply_normalization(X: np.ndarray, y: np.ndarray, method: str) -> Tuple[np.ndarray, np.ndarray]:
    """Apply normalization to input data."""
    if method == 'standard':
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y = (y - np.mean(y)) / np.std(y)
    elif method == 'minmax':
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif method == 'robust':
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        y = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    return X, y

def _initialize_parameters(X: np.ndarray) -> np.ndarray:
    """Initialize parameters for optimization."""
    return np.zeros(X.shape[1])

def _get_loss_function(loss_func: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get loss function based on string identifier."""
    if loss_func == 'mse':
        return _mean_squared_error
    elif loss_func == 'mae':
        return _mean_absolute_error
    elif loss_func == 'r2':
        return _r_squared
    elif loss_func == 'logloss':
        return _log_loss
    else:
        raise ValueError(f"Unknown loss function: {loss_func}")

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

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    loss_func: Callable[[np.ndarray, np.ndarray], float],
    params: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01
) -> np.ndarray:
    """Perform gradient descent optimization."""
    for _ in range(max_iter):
        y_pred = X @ params
        grad = -2 * X.T @ (y - y_pred) / len(y)
        params_new = params - learning_rate * grad
        if np.linalg.norm(params_new - params) < tol:
            break
        params = params_new
    return params

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    loss_func: Callable[[np.ndarray, np.ndarray], float],
    params: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Perform Newton's method optimization."""
    for _ in range(max_iter):
        y_pred = X @ params
        grad = -2 * X.T @ (y - y_pred) / len(y)
        hessian = 2 * X.T @ X / len(y)
        params_new = params - np.linalg.inv(hessian) @ grad
        if np.linalg.norm(params_new - params) < tol:
            break
        params = params_new
    return params

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    loss_func: Callable[[np.ndarray, np.ndarray], float],
    params: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Perform coordinate descent optimization."""
    for _ in range(max_iter):
        params_old = params.copy()
        for i in range(X.shape[1]):
            X_i = X[:, i]
            params[i] = np.linalg.lstsq(X_i.reshape(-1, 1), y - X @ params_old + params_old[i] * X_i, rcond=None)[0]
        if np.linalg.norm(params - params_old) < tol:
            break
    return params

def _closed_form_solution(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute closed-form solution."""
    return np.linalg.inv(X.T @ X) @ X.T @ y

def _apply_regularization(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply regularization to parameters."""
    if method == 'l1':
        return _l1_regularization(params, X, y)
    elif method == 'l2':
        return _l2_regularization(params, X, y)
    elif method == 'elasticnet':
        return _elasticnet_regularization(params, X, y)
    else:
        raise ValueError(f"Unknown regularization method: {method}")

def _l1_regularization(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Apply L1 regularization."""
    # Simplified implementation - in practice would use coordinate descent or similar
    alpha = 0.1
    return np.sign(params) * np.maximum(np.abs(params) - alpha, 0)

def _l2_regularization(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Apply L2 regularization."""
    # Simplified implementation - in practice would use gradient descent or similar
    alpha = 0.1
    return params / (1 + alpha * np.linalg.norm(params))

def _elasticnet_regularization(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Apply elastic net regularization."""
    # Simplified implementation - in practice would use coordinate descent or similar
    alpha = 0.1
    l1_ratio = 0.5
    return (params / (1 + alpha * l1_ratio)) * np.maximum(1, np.abs(params) / (alpha * (1 - l1_ratio)))

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    loss_func: Callable[[np.ndarray, np.ndarray], float],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    y_pred = X @ params
    metrics = {
        'loss': loss_func(y, y_pred),
        'mse': _mean_squared_error(y, y_pred),
        'mae': _mean_absolute_error(y, y_pred),
        'r2': _r_squared(y, y_pred)
    }
    if custom_metric is not None:
        metrics['custom'] = custom_metric(y, y_pred)
    return metrics

################################################################################
# normalisation_non_lineaire
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def normalisation_non_lineaire_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalisation: str = 'standard',
    distance_metric: Union[str, Callable] = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Fit a non-linear normalization model to the data.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    y : Optional[np.ndarray], default=None
        Target values if available.
    normalisation : str, default='standard'
        Type of normalization: 'none', 'standard', 'minmax', 'robust'.
    distance_metric : Union[str, Callable], default='euclidean'
        Distance metric to use: 'euclidean', 'manhattan', 'cosine', 'minkowski',
        or a custom callable.
    solver : str, default='gradient_descent'
        Solver to use: 'closed_form', 'gradient_descent', 'newton',
        'coordinate_descent'.
    regularization : Optional[str], default=None
        Regularization type: 'none', 'l1', 'l2', 'elasticnet'.
    tol : float, default=1e-4
        Tolerance for stopping criteria.
    max_iter : int, default=1000
        Maximum number of iterations.
    custom_metric : Optional[Callable], default=None
        Custom metric function if needed.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Initialize parameters
    params = {
        'normalisation': normalisation,
        'distance_metric': distance_metric,
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'max_iter': max_iter
    }

    # Apply normalization
    X_normalized = _apply_normalization(X, normalisation)

    # Choose distance metric
    distance_func = _get_distance_function(distance_metric)

    # Solve the normalization problem
    result, metrics = _solve_normalization(
        X_normalized,
        y,
        distance_func,
        solver,
        regularization,
        tol,
        max_iter,
        custom_metric,
        **kwargs
    )

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y is not None and len(X) != len(y):
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values.")
    if y is not None and np.any(np.isnan(y)):
        raise ValueError("y contains NaN values.")

def _apply_normalization(X: np.ndarray, normalisation: str) -> np.ndarray:
    """Apply the specified normalization to the data."""
    if normalisation == 'none':
        return X
    elif normalisation == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif normalisation == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif normalisation == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization type: {normalisation}")

def _get_distance_function(metric: Union[str, Callable]) -> Callable:
    """Get the distance function based on the metric."""
    if callable(metric):
        return metric
    elif metric == 'euclidean':
        return lambda x, y: np.linalg.norm(x - y)
    elif metric == 'manhattan':
        return lambda x, y: np.sum(np.abs(x - y))
    elif metric == 'cosine':
        return lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    elif metric == 'minkowski':
        return lambda x, y, p=3: np.sum(np.abs(x - y) ** p) ** (1 / p)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _solve_normalization(
    X: np.ndarray,
    y: Optional[np.ndarray],
    distance_func: Callable,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    custom_metric: Optional[Callable],
    **kwargs
) -> tuple:
    """Solve the normalization problem using the specified solver."""
    if solver == 'closed_form':
        return _solve_closed_form(X, y, distance_func)
    elif solver == 'gradient_descent':
        return _solve_gradient_descent(X, y, distance_func, regularization, tol, max_iter, **kwargs)
    elif solver == 'newton':
        return _solve_newton(X, y, distance_func, regularization, tol, max_iter, **kwargs)
    elif solver == 'coordinate_descent':
        return _solve_coordinate_descent(X, y, distance_func, regularization, tol, max_iter, **kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _solve_closed_form(X: np.ndarray, y: Optional[np.ndarray], distance_func: Callable) -> tuple:
    """Solve the normalization problem in closed form."""
    # Placeholder for closed-form solution
    result = np.zeros_like(X)
    metrics = {'mse': 0.0}
    return result, metrics

def _solve_gradient_descent(
    X: np.ndarray,
    y: Optional[np.ndarray],
    distance_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    **kwargs
) -> tuple:
    """Solve the normalization problem using gradient descent."""
    # Placeholder for gradient descent solution
    result = np.zeros_like(X)
    metrics = {'mse': 0.0}
    return result, metrics

def _solve_newton(
    X: np.ndarray,
    y: Optional[np.ndarray],
    distance_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    **kwargs
) -> tuple:
    """Solve the normalization problem using Newton's method."""
    # Placeholder for Newton's method solution
    result = np.zeros_like(X)
    metrics = {'mse': 0.0}
    return result, metrics

def _solve_coordinate_descent(
    X: np.ndarray,
    y: Optional[np.ndarray],
    distance_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    **kwargs
) -> tuple:
    """Solve the normalization problem using coordinate descent."""
    # Placeholder for coordinate descent solution
    result = np.zeros_like(X)
    metrics = {'mse': 0.0}
    return result, metrics

################################################################################
# transformations_non_lineaires
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def transformations_non_lineaires_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalisation: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_transform: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Fit non-linear transformations to data.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : Optional[np.ndarray]
        Target values array of shape (n_samples,). If None, unsupervised mode.
    normalisation : str
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : Union[str, Callable]
        Metric to optimize: 'mse', 'mae', 'r2', 'logloss', or custom callable.
    distance : Union[str, Callable]
        Distance metric: 'euclidean', 'manhattan', 'cosine', 'minkowski', or custom.
    solver : str
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'.
    regularization : Optional[str]
        Regularization type: None, 'l1', 'l2', or 'elasticnet'.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_transform : Optional[Callable]
        Custom transformation function to apply.
    **kwargs
        Additional solver-specific parameters.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, used parameters, and warnings.

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = transformations_non_lineaires_fit(X, y, normalisation='standard', metric='mse')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _apply_normalization(X, normalisation)

    # Apply custom transform if provided
    if custom_transform is not None:
        X_normalized = custom_transform(X_normalized)

    # Initialize parameters
    params = _initialize_parameters(X_normalized.shape[1], y is not None)

    # Choose solver
    if solver == 'closed_form':
        params = _solve_closed_form(X_normalized, y)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(X_normalized, y, metric, distance,
                                        regularization, tol, max_iter, **kwargs)
    elif solver == 'newton':
        params = _solve_newton(X_normalized, y, metric, distance,
                              regularization, tol, max_iter, **kwargs)
    elif solver == 'coordinate_descent':
        params = _solve_coordinate_descent(X_normalized, y, metric, distance,
                                          regularization, tol, max_iter, **kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(X_normalized, y, params, metric)

    # Prepare output
    result = {
        'result': params,
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

def _validate_inputs(X: np.ndarray, y: Optional[np.ndarray]) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or (y is not None and np.any(np.isnan(y))):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or (y is not None and np.any(np.isinf(y))):
        raise ValueError("Input arrays must not contain infinite values")

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to data."""
    if method == 'none':
        return X
    elif method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
        return (X - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _initialize_parameters(n_features: int, is_supervised: bool) -> np.ndarray:
    """Initialize parameters for optimization."""
    if is_supervised:
        return np.zeros(n_features + 1)  # +1 for bias term
    else:
        return np.zeros(n_features)

def _solve_closed_form(X: np.ndarray, y: Optional[np.ndarray]) -> np.ndarray:
    """Solve using closed-form solution."""
    if y is None:
        raise ValueError("Closed form solution requires supervised learning (y cannot be None)")
    X_with_bias = np.column_stack([X, np.ones(X.shape[0])])
    params = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
    return params

def _solve_gradient_descent(
    X: np.ndarray,
    y: Optional[np.ndarray],
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    **kwargs
) -> np.ndarray:
    """Solve using gradient descent."""
    if y is None:
        raise ValueError("Gradient descent requires supervised learning (y cannot be None)")
    params = _initialize_parameters(X.shape[1], True)
    learning_rate = kwargs.get('learning_rate', 0.01)

    for _ in range(max_iter):
        gradients = _compute_gradient(X, y, params, metric, distance, regularization)
        params -= learning_rate * gradients
        if np.linalg.norm(gradients) < tol:
            break

    return params

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str]
) -> np.ndarray:
    """Compute gradient for optimization."""
    X_with_bias = np.column_stack([X, np.ones(X.shape[0])])
    predictions = X_with_bias @ params

    if metric == 'mse':
        error = predictions - y
    elif callable(metric):
        error = metric(predictions, y)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    gradient = X_with_bias.T @ error / y.size

    if regularization == 'l1':
        gradient[1:] += np.sign(params[1:])
    elif regularization == 'l2':
        gradient[1:] += params[1:]
    elif regularization == 'elasticnet':
        gradient[1:] += (np.sign(params[1:]) + params[1:])

    return gradient

def _calculate_metrics(
    X: np.ndarray,
    y: Optional[np.ndarray],
    params: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Calculate performance metrics."""
    if y is None:
        return {'metrics': {}}

    X_with_bias = np.column_stack([X, np.ones(X.shape[0])])
    predictions = X_with_bias @ params

    metrics_dict = {}

    if metric == 'mse':
        mse = np.mean((predictions - y) ** 2)
        metrics_dict['mse'] = mse
    elif metric == 'mae':
        mae = np.mean(np.abs(predictions - y))
        metrics_dict['mae'] = mae
    elif metric == 'r2':
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        metrics_dict['r2'] = r2
    elif callable(metric):
        custom_metric = metric(predictions, y)
        metrics_dict['custom'] = custom_metric
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics_dict

def _solve_newton(
    X: np.ndarray,
    y: Optional[np.ndarray],
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    **kwargs
) -> np.ndarray:
    """Solve using Newton's method."""
    if y is None:
        raise ValueError("Newton's method requires supervised learning (y cannot be None)")
    params = _initialize_parameters(X.shape[1], True)

    for _ in range(max_iter):
        gradient = _compute_gradient(X, y, params, metric, distance, regularization)
        hessian = _compute_hessian(X, y, params, metric)

        if np.linalg.cond(hessian) < 1 / tol:
            params -= np.linalg.inv(hessian) @ gradient
        else:
            params -= 1e-4 * gradient

        if np.linalg.norm(gradient) < tol:
            break

    return params

def _compute_hessian(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable]
) -> np.ndarray:
    """Compute Hessian matrix for Newton's method."""
    X_with_bias = np.column_stack([X, np.ones(X.shape[0])])
    predictions = X_with_bias @ params

    if metric == 'mse':
        hessian = 2 * X_with_bias.T @ X_with_bias / y.size
    elif callable(metric):
        hessian = metric(predictions, y, hessian=True)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return hessian

def _solve_coordinate_descent(
    X: np.ndarray,
    y: Optional[np.ndarray],
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    **kwargs
) -> np.ndarray:
    """Solve using coordinate descent."""
    if y is None:
        raise ValueError("Coordinate descent requires supervised learning (y cannot be None)")
    params = _initialize_parameters(X.shape[1], True)
    learning_rate = kwargs.get('learning_rate', 0.01)

    for _ in range(max_iter):
        for i in range(params.size):
            # Save current parameter
            old_param = params[i]

            # Compute gradient for this parameter
            gradients = _compute_gradient(X, y, params, metric, distance, regularization)

            # Update parameter
            params[i] -= learning_rate * gradients[i]

        if np.linalg.norm(gradients) < tol:
            break

    return params

################################################################################
# interaction_terms
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def interaction_terms_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
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
    Fit interaction terms model to data.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    normalizer : Optional[Callable]
        Function to normalize features. Defaults to None.
    metric : Union[str, Callable]
        Metric to evaluate model performance. Can be 'mse', 'mae', 'r2', or custom callable.
    distance : str
        Distance metric for interaction terms. Defaults to 'euclidean'.
    solver : str
        Solver method. Can be 'closed_form', 'gradient_descent', or 'newton'.
    regularization : Optional[str]
        Regularization type. Can be 'l1', 'l2', or None.
    tol : float
        Tolerance for convergence. Defaults to 1e-4.
    max_iter : int
        Maximum number of iterations. Defaults to 1000.
    custom_metric : Optional[Callable]
        Custom metric function if not using built-in metrics.
    custom_distance : Optional[Callable]
        Custom distance function if not using built-in distances.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Input validation
    _validate_inputs(X, y)

    # Normalize data if normalizer is provided
    X_normalized = _apply_normalization(X, normalizer)

    # Prepare interaction terms
    X_interaction = _prepare_interaction_terms(X_normalized)

    # Solve for interaction terms
    coefficients = _solve_interaction_terms(
        X_interaction, y,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter
    )

    # Calculate metrics
    metrics = _calculate_metrics(
        y, X_interaction @ coefficients,
        metric=metric,
        custom_metric=custom_metric
    )

    # Prepare output
    result = {
        'result': coefficients,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
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

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to features."""
    if normalizer is None:
        return X
    try:
        return normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _prepare_interaction_terms(X: np.ndarray) -> np.ndarray:
    """Prepare interaction terms matrix."""
    n_samples, n_features = X.shape
    interaction_matrix = np.zeros((n_samples, n_features * (n_features + 1) // 2))

    idx = 0
    for i in range(n_features):
        interaction_matrix[:, idx] = X[:, i]
        idx += 1
        for j in range(i + 1, n_features):
            interaction_matrix[:, idx] = X[:, i] * X[:, j]
            idx += 1

    return interaction_matrix

def _solve_interaction_terms(
    X: np.ndarray,
    y: np.ndarray,
    *,
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Solve for interaction terms coefficients."""
    if solver == 'closed_form':
        return _solve_closed_form(X, y, regularization)
    elif solver == 'gradient_descent':
        return _solve_gradient_descent(X, y, tol=tol, max_iter=max_iter)
    elif solver == 'newton':
        return _solve_newton(X, y, tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _solve_closed_form(X: np.ndarray, y: np.ndarray, regularization: Optional[str]) -> np.ndarray:
    """Closed form solution for interaction terms."""
    if regularization is None:
        coefficients = np.linalg.pinv(X) @ y
    elif regularization == 'l2':
        coefficients = _ridge_regression(X, y)
    else:
        raise ValueError(f"Unsupported regularization: {regularization}")
    return coefficients

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Gradient descent solution for interaction terms."""
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ coefficients - y) / len(y)
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
    max_iter: int = 1000
) -> np.ndarray:
    """Newton's method solution for interaction terms."""
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)

    for _ in range(max_iter):
        residuals = X @ coefficients - y
        gradient = 2 * X.T @ residuals / len(y)
        hessian = 2 * X.T @ X / len(y)

        try:
            new_coefficients = coefficients - np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break

        if np.linalg.norm(new_coefficients - coefficients) < tol:
            break
        coefficients = new_coefficients

    return coefficients

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: Union[str, Callable] = 'mse',
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Calculate evaluation metrics."""
    metrics_dict = {}

    if metric == 'mse':
        metrics_dict['mse'] = np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        metrics_dict['mae'] = np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        metrics_dict['r2'] = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    elif callable(metric):
        metrics_dict['custom_metric'] = metric(y_true, y_pred)
    elif custom_metric is not None:
        metrics_dict['custom_metric'] = custom_metric(y_true, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics_dict

def _ridge_regression(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Ridge regression implementation."""
    I = np.identity(X.shape[1])
    coefficients = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    return coefficients

# Example usage:
"""
X = np.random.rand(100, 3)
y = X[:, 0] * X[:, 1] + np.random.normal(0, 0.1, 100)

result = interaction_terms_fit(
    X,
    y,
    normalizer=None,
    metric='mse',
    solver='closed_form'
)

print(result['metrics'])
"""

################################################################################
# polynomial_features
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_inputs(X: np.ndarray, degree: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    if degree < 1:
        raise ValueError("degree must be at least 1")

def normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
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
        raise ValueError(f"Unknown normalization method: {method}")

def compute_polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """Compute polynomial features."""
    n_samples, n_features = X.shape
    n_output_features = int(np.sum([degree + 1 for _ in range(n_features)]))
    X_poly = np.ones((n_samples, n_output_features))

    column_index = 0
    for i in range(n_features):
        for d in range(1, degree + 1):
            X_poly[:, column_index] = np.power(X[:, i], d)
            column_index += 1

    return X_poly

def polynomial_features_fit(
    X: np.ndarray,
    degree: int = 2,
    normalize_method: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form'
) -> Dict[str, Union[np.ndarray, Dict, str]]:
    """
    Fit polynomial features to the input data.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    degree : int, optional
        Degree of the polynomial features, by default 2.
    normalize_method : str, optional
        Normalization method, by default 'standard'.
    metric : Union[str, Callable], optional
        Metric to evaluate the fit, by default 'mse'.
    solver : str, optional
        Solver to use for fitting, by default 'closed_form'.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict, str]]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    validate_inputs(X, degree)
    X_normalized = normalize_data(X, method=normalize_method)
    X_poly = compute_polynomial_features(X_normalized, degree)

    # Placeholder for the actual fitting logic
    if solver == 'closed_form':
        # Example: Closed form solution for linear regression
        y = np.random.rand(X_poly.shape[0])  # Placeholder for actual target
        coefficients = np.linalg.pinv(X_poly.T @ X_poly) @ X_poly.T @ y
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Placeholder for metrics calculation
    if isinstance(metric, str):
        if metric == 'mse':
            y_pred = X_poly @ coefficients
            mse_value = np.mean((y - y_pred) ** 2)
            metrics = {'mse': mse_value}
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        metrics = {'custom_metric': metric(X_poly, coefficients)}

    return {
        'result': X_poly,
        'metrics': metrics,
        'params_used': {
            'degree': degree,
            'normalize_method': normalize_method,
            'metric': metric,
            'solver': solver
        },
        'warnings': []
    }

# Example usage:
# X = np.random.rand(100, 3)
# result = polynomial_features_fit(X, degree=2, normalize_method='standard', metric='mse')

################################################################################
# spline_regression
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def spline_regression_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    degree: int = 3,
    n_knots: Optional[int] = None,
    knots: Optional[np.ndarray] = None,
    normalization: str = "standard",
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit a spline regression model to the data.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    degree : int, optional
        Degree of the spline basis functions. Default is 3.
    n_knots : int, optional
        Number of knots to use. If None, uses default based on data size.
    knots : np.ndarray, optional
        Custom array of knot positions. If provided, overrides n_knots.
    normalization : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'. Default is 'standard'.
    metric : str or callable, optional
        Metric to evaluate model performance: 'mse', 'mae', 'r2', or custom callable. Default is 'mse'.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', or 'newton'. Default is 'closed_form'.
    regularization : str, optional
        Regularization type: 'none', 'l1', 'l2', or 'elasticnet'. Default is None.
    alpha : float, optional
        Regularization strength. Default is 1.0.
    tol : float, optional
        Tolerance for convergence. Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    random_state : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': Fitted model coefficients.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings encountered.

    Example
    -------
    >>> X = np.random.rand(100, 2)
    >>> y = np.sin(X[:, 0] + X[:, 1])
    >>> result = spline_regression_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_norm, y_norm = _apply_normalization(X, y, normalization)

    # Prepare spline basis
    if knots is None:
        knots = _determine_knots(X, n_knots)
    basis = _create_spline_basis(X_norm, knots, degree)

    # Choose solver
    if solver == "closed_form":
        coefficients = _solve_closed_form(basis, y_norm)
    elif solver == "gradient_descent":
        coefficients = _solve_gradient_descent(
            basis, y_norm, alpha=alpha, tol=tol, max_iter=max_iter,
            random_state=random_state
        )
    elif solver == "newton":
        coefficients = _solve_newton(
            basis, y_norm, alpha=alpha, tol=tol, max_iter=max_iter,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if required
    if regularization is not None:
        coefficients = _apply_regularization(
            basis, y_norm, coefficients, regularization, alpha
        )

    # Compute metrics
    y_pred = _predict(basis, coefficients)
    metrics = _compute_metrics(y_norm, y_pred, metric)

    # Prepare output
    return {
        "result": coefficients,
        "metrics": metrics,
        "params_used": {
            "degree": degree,
            "n_knots": len(knots) if knots is not None else n_knots,
            "normalization": normalization,
            "metric": metric,
            "solver": solver,
            "regularization": regularization,
            "alpha": alpha
        },
        "warnings": []
    }

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
) -> tuple[np.ndarray, np.ndarray]:
    """Apply normalization to input data."""
    if method == "none":
        return X, y
    elif method == "standard":
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y_norm = (y - np.mean(y)) / np.std(y)
    elif method == "minmax":
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif method == "robust":
        X_norm = (X - np.median(X, axis=0)) / (
            np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
        )
        y_norm = (y - np.median(y)) / (
            np.percentile(y, 75) - np.percentile(y, 25)
        )
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return X_norm, y_norm

def _determine_knots(X: np.ndarray, n_knots: Optional[int]) -> np.ndarray:
    """Determine knot positions."""
    if n_knots is None:
        n_knots = min(10, X.shape[0] // 2)
    quantiles = np.linspace(0, 1, n_knots + 2)[1:-1]
    return np.percentile(X, quantiles * 100)

def _create_spline_basis(
    X: np.ndarray,
    knots: np.ndarray,
    degree: int
) -> np.ndarray:
    """Create spline basis matrix."""
    n_samples, n_features = X.shape
    basis = np.zeros((n_samples, len(knots) * (degree + 1)))
    for i in range(n_features):
        basis[:, i*(degree+1):(i+1)*(degree+1)] = _create_b_spline(X[:, i], knots, degree)
    return basis

def _create_b_spline(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """Create B-spline basis for 1D data."""
    n_samples = len(x)
    n_basis = len(knots) + degree
    basis = np.zeros((n_samples, n_basis))
    for i in range(n_basis):
        basis[:, i] = _b_spline(x, knots, degree, i)
    return basis

def _b_spline(
    x: np.ndarray,
    knots: np.ndarray,
    degree: int,
    i: int
) -> np.ndarray:
    """Compute B-spline basis function."""
    if degree == 0:
        return (x >= knots[i]) & (x < knots[i+1])
    coeff1 = _b_spline(x, knots, degree-1, i) * (
        (x - knots[i]) / (knots[i+degree] - knots[i])
    )
    coeff2 = _b_spline(x, knots, degree-1, i+1) * (
        (knots[i+degree+1] - x) / (knots[i+degree+1] - knots[i+1])
    )
    return coeff1 + coeff2

def _solve_closed_form(basis: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve using closed-form solution."""
    return np.linalg.pinv(basis) @ y

def _solve_gradient_descent(
    basis: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1.0,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> np.ndarray:
    """Solve using gradient descent."""
    if random_state is not None:
        np.random.seed(random_state)
    coefficients = np.random.randn(basis.shape[1])
    for _ in range(max_iter):
        gradient = 2 * basis.T @ (basis @ coefficients - y)
        coefficients -= alpha * gradient
        if np.linalg.norm(gradient) < tol:
            break
    return coefficients

def _solve_newton(
    basis: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1.0,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> np.ndarray:
    """Solve using Newton's method."""
    if random_state is not None:
        np.random.seed(random_state)
    coefficients = np.random.randn(basis.shape[1])
    for _ in range(max_iter):
        residuals = basis @ coefficients - y
        gradient = 2 * basis.T @ residuals
        hessian = 2 * basis.T @ basis
        coefficients -= np.linalg.pinv(hessian) @ gradient
        if np.linalg.norm(gradient) < tol:
            break
    return coefficients

def _apply_regularization(
    basis: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    method: str,
    alpha: float
) -> np.ndarray:
    """Apply regularization to coefficients."""
    if method == "l1":
        return _soft_threshold(coefficients, alpha)
    elif method == "l2":
        return coefficients / (1 + 2 * alpha * np.linalg.norm(coefficients))
    elif method == "elasticnet":
        l1_coeff = _soft_threshold(coefficients, alpha * 0.5)
        l2_coeff = coefficients / (1 + alpha * 0.5 * np.linalg.norm(coefficients))
        return l1_coeff + l2_coeff
    else:
        raise ValueError(f"Unknown regularization method: {method}")

def _soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """Apply soft thresholding operator."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def _predict(basis: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Make predictions using the fitted model."""
    return basis @ coefficients

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}
    if isinstance(metric, str):
        if metric == "mse":
            metrics["mse"] = np.mean((y_true - y_pred) ** 2)
        elif metric == "mae":
            metrics["mae"] = np.mean(np.abs(y_true - y_pred))
        elif metric == "r2":
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics["r2"] = 1 - ss_res / ss_tot
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        metrics["custom"] = metric(y_true, y_pred)
    return metrics

################################################################################
# gaussian_processes
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def _normalize_data(X: np.ndarray, y: np.ndarray,
                   normalization: str = 'standard',
                   custom_norm: Optional[Callable] = None) -> tuple:
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
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - X_median) / (X_iqr + 1e-8)
        y_median = np.median(y)
        y_iqr = np.subtract(*np.percentile(y, [75, 25]))
        y_norm = (y - y_median) / (y_iqr + 1e-8)
    else:
        X_norm = X.copy()
        y_norm = y.copy()

    return X_norm, y_norm

def _compute_kernel(X: np.ndarray, kernel_func: Callable,
                   kernel_params: Dict) -> np.ndarray:
    """Compute the kernel matrix."""
    return kernel_func(X, **kernel_params)

def _compute_closed_form_solution(K: np.ndarray,
                                y: np.ndarray,
                                alpha: float = 1e-8) -> np.ndarray:
    """Compute the closed form solution for GP parameters."""
    K += alpha * np.eye(K.shape[0])
    return np.linalg.solve(K, y)

def _compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute evaluation metrics."""
    return {name: func(y_true, y_pred) for name, func in metric_funcs.items()}

def gaussian_processes_fit(X: np.ndarray,
                          y: np.ndarray,
                          kernel_func: Callable = None,
                          kernel_params: Dict = None,
                          normalization: str = 'standard',
                          custom_norm: Optional[Callable] = None,
                          solver: str = 'closed_form',
                          alpha: float = 1e-8,
                          metric_funcs: Dict[str, Callable] = None) -> Dict:
    """
    Fit a Gaussian Process model to the data.

    Parameters
    ----------
    X : np.ndarray
        Input features (n_samples, n_features)
    y : np.ndarray
        Target values (n_samples,)
    kernel_func : Callable
        Kernel function to use. Default is RBF kernel.
    kernel_params : Dict
        Parameters for the kernel function.
    normalization : str or Callable
        Normalization method to apply. Options: 'standard', 'minmax', 'robust'
    custom_norm : Callable
        Custom normalization function.
    solver : str
        Solver to use. Options: 'closed_form'
    alpha : float
        Regularization parameter.
    metric_funcs : Dict[str, Callable]
        Dictionary of metric functions to compute.

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set default kernel if not provided
    if kernel_func is None:
        def rbf_kernel(X, length_scale=1.0):
            pairwise_sq_dists = np.sum(X**2, axis=1)[:, None] + \
                              np.sum(X**2, axis=1)[None, :] - \
                              2 * np.dot(X, X.T)
            return np.exp(-0.5 / length_scale**2 * pairwise_sq_dists)
        kernel_func = rbf_kernel
    if kernel_params is None:
        kernel_params = {}

    # Normalize data
    X_norm, y_norm = _normalize_data(X, y, normalization, custom_norm)

    # Compute kernel matrix
    K = _compute_kernel(X_norm, kernel_func, kernel_params)

    # Solve for parameters
    if solver == 'closed_form':
        params = _compute_closed_form_solution(K, y_norm, alpha)
    else:
        raise ValueError(f"Solver {solver} not implemented")

    # Compute predictions
    y_pred = np.dot(K, params)

    # Compute metrics if provided
    metrics = {}
    if metric_funcs is not None:
        metrics = _compute_metrics(y, y_pred, metric_funcs)

    # Prepare output
    result = {
        'result': y_pred,
        'metrics': metrics,
        'params_used': {
            'kernel_func': kernel_func.__name__,
            'kernel_params': kernel_params,
            'normalization': normalization,
            'solver': solver,
            'alpha': alpha
        },
        'warnings': []
    }

    return result

# Example usage:
"""
from sklearn.metrics import mean_squared_error, r2_score

X = np.random.rand(100, 5)
y = np.sin(X[:, 0]) + 0.1 * np.random.randn(100)

metrics = {
    'mse': mean_squared_error,
    'r2': r2_score
}

result = gaussian_processes_fit(X, y,
                              kernel_params={'length_scale': 0.5},
                              metric_funcs=metrics)
"""
