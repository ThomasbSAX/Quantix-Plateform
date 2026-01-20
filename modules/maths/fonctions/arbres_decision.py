"""
Quantix – Module arbres_decision
Généré automatiquement
Date: 2026-01-06
"""

################################################################################
# critere_gini
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> None:
    """
    Validate input arrays for Gini criterion computation.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels or probabilities.
    sample_weight : Optional[np.ndarray], default=None
        Sample weights.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if y_true.ndim != 1:
        raise ValueError("y_true must be 1-dimensional")
    if y_pred.ndim != 1:
        raise ValueError("y_pred must be 1-dimensional")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    if sample_weight is not None:
        if len(sample_weight) != len(y_true):
            raise ValueError("sample_weight must have the same length as y_true")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight must contain non-negative values")

def _compute_gini(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> float:
    """
    Compute Gini impurity for a given split.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted probabilities for the positive class.
    sample_weight : Optional[np.ndarray], default=None
        Sample weights.

    Returns
    ------
    float
        Gini impurity value.
    """
    if sample_weight is None:
        sample_weight = np.ones_like(y_true)

    n_samples = len(y_true)
    weighted_y_true = y_true * sample_weight
    weighted_pred = y_pred * sample_weight

    # Compute probabilities
    p_positive = np.sum(weighted_y_true) / np.sum(sample_weight)
    p_negative = 1 - p_positive

    # Compute Gini impurity
    gini = 2 * p_positive * p_negative

    return float(gini)

def _compute_weighted_gini(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> float:
    """
    Compute weighted Gini impurity for a split.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted probabilities for the positive class.
    sample_weight : Optional[np.ndarray], default=None
        Sample weights.

    Returns
    ------
    float
        Weighted Gini impurity value.
    """
    if sample_weight is None:
        return _compute_gini(y_true, y_pred)

    n_samples = len(y_true)
    total_weight = np.sum(sample_weight)

    # Compute probabilities for each class
    weighted_y_true = y_true * sample_weight
    p_positive = np.sum(weighted_y_true) / total_weight

    # Compute weighted Gini
    gini = 2 * p_positive * (1 - p_positive)

    return float(gini)

def critere_gini_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    weighted: bool = False
) -> Dict[str, Union[float, Dict]]:
    """
    Compute Gini criterion for decision tree splits.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (binary classification).
    y_pred : np.ndarray
        Predicted probabilities for the positive class.
    sample_weight : Optional[np.ndarray], default=None
        Sample weights.
    weighted : bool, default=False
        Whether to use weighted Gini impurity.

    Returns
    ------
    Dict[str, Union[float, Dict]]
        Dictionary containing:
        - "result": computed Gini value
        - "metrics": dictionary of additional metrics
        - "params_used": parameters used in computation
        - "warnings": any warnings generated

    Examples
    --------
    >>> y_true = np.array([1, 0, 1, 1])
    >>> y_pred = np.array([0.9, 0.1, 0.8, 0.7])
    >>> critere_gini_fit(y_true, y_pred)
    {
        'result': 0.42,
        'metrics': {'p_positive': 0.75, 'p_negative': 0.25},
        'params_used': {'weighted': False, 'sample_weight': None},
        'warnings': []
    }
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred, sample_weight)

    # Choose computation method based on parameters
    if weighted:
        gini = _compute_weighted_gini(y_true, y_pred, sample_weight)
    else:
        gini = _compute_gini(y_true, y_pred, sample_weight)

    # Compute additional metrics
    if sample_weight is None:
        p_positive = np.mean(y_true)
    else:
        p_positive = np.sum(y_true * sample_weight) / np.sum(sample_weight)

    metrics = {
        'p_positive': float(p_positive),
        'p_negative': 1 - p_positive
    }

    # Prepare output dictionary
    result = {
        'result': gini,
        'metrics': metrics,
        'params_used': {
            'weighted': weighted,
            'sample_weight': sample_weight is not None
        },
        'warnings': []
    }

    return result

################################################################################
# critere_entropy
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> None:
    """Validate input arrays."""
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-dimensional arrays")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if sample_weight is not None:
        if len(sample_weight) != len(y_true):
            raise ValueError("sample_weight must have the same length as y_true")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight values must be non-negative")

def _compute_class_probabilities(
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute class probabilities with optional sample weights."""
    if sample_weight is None:
        sample_weight = np.ones_like(y, dtype=float)

    classes, counts = np.unique(y, return_counts=True)
    weighted_counts = np.bincount(y.astype(int), weights=sample_weight)

    total_weight = np.sum(sample_weight)
    if total_weight == 0:
        raise ValueError("Total sample weight cannot be zero")

    probs = weighted_counts / total_weight
    return {cls: prob for cls, prob in zip(classes, probs)}

def _compute_entropy(
    probabilities: Dict[str, float]
) -> float:
    """Compute entropy from class probabilities."""
    entropy = 0.0
    for prob in probabilities.values():
        if prob > 0:
            entropy -= prob * np.log2(prob)
    return entropy

def critere_entropy_compute(
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
    compute_metrics: bool = True
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute entropy criterion for decision trees.

    Parameters:
    -----------
    y_true : np.ndarray
        True class labels.
    y_pred : Optional[np.ndarray]
        Predicted class labels. If None, computes entropy of the true distribution.
    sample_weight : Optional[np.ndarray]
        Sample weights. Default is None (equal weights).
    compute_metrics : bool
        Whether to compute additional metrics. Default is True.

    Returns:
    --------
    Dict containing:
        - "result": computed entropy
        - "metrics": additional metrics if compute_metrics is True
        - "params_used": parameters used in computation
        - "warnings": any warnings generated

    Example:
    --------
    >>> y_true = np.array([0, 1, 0, 1])
    >>> result = critere_entropy_compute(y_true)
    """
    _validate_inputs(y_true, y_pred if y_pred is not None else np.zeros_like(y_true), sample_weight)

    params_used = {
        "sample_weight": "provided" if sample_weight is not None else "uniform",
        "compute_metrics": compute_metrics
    }

    warnings = []

    if y_pred is None:
        probabilities = _compute_class_probabilities(y_true, sample_weight)
    else:
        # For decision trees, we typically compute entropy of the true distribution
        probabilities = _compute_class_probabilities(y_true, sample_weight)

    entropy = _compute_entropy(probabilities)

    metrics = {}
    if compute_metrics:
        metrics["class_probabilities"] = probabilities
        metrics["num_classes"] = len(probabilities)

    return {
        "result": entropy,
        "metrics": metrics if compute_metrics else {},
        "params_used": params_used,
        "warnings": warnings
    }

################################################################################
# critere_mse
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> None:
    """Validate input arrays."""
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-dimensional arrays")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if sample_weight is not None:
        if len(sample_weight) != len(y_true):
            raise ValueError("sample_weight must have the same length as y_true")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative values")

def _compute_mse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> float:
    """Compute Mean Squared Error."""
    residuals = y_true - y_pred
    if sample_weight is None:
        return np.mean(residuals ** 2)
    else:
        weighted_residuals = residuals * sample_weight
        return np.sum(weighted_residuals ** 2) / np.sum(sample_weight)

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    metric_funcs: Dict[str, Callable] = None
) -> Dict[str, float]:
    """Compute additional metrics."""
    if metric_funcs is None:
        return {}

    metrics = {}
    for name, func in metric_funcs.items():
        try:
            if sample_weight is None:
                metrics[name] = func(y_true, y_pred)
            else:
                metrics[name] = func(y_true, y_pred, sample_weight=sample_weight)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def critere_mse_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    metric_funcs: Optional[Dict[str, Callable]] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute Mean Squared Error (MSE) criterion for decision trees.

    Parameters:
    -----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.
    sample_weight : Optional[np.ndarray], default=None
        Individual weights for each sample.
    metric_funcs : Optional[Dict[str, Callable]], default=None
        Additional metrics to compute (e.g., {'mae': mae_func}).

    Returns:
    --------
    Dict containing:
        - 'result': float, the computed MSE
        - 'metrics': dict of additional metrics
        - 'params_used': dict of parameters used
        - 'warnings': list of warnings

    Example:
    --------
    >>> y_true = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5, 0.0, 2, 8])
    >>> result = critere_mse_fit(y_true, y_pred)
    """
    _validate_inputs(y_true, y_pred, sample_weight)

    result = _compute_mse(y_true, y_pred, sample_weight)
    metrics = _compute_metrics(y_true, y_pred, sample_weight, metric_funcs)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'sample_weight': sample_weight is not None,
            'additional_metrics': metric_funcs is not None
        },
        'warnings': []
    }

################################################################################
# splitting_strategy
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def splitting_strategy_fit(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable] = 'gini',
    max_depth: int = 5,
    min_samples_split: int = 2,
    min_impurity_decrease: float = 0.0,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute the optimal splitting strategy for a decision tree.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    metric : str or callable, optional
        The impurity criterion to evaluate splits. Options: 'gini', 'entropy',
        or a custom callable function.
    max_depth : int, optional
        Maximum depth of the tree.
    min_samples_split : int, optional
        Minimum number of samples required to split a node.
    min_impurity_decrease : float, optional
        Minimum impurity decrease required for a split to be performed.
    random_state : int or None, optional
        Random seed for reproducibility.

    Returns:
    --------
    dict
        A dictionary containing the splitting strategy results, metrics,
        parameters used, and any warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set random state if provided
    rng = np.random.RandomState(random_state) if random_state is not None else None

    # Initialize result dictionary
    result = {
        'result': None,
        'metrics': {},
        'params_used': {
            'metric': metric,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_impurity_decrease': min_impurity_decrease
        },
        'warnings': []
    }

    # Compute the splitting strategy
    result['result'] = _compute_splitting_strategy(
        X, y, metric, max_depth, min_samples_split,
        min_impurity_decrease, rng
    )

    # Compute metrics
    result['metrics'] = _compute_metrics(result['result'], X, y)

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate the input data."""
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains NaN or infinite values.")
    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("y contains NaN or infinite values.")

def _compute_splitting_strategy(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    max_depth: int,
    min_samples_split: int,
    min_impurity_decrease: float,
    rng: Optional[np.random.RandomState]
) -> Dict[str, Any]:
    """Compute the optimal splitting strategy."""
    # Initialize the tree structure
    tree = {
        'left': None,
        'right': None,
        'feature': None,
        'threshold': None,
        'impurity': _compute_impurity(y, metric),
        'n_samples': X.shape[0],
        'is_leaf': True
    }

    _grow_tree(
        tree, X, y, metric, max_depth, min_samples_split,
        min_impurity_decrease, rng
    )

    return tree

def _grow_tree(
    node: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    max_depth: int,
    min_samples_split: int,
    min_impurity_decrease: float,
    rng: Optional[np.random.RandomState]
) -> None:
    """Recursively grow the decision tree."""
    if node['is_leaf']:
        return

    n_samples = node['n_samples']
    if (max_depth <= 0 or
        n_samples < min_samples_split):
        node['is_leaf'] = True
        return

    # Find the best split
    best_split = _find_best_split(
        X, y, metric, min_impurity_decrease, rng
    )

    if best_split['gain'] <= 0:
        node['is_leaf'] = True
        return

    # Split the node
    left_mask = X[:, best_split['feature']] <= best_split['threshold']
    right_mask = ~left_mask

    node['feature'] = best_split['feature']
    node['threshold'] = best_split['threshold']
    node['impurity'] = _compute_impurity(y, metric)
    node['is_leaf'] = False

    # Create left and right children
    node['left'] = {
        'left': None,
        'right': None,
        'feature': None,
        'threshold': None,
        'impurity': _compute_impurity(y[left_mask], metric),
        'n_samples': np.sum(left_mask),
        'is_leaf': True
    }

    node['right'] = {
        'left': None,
        'right': None,
        'feature': None,
        'threshold': None,
        'impurity': _compute_impurity(y[right_mask], metric),
        'n_samples': np.sum(right_mask),
        'is_leaf': True
    }

    # Recursively grow the children
    _grow_tree(
        node['left'], X[left_mask], y[left_mask],
        metric, max_depth - 1, min_samples_split,
        min_impurity_decrease, rng
    )

    _grow_tree(
        node['right'], X[right_mask], y[right_mask],
        metric, max_depth - 1, min_samples_split,
        min_impurity_decrease, rng
    )

def _find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    min_impurity_decrease: float,
    rng: Optional[np.random.RandomState]
) -> Dict[str, Any]:
    """Find the best split for a node."""
    n_features = X.shape[1]
    best_split = {
        'feature': None,
        'threshold': None,
        'gain': -1
    }

    # Randomly permute features if a random state is provided
    if rng is not None:
        features = rng.permutation(n_features)
    else:
        features = np.arange(n_features)

    for feature in features:
        thresholds = _find_thresholds(X[:, feature])
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask

            if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
                continue

            gain = _compute_gain(
                y, y[left_mask], y[right_mask],
                metric
            )

            if (gain > best_split['gain'] and
                gain >= min_impurity_decrease):
                best_split = {
                    'feature': feature,
                    'threshold': threshold,
                    'gain': gain
                }

    return best_split

def _find_thresholds(values: np.ndarray) -> np.ndarray:
    """Find unique thresholds for splitting."""
    return np.unique(values)

def _compute_impurity(y: np.ndarray, metric: Union[str, Callable]) -> float:
    """Compute the impurity of a node."""
    if callable(metric):
        return metric(y)
    elif metric == 'gini':
        return _compute_gini_impurity(y)
    elif metric == 'entropy':
        return _compute_entropy_impurity(y)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _compute_gini_impurity(y: np.ndarray) -> float:
    """Compute the Gini impurity."""
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)

def _compute_entropy_impurity(y: np.ndarray) -> float:
    """Compute the entropy impurity."""
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def _compute_gain(
    y: np.ndarray,
    y_left: np.ndarray,
    y_right: np.ndarray,
    metric: Union[str, Callable]
) -> float:
    """Compute the information gain for a split."""
    n = len(y)
    n_left, n_right = len(y_left), len(y_right)

    impurity_parent = _compute_impurity(y, metric)
    impurity_left = _compute_impurity(y_left, metric)
    impurity_right = _compute_impurity(y_right, metric)

    gain = impurity_parent - (
        (n_left / n) * impurity_left +
        (n_right / n) * impurity_right
    )

    return gain

def _compute_metrics(
    tree: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, float]:
    """Compute metrics for the splitting strategy."""
    # Example metric: tree depth
    max_depth = _compute_max_depth(tree)
    return {
        'max_depth': max_depth
    }

def _compute_max_depth(node: Dict[str, Any]) -> int:
    """Compute the maximum depth of a tree."""
    if node['is_leaf']:
        return 0
    left_depth = _compute_max_depth(node['left'])
    right_depth = _compute_max_depth(node['right'])
    return 1 + max(left_depth, right_depth)

################################################################################
# pruning
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def pruning_fit(
    tree: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: Union[str, Callable] = "mse",
    min_samples_leaf: int = 1,
    max_depth: Optional[int] = None,
    alpha: float = 0.01,
    normalize: str = "none",
    custom_metric: Optional[Callable] = None,
    tol: float = 1e-4,
    max_iter: int = 100
) -> Dict[str, Any]:
    """
    Perform pruning on a decision tree using cost complexity pruning.

    Parameters:
    -----------
    tree : Any
        The decision tree to be pruned.
    X_val : np.ndarray
        Validation features.
    y_val : np.ndarray
        Validation targets.
    metric : Union[str, Callable], optional
        Metric to use for pruning. Can be "mse", "mae", "r2", or a custom callable.
    min_samples_leaf : int, optional
        Minimum number of samples required to be at a leaf node.
    max_depth : Optional[int], optional
        Maximum depth of the tree. If None, nodes are expanded until all leaves are pure.
    alpha : float, optional
        Complexity parameter for pruning.
    normalize : str, optional
        Normalization method. Can be "none", "standard", "minmax", or "robust".
    custom_metric : Optional[Callable], optional
        Custom metric function if not using built-in metrics.
    tol : float, optional
        Tolerance for stopping criteria.
    max_iter : int, optional
        Maximum number of iterations.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the pruned tree, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X_val, y_val)

    # Normalize data if required
    X_norm = _normalize_data(X_val, normalize)

    # Choose metric function
    if isinstance(metric, str):
        metric_func = _get_metric_function(metric)
    else:
        metric_func = metric

    # Perform pruning
    pruned_tree, metrics, warnings = _prune_tree(
        tree,
        X_norm,
        y_val,
        metric_func,
        min_samples_leaf,
        max_depth,
        alpha,
        tol,
        max_iter
    )

    return {
        "result": pruned_tree,
        "metrics": metrics,
        "params_used": {
            "min_samples_leaf": min_samples_leaf,
            "max_depth": max_depth,
            "alpha": alpha,
            "normalize": normalize
        },
        "warnings": warnings
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("X and y must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("X and y must not contain infinite values.")

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize data based on the specified method."""
    if method == "none":
        return X
    elif method == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _get_metric_function(metric: str) -> Callable:
    """Get the metric function based on the specified metric."""
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
    return 1 - (ss_res / (ss_tot + 1e-8))

def _prune_tree(
    tree: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric_func: Callable,
    min_samples_leaf: int,
    max_depth: Optional[int],
    alpha: float,
    tol: float,
    max_iter: int
) -> tuple:
    """Perform pruning on the decision tree."""
    warnings = []
    metrics = {}

    # Initial metric
    y_pred = _predict_tree(tree, X_val)
    initial_metric = metric_func(y_val, y_pred)
    metrics["initial"] = initial_metric

    # Pruning loop
    for i in range(max_iter):
        # Get current tree structure
        current_tree = _get_current_tree(tree)

        # Calculate cost complexity
        cost_complexity = _calculate_cost_complexity(current_tree, alpha)

        # Check stopping criteria
        if np.abs(cost_complexity - initial_metric) < tol:
            warnings.append("Pruning stopped due to tolerance.")
            break

        # Update tree
        tree = _update_tree(tree, current_tree, min_samples_leaf, max_depth)

        # Update predictions and metric
        y_pred = _predict_tree(tree, X_val)
        current_metric = metric_func(y_val, y_pred)
        metrics[f"iteration_{i}"] = current_metric

    return tree, metrics, warnings

def _predict_tree(tree: Any, X: np.ndarray) -> np.ndarray:
    """Predict using the decision tree."""
    # Placeholder for actual prediction logic
    return np.zeros(X.shape[0])

def _get_current_tree(tree: Any) -> Any:
    """Get the current tree structure."""
    # Placeholder for actual tree structure retrieval
    return tree

def _calculate_cost_complexity(tree: Any, alpha: float) -> float:
    """Calculate the cost complexity of the tree."""
    # Placeholder for actual cost complexity calculation
    return 0.0

def _update_tree(
    tree: Any,
    current_tree: Any,
    min_samples_leaf: int,
    max_depth: Optional[int]
) -> Any:
    """Update the tree based on pruning criteria."""
    # Placeholder for actual tree update logic
    return tree

################################################################################
# max_depth
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data."""
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

def _compute_criterion(
    X: np.ndarray,
    y: np.ndarray,
    criterion: str = "gini",
    custom_criterion: Optional[Callable] = None
) -> float:
    """Compute the splitting criterion."""
    if custom_criterion is not None:
        return custom_criterion(X, y)

    if criterion == "gini":
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1 - np.sum(probs ** 2)
    elif criterion == "entropy":
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        probs = np.where(probs == 0, 1, probs)
        return -np.sum(probs * np.log2(probs))
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

def _find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: Optional[np.ndarray] = None,
    max_features: Optional[int] = None
) -> Dict[str, Any]:
    """Find the best split for a given node."""
    if feature_indices is None:
        feature_indices = np.arange(X.shape[1])
    elif max_features is not None and len(feature_indices) > max_features:
        feature_indices = np.random.choice(
            feature_indices, size=max_features, replace=False
        )

    best_split = {"feature": None, "threshold": None, "value": float("inf")}

    for feature in feature_indices:
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            left_value = _compute_criterion(X[left_mask], y[left_mask])
            right_value = _compute_criterion(X[right_mask], y[right_mask])

            split_value = (np.sum(left_mask) * left_value +
                          np.sum(right_mask) * right_value) / len(y)

            if split_value < best_split["value"]:
                best_split.update({
                    "feature": feature,
                    "threshold": threshold,
                    "value": split_value
                })

    return best_split

def _build_tree(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int,
    min_samples_split: int = 2,
    current_depth: int = 0,
    feature_indices: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Recursively build the decision tree."""
    n_samples = X.shape[0]

    if (current_depth >= max_depth or
        n_samples < min_samples_split or
        len(np.unique(y)) == 1):
        return {
            "is_leaf": True,
            "value": np.bincount(y.astype(int)).argmax()
        }

    best_split = _find_best_split(X, y, feature_indices)

    if best_split["feature"] is None:
        return {
            "is_leaf": True,
            "value": np.bincount(y.astype(int)).argmax()
        }

    left_mask = X[:, best_split["feature"]] <= best_split["threshold"]
    right_mask = ~left_mask

    left_subtree = _build_tree(
        X[left_mask],
        y[left_mask],
        max_depth,
        min_samples_split,
        current_depth + 1
    )

    right_subtree = _build_tree(
        X[right_mask],
        y[right_mask],
        max_depth,
        min_samples_split,
        current_depth + 1
    )

    return {
        "is_leaf": False,
        "feature": best_split["feature"],
        "threshold": best_split["threshold"],
        "left": left_subtree,
        "right": right_subtree
    }

def max_depth_fit(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int = 5,
    criterion: str = "gini",
    custom_criterion: Optional[Callable] = None,
    max_features: Optional[int] = None,
    min_samples_split: int = 2
) -> Dict[str, Any]:
    """
    Fit a decision tree with maximum depth constraint.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    max_depth : int, optional
        Maximum depth of the tree (default: 5)
    criterion : str or callable, optional
        Splitting criterion ("gini" or "entropy") (default: "gini")
    custom_criterion : callable, optional
        Custom splitting criterion function
    max_features : int or None, optional
        Number of features to consider for splitting (default: None)
    min_samples_split : int, optional
        Minimum number of samples required to split a node (default: 2)

    Returns:
    --------
    dict
        Dictionary containing the fitted tree structure and metadata

    Example:
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([0, 1, 0])
    >>> result = max_depth_fit(X, y, max_depth=2)
    """
    _validate_inputs(X, y)

    if max_depth <= 0:
        raise ValueError("max_depth must be a positive integer")

    if min_samples_split <= 1:
        raise ValueError("min_samples_split must be at least 2")

    if max_features is not None and (max_features <= 0 or max_features > X.shape[1]):
        raise ValueError("max_features must be between 0 and n_features")

    tree = _build_tree(
        X,
        y,
        max_depth,
        min_samples_split=min_samples_split,
        feature_indices=np.arange(X.shape[1]) if max_features is None else None
    )

    return {
        "result": tree,
        "metrics": {},
        "params_used": {
            "max_depth": max_depth,
            "criterion": criterion if custom_criterion is None else "custom",
            "max_features": max_features,
            "min_samples_split": min_samples_split
        },
        "warnings": []
    }

################################################################################
# min_samples_split
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    min_samples_split: int = 2
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if not isinstance(min_samples_split, int) or min_samples_split < 2:
        raise ValueError("min_samples_split must be an integer >= 2")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")

def _compute_split_criterion(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    criterion: str = 'gini',
    custom_criterion: Optional[Callable] = None
) -> float:
    """Compute the splitting criterion for a given dataset."""
    if custom_criterion is not None:
        return custom_criterion(X, y)

    n_samples = X.shape[0]
    if criterion == 'gini':
        if y is None:
            raise ValueError("y must be provided for gini criterion")
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / n_samples
        return 1 - np.sum(probabilities ** 2)
    elif criterion == 'entropy':
        if y is None:
            raise ValueError("y must be provided for entropy criterion")
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / n_samples
        probabilities = np.where(probabilities == 0, 1, probabilities)
        return -np.sum(probabilities * np.log2(probabilities))
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

def _find_optimal_split(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    criterion: str = 'gini',
    custom_criterion: Optional[Callable] = None,
    min_samples_split: int = 2
) -> Dict[str, Any]:
    """Find the optimal split for a given dataset."""
    if X.shape[0] < min_samples_split:
        return {
            'split': None,
            'feature_index': None,
            'threshold': None,
            'left_indices': None,
            'right_indices': None
        }

    n_features = X.shape[1]
    best_criterion = _compute_split_criterion(X, y, criterion, custom_criterion)
    best_feature_index = None
    best_threshold = None

    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            left_indices = X[:, feature_index] <= threshold
            right_indices = ~left_indices

            if np.sum(left_indices) < min_samples_split or np.sum(right_indices) < min_samples_split:
                continue

            left_criterion = _compute_split_criterion(X[left_indices], y[left_indices] if y is not None else None, criterion, custom_criterion)
            right_criterion = _compute_split_criterion(X[right_indices], y[right_indices] if y is not None else None, criterion, custom_criterion)
            current_criterion = left_criterion + right_criterion

            if current_criterion < best_criterion:
                best_criterion = current_criterion
                best_feature_index = feature_index
                best_threshold = threshold

    if best_feature_index is not None:
        left_indices = X[:, best_feature_index] <= best_threshold
        right_indices = ~left_indices
    else:
        left_indices = None
        right_indices = None

    return {
        'split': best_criterion,
        'feature_index': best_feature_index,
        'threshold': best_threshold,
        'left_indices': left_indices,
        'right_indices': right_indices
    }

def min_samples_split_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    criterion: str = 'gini',
    custom_criterion: Optional[Callable] = None,
    min_samples_split: int = 2
) -> Dict[str, Any]:
    """
    Find the optimal split for a given dataset based on min_samples_split.

    Parameters:
    -----------
    X : np.ndarray
        Input features.
    y : Optional[np.ndarray]
        Target values (for supervised learning).
    criterion : str, optional
        The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain.
    custom_criterion : Optional[Callable]
        A custom function to compute the splitting criterion. If provided, overrides the `criterion` parameter.
    min_samples_split : int, optional
        The minimum number of samples required to split an internal node.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the split information and metrics.
    """
    _validate_inputs(X, y, min_samples_split)
    result = _find_optimal_split(X, y, criterion, custom_criterion, min_samples_split)

    return {
        'result': result,
        'metrics': {'split_criterion': result['split']},
        'params_used': {
            'criterion': criterion,
            'custom_criterion': custom_criterion is not None,
            'min_samples_split': min_samples_split
        },
        'warnings': []
    }

################################################################################
# min_samples_leaf
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_input(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data for min_samples_leaf."""
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

def compute_criterion(
    X: np.ndarray,
    y: np.ndarray,
    criterion: str = "mse",
    custom_criterion: Optional[Callable] = None
) -> float:
    """Compute the criterion for a given split."""
    if custom_criterion is not None:
        return custom_criterion(X, y)
    if criterion == "mse":
        return np.mean((y - np.mean(y)) ** 2)
    elif criterion == "mae":
        return np.mean(np.abs(y - np.mean(y)))
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

def find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    min_samples_leaf: int = 1,
    criterion: str = "mse",
    custom_criterion: Optional[Callable] = None
) -> Dict[str, Any]:
    """Find the best split for a given node."""
    n_samples = X.shape[0]
    if min_samples_leaf <= 0:
        raise ValueError("min_samples_leaf must be a positive integer.")
    if min_samples_leaf > n_samples:
        raise ValueError("min_samples_leaf cannot be greater than the number of samples.")

    best_criterion = compute_criterion(X, y, criterion, custom_criterion)
    best_split = None

    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask

            if np.sum(left_mask) < min_samples_leaf or np.sum(right_mask) < min_samples_leaf:
                continue

            left_criterion = compute_criterion(X[left_mask], y[left_mask], criterion, custom_criterion)
            right_criterion = compute_criterion(X[right_mask], y[right_mask], criterion, custom_criterion)
            total_criterion = left_criterion + right_criterion

            if total_criterion < best_criterion:
                best_criterion = total_criterion
                best_split = {
                    "feature": feature,
                    "threshold": threshold,
                    "left_mask": left_mask,
                    "right_mask": right_mask
                }

    return best_split

def min_samples_leaf_fit(
    X: np.ndarray,
    y: np.ndarray,
    min_samples_leaf: int = 1,
    criterion: str = "mse",
    custom_criterion: Optional[Callable] = None,
    max_depth: int = None
) -> Dict[str, Any]:
    """
    Fit a decision tree with min_samples_leaf constraint.

    Parameters:
    - X: Input features (2D array).
    - y: Target values (1D array).
    - min_samples_leaf: Minimum number of samples required to be at a leaf node.
    - criterion: Criterion to measure the quality of a split ("mse", "mae").
    - custom_criterion: Custom criterion function.
    - max_depth: Maximum depth of the tree.

    Returns:
    - A dictionary containing the tree structure, metrics, and other information.
    """
    validate_input(X, y)

    def build_tree(node_X: np.ndarray, node_y: np.ndarray, depth: int = 0) -> Dict[str, Any]:
        if max_depth is not None and depth >= max_depth:
            return {
                "is_leaf": True,
                "value": np.mean(node_y),
                "n_samples": node_X.shape[0]
            }

        best_split = find_best_split(node_X, node_y, min_samples_leaf, criterion, custom_criterion)
        if best_split is None:
            return {
                "is_leaf": True,
                "value": np.mean(node_y),
                "n_samples": node_X.shape[0]
            }

        left_node = build_tree(node_X[best_split["left_mask"]], node_y[best_split["left_mask"]], depth + 1)
        right_node = build_tree(node_X[best_split["right_mask"]], node_y[best_split["right_mask"]], depth + 1)

        return {
            "is_leaf": False,
            "feature": best_split["feature"],
            "threshold": best_split["threshold"],
            "left": left_node,
            "right": right_node,
            "n_samples": node_X.shape[0]
        }

    tree = build_tree(X, y)

    metrics = {
        "criterion": criterion,
        "min_samples_leaf": min_samples_leaf
    }

    return {
        "result": tree,
        "metrics": metrics,
        "params_used": {
            "min_samples_leaf": min_samples_leaf,
            "criterion": criterion,
            "max_depth": max_depth
        },
        "warnings": []
    }

# Example usage:
# tree = min_samples_leaf_fit(X_train, y_train, min_samples_leaf=5, criterion="mse")

################################################################################
# max_features
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if y is not None and np.any(np.isnan(y)):
        raise ValueError("y contains NaN values")

def _compute_feature_importance(X: np.ndarray, y: np.ndarray,
                              criterion: str = 'gini',
                              max_depth: int = None) -> np.ndarray:
    """Compute feature importance based on a decision tree."""
    # This is a simplified version - in practice you would implement or call an actual decision tree
    n_features = X.shape[1]
    importance = np.zeros(n_features)

    # Placeholder for actual decision tree implementation
    # In practice, you would split the data and compute importance based on criterion reduction
    for i in range(n_features):
        # Simulate feature importance calculation
        importance[i] = np.var(X[:, i])

    return importance / importance.sum()  # Normalize

def _select_features(importance: np.ndarray,
                    n_features: Optional[int] = None,
                    threshold: Optional[float] = None) -> np.ndarray:
    """Select features based on importance scores."""
    if n_features is not None and threshold is not None:
        raise ValueError("Cannot specify both n_features and threshold")

    if n_features is not None:
        idx = np.argsort(importance)[-n_features:]
    elif threshold is not None:
        idx = np.where(importance >= threshold)[0]
    else:
        raise ValueError("Must specify either n_features or threshold")

    return idx

def max_features_fit(X: np.ndarray,
                    y: Optional[np.ndarray] = None,
                    criterion: str = 'gini',
                    max_depth: int = None,
                    n_features: Optional[int] = None,
                    threshold: Optional[float] = None,
                    normalize_importance: bool = True) -> Dict[str, Any]:
    """
    Compute and select important features using decision tree criteria.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values if available
    criterion : str
        The function to measure the quality of a split. Supported criteria are "gini" for Gini impurity
        and "entropy" for information gain.
    max_depth : int, optional
        The maximum depth of the tree. If None, nodes are expanded until all leaves are pure or
        until all leaves contain less than min_samples_split samples.
    n_features : int, optional
        Number of top features to select. Either this or threshold must be specified.
    threshold : float, optional
        Minimum importance score for a feature to be selected. Either this or n_features must be specified.
    normalize_importance : bool
        Whether to normalize importance scores to sum to 1.

    Returns:
    --------
    dict
        Dictionary containing:
        - 'selected_features': array of selected feature indices
        - 'importance_scores': array of importance scores for all features
        - 'params_used': dictionary of parameters actually used
        - 'warnings': list of any warnings generated

    Example:
    --------
    >>> X = np.random.rand(100, 20)
    >>> y = np.random.randint(0, 2, 100)
    >>> result = max_features_fit(X, y, criterion='gini', n_features=5)
    """
    warnings = []

    # Validate inputs
    _validate_inputs(X, y)

    # Compute feature importance
    if y is not None:
        importance = _compute_feature_importance(X, y, criterion=criterion, max_depth=max_depth)
    else:
        # For unsupervised case
        importance = np.var(X, axis=0)
        warnings.append("Unsupervised mode: using variance as importance measure")

    if normalize_importance:
        importance = importance / np.sum(importance)

    # Select features
    selected_features = _select_features(importance, n_features=n_features, threshold=threshold)

    return {
        'result': selected_features,
        'metrics': {'importance_scores': importance},
        'params_used': {
            'criterion': criterion,
            'max_depth': max_depth,
            'n_features': n_features if n_features is not None else 'all above threshold',
            'threshold': threshold,
            'normalize_importance': normalize_importance
        },
        'warnings': warnings
    }

################################################################################
# class_weight
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> None:
    """Validate input arrays."""
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array")
    if sample_weight is not None and not isinstance(sample_weight, np.ndarray):
        raise TypeError("sample_weight must be a numpy array or None")
    if len(y) == 0:
        raise ValueError("y cannot be empty")
    if sample_weight is not None and len(y) != len(sample_weight):
        raise ValueError("y and sample_weight must have the same length")
    if np.any(np.isnan(y)):
        raise ValueError("y contains NaN values")

def _compute_class_counts(
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> Dict[int, float]:
    """Compute class counts with optional sample weights."""
    classes = np.unique(y)
    class_counts = {}

    if sample_weight is None:
        for cls in classes:
            class_counts[cls] = np.sum(y == cls)
    else:
        for cls in classes:
            class_counts[cls] = np.sum(sample_weight[y == cls])

    return class_counts

def _apply_normalization(
    class_counts: Dict[int, float],
    normalization: str = 'none',
    custom_normalization: Optional[Callable] = None
) -> Dict[int, float]:
    """Apply normalization to class weights."""
    if custom_normalization is not None:
        return {cls: custom_normalization(count) for cls, count in class_counts.items()}

    total = sum(class_counts.values())

    if normalization == 'none':
        return class_counts
    elif normalization == 'standard':
        mean = total / len(class_counts)
        std = np.sqrt(sum((count - mean) ** 2 for count in class_counts.values()) / len(class_counts))
        return {cls: (count - mean) / std for cls, count in class_counts.items()}
    elif normalization == 'minmax':
        min_val = min(class_counts.values())
        max_val = max(class_counts.values())
        return {cls: (count - min_val) / (max_val - min_val) for cls, count in class_counts.items()}
    elif normalization == 'robust':
        median = np.median(list(class_counts.values()))
        iqr = np.percentile(list(class_counts.values()), 75) - np.percentile(list(class_counts.values()), 25)
        return {cls: (count - median) / iqr for cls, count in class_counts.items()}
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _compute_class_weights(
    class_counts: Dict[int, float],
    metric: str = 'balanced',
    custom_metric: Optional[Callable] = None
) -> Dict[int, float]:
    """Compute class weights based on the specified metric."""
    if custom_metric is not None:
        return {cls: custom_metric(count) for cls, count in class_counts.items()}

    total = sum(class_counts.values())

    if metric == 'balanced':
        return {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
    elif metric == 'inverse':
        return {cls: 1.0 / count for cls, count in class_counts.items()}
    elif metric == 'sqrt_inverse':
        return {cls: 1.0 / np.sqrt(count) for cls, count in class_counts.items()}
    elif metric == 'log':
        return {cls: 1.0 / np.log(count + 1) for cls, count in class_counts.items()}
    else:
        raise ValueError(f"Unknown metric: {metric}")

def class_weight_fit(
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    normalization: str = 'none',
    metric: str = 'balanced',
    custom_normalization: Optional[Callable[[float], float]] = None,
    custom_metric: Optional[Callable[[float], float]] = None
) -> Dict[str, Union[Dict[int, float], Dict[str, str], list]]:
    """
    Compute class weights for decision trees.

    Parameters
    ----------
    y : np.ndarray
        Target values.
    sample_weight : Optional[np.ndarray], default=None
        Sample weights.
    normalization : str, default='none'
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str, default='balanced'
        Metric for computing class weights ('balanced', 'inverse', 'sqrt_inverse', 'log').
    custom_normalization : Optional[Callable], default=None
        Custom normalization function.
    custom_metric : Optional[Callable], default=None
        Custom metric function.

    Returns
    -------
    Dict[str, Union[Dict[int, float], Dict[str, str], list]]
        Dictionary containing:
        - 'result': Computed class weights
        - 'params_used': Parameters used in computation
        - 'warnings': List of warnings

    Example
    -------
    >>> y = np.array([0, 1, 1, 0, 1])
    >>> class_weight_fit(y)
    {
        'result': {0: 2.0, 1: 1.0},
        'params_used': {'normalization': 'none', 'metric': 'balanced'},
        'warnings': []
    }
    """
    _validate_inputs(y, sample_weight)

    class_counts = _compute_class_counts(y, sample_weight)
    normalized_counts = _apply_normalization(class_counts, normalization, custom_normalization)
    class_weights = _compute_class_weights(normalized_counts, metric, custom_metric)

    return {
        'result': class_weights,
        'params_used': {
            'normalization': normalization,
            'metric': metric
        },
        'warnings': []
    }

################################################################################
# random_state
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def random_state_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    criterion: str = "gini",
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: Optional[Union[int, float]] = None,
    random_state: Optional[int] = None,
    splitter: str = "best",
    max_leaf_nodes: Optional[int] = None,
    min_impurity_decrease: float = 0.0,
    class_weight: Optional[Union[Dict[int, int], str]] = None,
    ccp_alpha: float = 0.0
) -> Dict[str, Any]:
    """
    Fit a decision tree classifier or regressor with random state control.

    Parameters:
    -----------
    X : np.ndarray
        Training data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,) or (n_samples, n_outputs).
    criterion : str
        The function to measure the quality of a split. Supported criteria are "gini" for classification and
        "mse", "friedman_mse", "mae" for regression.
    max_depth : int, optional
        The maximum depth of the tree. If None, nodes are expanded until all leaves are pure or until all
        leaves contain min_samples_split samples.
    min_samples_split : int, optional
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, optional
        The minimum number of samples required to be at a leaf node.
    max_features : int, float or {"auto", "sqrt", "log2"}, optional
        The number of features to consider when looking for the best split.
    random_state : int, optional
        Controls the randomness of the estimator. Pass an int for reproducible output across multiple function calls.
    splitter : {"best", "random"}, optional
        The strategy used to choose the split at each node.
    max_leaf_nodes : int, optional
        Grow a tree with `max_leaf_nodes` in best-first fashion.
    min_impurity_decrease : float, optional
        A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
    class_weight : dict, list of lists, "balanced", default=None
        Weights associated with classes in the form ``{class_label: weight}``.
    ccp_alpha : non-negative float, optional
        Complexity parameter used for Minimal Cost-Complexity Pruning.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the fitted model and related information.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set random state for reproducibility
    if random_state is not None:
        np.random.seed(random_state)

    # Initialize the tree structure
    tree = _initialize_tree(
        X, y,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        splitter=splitter,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        class_weight=class_weight,
        ccp_alpha=ccp_alpha
    )

    # Fit the tree
    _fit_tree(tree, X, y)

    return {
        "result": tree,
        "metrics": _compute_metrics(X, y, tree),
        "params_used": {
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "random_state": random_state,
            "splitter": splitter,
            "max_leaf_nodes": max_leaf_nodes,
            "min_impurity_decrease": min_impurity_decrease,
            "class_weight": class_weight,
            "ccp_alpha": ccp_alpha
        },
        "warnings": _check_warnings(tree)
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

def _initialize_tree(
    X: np.ndarray,
    y: np.ndarray,
    *,
    criterion: str = "gini",
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: Optional[Union[int, float]] = None,
    splitter: str = "best",
    max_leaf_nodes: Optional[int] = None,
    min_impurity_decrease: float = 0.0,
    class_weight: Optional[Union[Dict[int, int], str]] = None,
    ccp_alpha: float = 0.0
) -> Dict[str, Any]:
    """Initialize the tree structure with given parameters."""
    return {
        "criterion": criterion,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "splitter": splitter,
        "max_leaf_nodes": max_leaf_nodes,
        "min_impurity_decrease": min_impurity_decrease,
        "class_weight": class_weight,
        "ccp_alpha": ccp_alpha,
        "nodes": []
    }

def _fit_tree(tree: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> None:
    """Fit the decision tree recursively."""
    root = _create_node(X, y, tree)
    tree["nodes"].append(root)
    _grow_tree(tree, root, X, y)

def _create_node(X: np.ndarray, y: np.ndarray, tree: Dict[str, Any]) -> Dict[str, Any]:
    """Create a node for the decision tree."""
    return {
        "left": None,
        "right": None,
        "feature_index": None,
        "threshold": None,
        "impurity": _compute_impurity(X, y, tree["criterion"]),
        "n_samples": X.shape[0],
        "class_distribution": _compute_class_distribution(y, tree["criterion"])
    }

def _grow_tree(tree: Dict[str, Any], node: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> None:
    """Recursively grow the decision tree."""
    if _is_leaf_node(node, tree):
        return

    best_feature, best_threshold = _find_best_split(X, y, tree)
    if best_feature is None:
        node["left"] = _create_leaf_node(X, y)
        return

    left_indices = X[:, best_feature] <= best_threshold
    right_indices = ~left_indices

    node["feature_index"] = best_feature
    node["threshold"] = best_threshold

    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]

    node["left"] = _create_node(X_left, y_left, tree)
    node["right"] = _create_node(X_right, y_right, tree)

    tree["nodes"].append(node["left"])
    tree["nodes"].append(node["right"])

    _grow_tree(tree, node["left"], X_left, y_left)
    _grow_tree(tree, node["right"], X_right, y_right)

def _is_leaf_node(node: Dict[str, Any], tree: Dict[str, Any]) -> bool:
    """Check if a node is a leaf node."""
    return (tree["max_depth"] is not None and _get_node_depth(node, tree) >= tree["max_depth"]) or \
           (node["n_samples"] < tree["min_samples_split"]) or \
           (_compute_impurity_reduction(node) < tree["min_impurity_decrease"])

def _get_node_depth(node: Dict[str, Any], tree: Dict[str, Any]) -> int:
    """Get the depth of a node in the tree."""
    if node["left"] is None and node["right"] is None:
        return 0
    left_depth = _get_node_depth(node["left"], tree) if node["left"] is not None else 0
    right_depth = _get_node_depth(node["right"], tree) if node["right"] is not None else 0
    return max(left_depth, right_depth) + 1

def _compute_impurity(X: np.ndarray, y: np.ndarray, criterion: str) -> float:
    """Compute the impurity of a node."""
    if criterion == "gini":
        return _compute_gini_impurity(y)
    elif criterion in ["mse", "friedman_mse", "mae"]:
        return _compute_regression_impurity(y, criterion)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

def _compute_gini_impurity(y: np.ndarray) -> float:
    """Compute the Gini impurity."""
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities ** 2)

def _compute_regression_impurity(y: np.ndarray, criterion: str) -> float:
    """Compute the impurity for regression."""
    if criterion == "mse":
        return np.mean((y - np.mean(y)) ** 2)
    elif criterion == "friedman_mse":
        return _compute_friedman_mse(y)
    elif criterion == "mae":
        return np.mean(np.abs(y - np.median(y)))
    else:
        raise ValueError(f"Unknown regression criterion: {criterion}")

def _compute_friedman_mse(y: np.ndarray) -> float:
    """Compute the Friedman MSE impurity."""
    return np.mean((y - np.median(y)) ** 2)

def _compute_class_distribution(y: np.ndarray, criterion: str) -> Dict[int, float]:
    """Compute the class distribution for a node."""
    if criterion in ["gini", "entropy"]:
        classes, counts = np.unique(y, return_counts=True)
        total = counts.sum()
        return {cls: count / total for cls, count in zip(classes, counts)}
    else:
        return {"value": np.mean(y)}

def _find_best_split(X: np.ndarray, y: np.ndarray, tree: Dict[str, Any]) -> tuple:
    """Find the best split for a node."""
    best_feature = None
    best_threshold = None
    best_impurity_reduction = -1

    n_features = X.shape[1]
    if tree["max_features"] is not None:
        if isinstance(tree["max_features"], int):
            n_features = min(n_features, tree["max_features"])
        elif isinstance(tree["max_features"], float):
            n_features = max(1, int(n_features * tree["max_features"]))
        elif tree["max_features"] == "auto":
            n_features = n_features
        elif tree["max_features"] == "sqrt":
            n_features = int(np.sqrt(n_features))
        elif tree["max_features"] == "log2":
            n_features = int(np.log2(n_features))
        else:
            raise ValueError(f"Unknown max_features: {tree['max_features']}")

    features = np.random.choice(n_features, size=n_features, replace=False)

    for feature in features:
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_indices = X[:, feature] <= threshold
            right_indices = ~left_indices

            if len(left_indices) == 0 or len(right_indices) == 0:
                continue

            left_impurity = _compute_impurity(X[left_indices], y[left_indices], tree["criterion"])
            right_impurity = _compute_impurity(X[right_indices], y[right_indices], tree["criterion"])
            n_left, n_right = len(left_indices), len(right_indices)
            total_impurity = (n_left * left_impurity + n_right * right_impurity) / (n_left + n_right)
            impurity_reduction = tree["nodes"][0]["impurity"] - total_impurity

            if impurity_reduction > best_impurity_reduction:
                best_impurity_reduction = impurity_reduction
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold

def _compute_impurity_reduction(node: Dict[str, Any]) -> float:
    """Compute the impurity reduction for a node."""
    if node["left"] is None or node["right"] is None:
        return 0
    left_impurity = node["left"]["impurity"]
    right_impurity = node["right"]["impurity"]
    n_left, n_right = node["left"]["n_samples"], node["right"]["n_samples"]
    total_impurity = (n_left * left_impurity + n_right * right_impurity) / (n_left + n_right)
    return node["impurity"] - total_impurity

def _create_leaf_node(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Create a leaf node."""
    return {
        "left": None,
        "right": None,
        "feature_index": None,
        "threshold": None,
        "impurity": _compute_impurity(X, y, "gini"),
        "n_samples": X.shape[0],
        "class_distribution": _compute_class_distribution(y, "gini")
    }

def _compute_metrics(X: np.ndarray, y: np.ndarray, tree: Dict[str, Any]) -> Dict[str, float]:
    """Compute metrics for the fitted tree."""
    y_pred = _predict(tree, X)
    if len(y.shape) == 1:
        return {"accuracy": np.mean(y_pred == y)}
    else:
        return {"mse": np.mean((y_pred - y) ** 2)}

def _predict(tree: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Predict using the fitted tree."""
    y_pred = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        y_pred[i] = _predict_sample(tree, X[i])
    return y_pred

def _predict_sample(tree: Dict[str, Any], sample: np.ndarray) -> float:
    """Predict a single sample using the tree."""
    node = tree["nodes"][0]
    while node["left"] is not None or node["right"] is not None:
        if sample[node["feature_index"]] <= node["threshold"]:
            node = node["left"]
        else:
            node = node["right"]
    return _get_leaf_value(node)

def _get_leaf_value(node: Dict[str, Any]) -> float:
    """Get the value of a leaf node."""
    if "class_distribution" in node:
        return max(node["class_distribution"], key=node["class_distribution"].get)
    else:
        return node["value"]

def _check_warnings(tree: Dict[str, Any]) -> list:
    """Check for warnings during tree fitting."""
    warnings = []
    if any(node["n_samples"] < 2 for node in tree["nodes"]):
        warnings.append("Some leaves have fewer than min_samples_leaf samples.")
    return warnings

################################################################################
# overfitting
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None
) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X_train, np.ndarray) or not isinstance(y_train, np.ndarray):
        raise TypeError("X_train and y_train must be numpy arrays")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train must have the same number of samples")
    if X_val is not None or y_val is not None:
        if (X_val is None) != (y_val is None):
            raise ValueError("Both X_val and y_val must be provided or omitted together")
        if not isinstance(X_val, np.ndarray) or not isinstance(y_val, np.ndarray):
            raise TypeError("X_val and y_val must be numpy arrays")
        if X_val.shape[0] != y_val.shape[0]:
            raise ValueError("X_val and y_val must have the same number of samples")
        if X_train.shape[1] != X_val.shape[1]:
            raise ValueError("X_train and X_val must have the same number of features")

def _normalize_data(
    data: np.ndarray,
    normalization: str = "standard",
    custom_normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize data based on specified method or custom function."""
    if custom_normalizer is not None:
        return custom_normalizer(data)

    data = np.array(data, dtype=np.float64)
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

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "mse",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> float:
    """Compute specified metric between true and predicted values."""
    if custom_metric is not None:
        return custom_metric(y_true, y_pred)

    if metric == "mse":
        return np.mean((y_true - y_pred) ** 2)
    elif metric == "mae":
        return np.mean(np.abs(y_true - y_pred))
    elif metric == "r2":
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    elif metric == "logloss":
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _calculate_overfitting(
    tree_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    max_depth: int = 5
) -> Dict[str, Any]:
    """Calculate overfitting metrics for a decision tree model."""
    train_scores = []
    val_scores = []

    for depth in range(1, max_depth + 1):
        tree_model.set_params(max_depth=depth)
        tree_model.fit(X_train, y_train)

        train_score = tree_model.score(X_train, y_train)
        train_scores.append(train_score)

        if X_val is not None and y_val is not None:
            val_score = tree_model.score(X_val, y_val)
        else:
            val_score = train_score  # Use training score if validation data not provided
        val_scores.append(val_score)

    return {
        "train_scores": train_scores,
        "val_scores": val_scores,
        "max_depths": list(range(1, max_depth + 1))
    }

def overfitting_fit(
    tree_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    normalization: str = "standard",
    metric: str = "mse",
    max_depth: int = 5,
    custom_normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute overfitting metrics for a decision tree model.

    Parameters:
    -----------
    tree_model : sklearn.tree.DecisionTreeClassifier or DecisionTreeRegressor
        The decision tree model to evaluate.
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training targets.
    X_val : Optional[np.ndarray]
        Validation features. If None, training data will be used for validation.
    y_val : Optional[np.ndarray]
        Validation targets. If None, training data will be used for validation.
    normalization : str or callable
        Normalization method ('none', 'standard', 'minmax', 'robust') or custom function.
    metric : str or callable
        Evaluation metric ('mse', 'mae', 'r2', 'logloss') or custom function.
    max_depth : int
        Maximum tree depth to evaluate.
    custom_normalizer : Optional[Callable]
        Custom normalization function.
    custom_metric : Optional[Callable]
        Custom metric function.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing overfitting results and metrics.
    """
    # Validate inputs
    _validate_inputs(X_train, y_train, X_val, y_val)

    # Normalize data
    X_train_norm = _normalize_data(X_train, normalization, custom_normalizer)
    if X_val is not None:
        X_val_norm = _normalize_data(X_val, normalization, custom_normalizer)
    else:
        X_val_norm = None

    # Calculate overfitting metrics
    results = _calculate_overfitting(
        tree_model,
        X_train_norm,
        y_train,
        X_val_norm,
        y_val if y_val is not None else None,
        max_depth
    )

    # Compute final metrics
    train_score = results["train_scores"][-1]
    val_score = results["val_scores"][-1]

    metrics = {
        "final_train_metric": _compute_metric(y_train, tree_model.predict(X_train_norm), metric, custom_metric),
        "final_val_metric": _compute_metric(y_val if y_val is not None else y_train,
                                          tree_model.predict(X_val_norm if X_val_norm is not None else X_train_norm),
                                          metric, custom_metric) if y_val is not None else train_score,
        "overfitting_gap": abs(train_score - val_score)
    }

    return {
        "result": results,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization if custom_normalizer is None else "custom",
            "metric": metric if custom_metric is None else "custom",
            "max_depth": max_depth
        },
        "warnings": []
    }

################################################################################
# underfitting
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def underfitting_fit(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int = 3,
    min_samples_split: int = 2,
    metric: Union[str, Callable] = 'mse',
    normalizer: Optional[Callable] = None,
    solver: str = 'exhaustive',
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Fit a decision tree with controlled underfitting.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    max_depth : int, optional
        Maximum depth of the tree. Higher values may reduce underfitting.
    min_samples_split : int, optional
        Minimum number of samples required to split a node.
    metric : str or callable, optional
        Metric to evaluate splits. Options: 'mse', 'mae', 'gini', or custom callable.
    normalizer : callable, optional
        Function to normalize features. If None, no normalization is applied.
    solver : str, optional
        Solver for finding optimal splits. Options: 'exhaustive', 'random'.
    random_state : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - 'result': Fitted tree structure
        - 'metrics': Evaluation metrics
        - 'params_used': Parameters used during fitting
        - 'warnings': Any warnings encountered

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = underfitting_fit(X, y, max_depth=2, metric='mse')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Initialize tree structure
    tree = {'max_depth': max_depth, 'min_samples_split': min_samples_split}

    # Build tree with controlled underfitting
    _build_tree(
        X_normalized, y,
        tree=tree,
        current_depth=0,
        metric=metric,
        solver=solver,
        random_state=random_state
    )

    # Calculate metrics
    y_pred = _predict(tree, X_normalized)
    metrics = _calculate_metrics(y, y_pred, metric)

    return {
        'result': tree,
        'metrics': metrics,
        'params_used': {
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'metric': metric,
            'normalizer': normalizer.__name__ if normalizer else None,
            'solver': solver
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
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply feature normalization if specified."""
    if normalizer is not None:
        return normalizer(X)
    return X

def _build_tree(
    X: np.ndarray,
    y: np.ndarray,
    tree: Dict[str, Any],
    current_depth: int,
    metric: Union[str, Callable],
    solver: str,
    random_state: Optional[int]
) -> None:
    """Recursively build the decision tree."""
    if current_depth >= tree['max_depth']:
        return

    # Find best split
    best_split = _find_best_split(
        X, y,
        min_samples=tree['min_samples_split'],
        metric=metric,
        solver=solver,
        random_state=random_state
    )

    if best_split is None:
        return

    # Split the data and recursively build subtrees
    left_mask = X[:, best_split['feature']] <= best_split['threshold']
    right_mask = ~left_mask

    if np.sum(left_mask) >= tree['min_samples_split']:
        tree['left'] = {}
        _build_tree(
            X[left_mask], y[left_mask],
            tree['left'],
            current_depth + 1,
            metric,
            solver,
            random_state
        )

    if np.sum(right_mask) >= tree['min_samples_split']:
        tree['right'] = {}
        _build_tree(
            X[right_mask], y[right_mask],
            tree['right'],
            current_depth + 1,
            metric,
            solver,
            random_state
        )

def _find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    min_samples: int,
    metric: Union[str, Callable],
    solver: str,
    random_state: Optional[int]
) -> Optional[Dict[str, Any]]:
    """Find the best feature and threshold to split on."""
    if solver == 'exhaustive':
        return _find_best_split_exhaustive(X, y, min_samples, metric)
    elif solver == 'random':
        return _find_best_split_random(X, y, min_samples, metric, random_state)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _find_best_split_exhaustive(
    X: np.ndarray,
    y: np.ndarray,
    min_samples: int,
    metric: Union[str, Callable]
) -> Optional[Dict[str, Any]]:
    """Exhaustively search for the best split."""
    best_score = -np.inf
    best_split = None

    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask

            if np.sum(left_mask) < min_samples or np.sum(right_mask) < min_samples:
                continue

            score = _evaluate_split(y[left_mask], y[right_mask], metric)
            if score > best_score:
                best_score = score
                best_split = {
                    'feature': feature,
                    'threshold': threshold,
                    'score': score
                }

    return best_split

def _find_best_split_random(
    X: np.ndarray,
    y: np.ndarray,
    min_samples: int,
    metric: Union[str, Callable],
    random_state: Optional[int]
) -> Optional[Dict[str, Any]]:
    """Randomly search for the best split."""
    np.random.seed(random_state)
    feature = np.random.choice(X.shape[1])
    threshold = np.random.choice(np.unique(X[:, feature]))

    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask

    if np.sum(left_mask) >= min_samples and np.sum(right_mask) >= min_samples:
        score = _evaluate_split(y[left_mask], y[right_mask], metric)
        return {
            'feature': feature,
            'threshold': threshold,
            'score': score
        }

    return None

def _evaluate_split(
    y_left: np.ndarray,
    y_right: np.ndarray,
    metric: Union[str, Callable]
) -> float:
    """Evaluate the quality of a split."""
    if callable(metric):
        return metric(y_left, y_right)
    elif metric == 'mse':
        return -np.mean((y_left - np.mean(y_left))**2 + (y_right - np.mean(y_right))**2)
    elif metric == 'mae':
        return -(np.mean(np.abs(y_left - np.median(y_left))) + np.mean(np.abs(y_right - np.median(y_right))))
    elif metric == 'gini':
        return _gini_impurity(y_left, y_right)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _gini_impurity(y_left: np.ndarray, y_right: np.ndarray) -> float:
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

def _predict(tree: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Predict using the fitted tree."""
    y_pred = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        node = tree
        while 'left' in node or 'right' in node:
            feature = node['best_split']['feature']
            threshold = node['best_split']['threshold']

            if X[i, feature] <= threshold:
                node = node.get('left', {})
            else:
                node = node.get('right', {})

        y_pred[i] = node['value']

    return y_pred

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    metrics = {}

    if callable(metric):
        metrics['custom'] = metric(y_true, y_pred)
    else:
        if metric == 'mse':
            metrics['mse'] = np.mean((y_true - y_pred)**2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred)**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            metrics['r2'] = 1 - (ss_res / ss_tot)

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
    metric: str = 'gini',
    max_depth: int = 3,
    min_samples_split: int = 2,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Compute feature importance from a decision tree.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize features. Default is identity.
    metric : str
        Impurity criterion: 'gini' or 'entropy'.
    max_depth : int
        Maximum depth of the tree.
    min_samples_split : int
        Minimum number of samples required to split a node.
    random_state : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict]]
        Dictionary containing:
        - 'result': array of feature importances
        - 'metrics': dictionary of computed metrics
        - 'params_used': dictionary of parameters used
        - 'warnings': list of warnings

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = feature_importance_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if needed
    X_normalized = normalizer(X)

    # Initialize random state
    rng = np.random.RandomState(random_state)

    # Build the decision tree
    tree = _build_decision_tree(
        X_normalized, y,
        metric=metric,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        rng=rng
    )

    # Compute feature importances
    importances = _compute_feature_importance(tree)

    # Prepare output
    result = {
        'result': importances,
        'metrics': {'metric_used': metric},
        'params_used': {
            'max_depth': max_depth,
            'min_samples_split': min_samples_split
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

def _build_decision_tree(
    X: np.ndarray,
    y: np.ndarray,
    *,
    metric: str = 'gini',
    max_depth: int = 3,
    min_samples_split: int = 2,
    rng: np.random.RandomState
) -> Dict:
    """Build a decision tree recursively."""
    if len(y) < min_samples_split or max_depth == 0:
        return _create_leaf_node(y)

    best_feature, best_threshold = _find_best_split(X, y, metric, rng)

    if best_feature is None:
        return _create_leaf_node(y)

    left_indices = X[:, best_feature] <= best_threshold
    right_indices = ~left_indices

    left_subtree = _build_decision_tree(
        X[left_indices],
        y[left_indices],
        metric=metric,
        max_depth=max_depth - 1,
        min_samples_split=min_samples_split,
        rng=rng
    )

    right_subtree = _build_decision_tree(
        X[right_indices],
        y[right_indices],
        metric=metric,
        max_depth=max_depth - 1,
        min_samples_split=min_samples_split,
        rng=rng
    )

    return {
        'feature': best_feature,
        'threshold': best_threshold,
        'left': left_subtree,
        'right': right_subtree
    }

def _find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    metric: str,
    rng: np.random.RandomState
) -> tuple:
    """Find the best feature and threshold to split on."""
    n_features = X.shape[1]
    best_score = -np.inf
    best_feature = None
    best_threshold = None

    for feature in rng.permutation(n_features):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            score = _compute_split_score(X[:, feature], y, threshold, metric)
            if score > best_score:
                best_score = score
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold

def _compute_split_score(
    feature: np.ndarray,
    y: np.ndarray,
    threshold: float,
    metric: str
) -> float:
    """Compute the score for a given split."""
    left_indices = feature <= threshold
    right_indices = ~left_indices

    if len(left_indices) == 0 or len(right_indices) == 0:
        return -np.inf

    if metric == 'gini':
        score = _gini_impurity(y[left_indices], y[right_indices])
    elif metric == 'entropy':
        score = _information_gain(y[left_indices], y[right_indices])
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return score

def _gini_impurity(left_y: np.ndarray, right_y: np.ndarray) -> float:
    """Compute Gini impurity for a split."""
    def gini(y):
        if len(y) == 0:
            return 0
        p = np.bincount(y) / len(y)
        return 1 - np.sum(p ** 2)

    left_gini = gini(left_y)
    right_gini = gini(right_y)
    n_left, n_right = len(left_y), len(right_y)
    total = (n_left * left_gini + n_right * right_gini) / (n_left + n_right)
    return total

def _information_gain(left_y: np.ndarray, right_y: np.ndarray) -> float:
    """Compute information gain for a split."""
    def entropy(y):
        if len(y) == 0:
            return 0
        p = np.bincount(y) / len(y)
        return -np.sum(p * np.log2(p + 1e-9))

    parent_entropy = entropy(np.concatenate([left_y, right_y]))
    n_left, n_right = len(left_y), len(right_y)
    child_entropy = (n_left * entropy(left_y) + n_right * entropy(right_y)) / (n_left + n_right)
    return parent_entropy - child_entropy

def _create_leaf_node(y: np.ndarray) -> Dict:
    """Create a leaf node with the most common class."""
    return {'class': np.bincount(y).argmax()}

def _compute_feature_importance(tree: Dict) -> np.ndarray:
    """Compute feature importances from a decision tree."""
    if 'feature' not in tree:
        return np.zeros(1)

    left_importance = _compute_feature_importance(tree['left'])
    right_importance = _compute_feature_importance(tree['right'])

    current_importance = np.zeros_like(left_importance)
    current_importance[tree['feature']] += 1

    return current_importance + left_importance + right_importance

################################################################################
# decision_boundary
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input data contains infinite values")

def _normalize_data(X: np.ndarray, y: np.ndarray, normalization: str) -> tuple:
    """Normalize input data based on specified method."""
    if normalization == "none":
        return X, y
    elif normalization == "standard":
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_normalized = (X - X_mean) / X_std
        return X_normalized, y
    elif normalization == "minmax":
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_normalized = (X - X_min) / (X_max - X_min)
        return X_normalized, y
    elif normalization == "robust":
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_normalized = (X - X_median) / X_iqr
        return X_normalized, y
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _compute_decision_boundary(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    distance: str = "euclidean",
    solver: str = "closed_form"
) -> Dict:
    """Compute decision boundary based on specified parameters."""
    # Placeholder for actual implementation
    result = {"boundary": np.zeros(X.shape[1])}
    metrics = {}
    params_used = {
        "normalization": "none",
        "metric": metric,
        "distance": distance,
        "solver": solver
    }
    warnings = []

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

def decision_boundary_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalization: str = "none",
    metric: Union[str, Callable] = "mse",
    distance: str = "euclidean",
    solver: str = "closed_form"
) -> Dict:
    """
    Compute decision boundary for classification or regression.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    normalization : str, optional
        Normalization method: "none", "standard", "minmax", or "robust"
    metric : str or callable, optional
        Metric to optimize: "mse", "mae", "r2", or custom callable
    distance : str, optional
        Distance metric: "euclidean", "manhattan", "cosine", or "minkowski"
    solver : str, optional
        Solver method: "closed_form", "gradient_descent", or "newton"

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings

    Example
    -------
    >>> X = np.random.rand(100, 2)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = decision_boundary_fit(X, y, normalization="standard", metric="mse")
    """
    _validate_inputs(X, y)
    X_normalized, y_normalized = _normalize_data(X, y, normalization)

    return _compute_decision_boundary(
        X_normalized,
        y_normalized,
        metric=metric,
        distance=distance,
        solver=solver
    )

################################################################################
# ensemble_methods
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def ensemble_methods_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    base_estimator: Optional[Callable] = None,
    n_estimators: int = 100,
    criterion: str = 'gini',
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: Union[int, float, str] = 'auto',
    bootstrap: bool = True,
    oob_score: bool = False,
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
    verbose: int = 0
) -> Dict[str, Any]:
    """
    Fit an ensemble of decision trees.

    Parameters:
    -----------
    X : np.ndarray
        Training data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    base_estimator : Optional[Callable], default=None
        The base estimator to fit on random subsets of the data.
    n_estimators : int, default=100
        The number of trees in the forest.
    criterion : str, default='gini'
        The function to measure the quality of a split.
    max_depth : Optional[int], default=None
        The maximum depth of the tree.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    max_features : Union[int, float, str], default='auto'
        The number of features to consider when looking for the best split.
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.
    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization accuracy.
    n_jobs : Optional[int], default=None
        The number of jobs to run in parallel.
    random_state : Optional[int], default=None
        Controls both the randomness of the bootstrapping of the samples and the sampling of the features.
    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the fitted ensemble model and related information.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Initialize ensemble parameters
    params_used = {
        'base_estimator': base_estimator,
        'n_estimators': n_estimators,
        'criterion': criterion,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'bootstrap': bootstrap,
        'oob_score': oob_score,
        'n_jobs': n_jobs,
        'random_state': random_state,
        'verbose': verbose
    }

    # Initialize ensemble model
    ensemble_model = _initialize_ensemble(X.shape[1], params_used)

    # Fit the ensemble
    result = _fit_ensemble(ensemble_model, X, y, params_used)

    # Calculate metrics
    metrics = _calculate_metrics(result, y)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
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
        raise ValueError("X and y must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("X and y must not contain infinite values.")

def _initialize_ensemble(n_features: int, params_used: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize the ensemble model."""
    return {
        'estimators': [],
        'n_features': n_features,
        'params_used': params_used
    }

def _fit_ensemble(ensemble_model: Dict[str, Any], X: np.ndarray, y: np.ndarray, params_used: Dict[str, Any]) -> Dict[str, Any]:
    """Fit the ensemble of decision trees."""
    for _ in range(params_used['n_estimators']):
        estimator = _fit_single_tree(X, y, params_used)
        ensemble_model['estimators'].append(estimator)
    return ensemble_model

def _fit_single_tree(X: np.ndarray, y: np.ndarray, params_used: Dict[str, Any]) -> Dict[str, Any]:
    """Fit a single decision tree."""
    # Placeholder for actual tree fitting logic
    return {
        'tree': None,
        'params_used': params_used
    }

def _calculate_metrics(result: Dict[str, Any], y: np.ndarray) -> Dict[str, float]:
    """Calculate metrics for the ensemble model."""
    # Placeholder for actual metric calculation logic
    return {
        'accuracy': 0.0,
        'oob_score': None if not result['params_used']['oob_score'] else 0.0
    }

# Example usage:
# ensemble_result = ensemble_methods_fit(X_train, y_train)

################################################################################
# random_forest
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def random_forest_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: Union[int, float] = 'auto',
    bootstrap: bool = True,
    criterion: str = 'gini',
    metric: Callable[[np.ndarray, np.ndarray], float] = None,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Fit a random forest model to the given data.

    Parameters:
    -----------
    X : np.ndarray
        Training data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    n_estimators : int, optional
        The number of trees in the forest (default=100).
    max_depth : int, optional
        The maximum depth of the tree (default=None).
    min_samples_split : int, optional
        The minimum number of samples required to split an internal node (default=2).
    min_samples_leaf : int, optional
        The minimum number of samples required to be at a leaf node (default=1).
    max_features : int or float, optional
        The number of features to consider when looking for the best split (default='auto').
    bootstrap : bool, optional
        Whether bootstrap samples are used when building trees (default=True).
    criterion : str, optional
        The function to measure the quality of a split (default='gini').
    metric : Callable, optional
        A custom metric function to evaluate the model (default=None).
    random_state : int, optional
        Controls both the randomness of the bootstrapping of the samples and the splitting of the nodes (default=None).

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the fitted model, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Initialize random forest parameters
    params_used = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'bootstrap': bootstrap,
        'criterion': criterion,
        'random_state': random_state
    }

    # Initialize warnings
    warnings = []

    # Check if max_features is valid
    if isinstance(max_features, str):
        if max_features == 'auto':
            max_features = int(np.sqrt(X.shape[1]))
        elif max_features == 'sqrt':
            max_features = int(np.sqrt(X.shape[1]))
        elif max_features == 'log2':
            max_features = int(np.log2(X.shape[1]))
        else:
            raise ValueError("max_features must be 'auto', 'sqrt', 'log2' or an integer")

    # Initialize the random forest
    trees = []
    for i in range(n_estimators):
        tree = _fit_single_tree(
            X, y,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            criterion=criterion,
            random_state=random_state if random_state is not None else np.random.randint(0, 1000)
        )
        trees.append(tree)

    # Calculate metrics
    metrics = {}
    if metric is not None:
        y_pred = _predict(trees, X)
        metrics['custom_metric'] = metric(y, y_pred)

    # Return the result
    return {
        'result': trees,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate the input data.

    Parameters:
    -----------
    X : np.ndarray
        Training data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).

    Raises:
    -------
    ValueError
        If the inputs are invalid.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("X and y must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("X and y must not contain infinite values.")

def _fit_single_tree(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: Optional[int],
    min_samples_split: int,
    min_samples_leaf: int,
    max_features: int,
    bootstrap: bool,
    criterion: str,
    random_state: int
) -> Dict[str, Any]:
    """
    Fit a single decision tree.

    Parameters:
    -----------
    X : np.ndarray
        Training data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    max_depth : int, optional
        The maximum depth of the tree.
    min_samples_split : int
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int
        The minimum number of samples required to be at a leaf node.
    max_features : int
        The number of features to consider when looking for the best split.
    bootstrap : bool
        Whether bootstrap samples are used when building trees.
    criterion : str
        The function to measure the quality of a split.
    random_state : int
        Controls both the randomness of the bootstrapping of the samples and the splitting of the nodes.

    Returns:
    --------
    Dict[str, Any]
        A dictionary representing the fitted tree.
    """
    np.random.seed(random_state)
    if bootstrap:
        indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
        X_sample = X[indices]
        y_sample = y[indices]
    else:
        X_sample = X
        y_sample = y

    tree = _build_tree(
        X_sample, y_sample,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion
    )

    return tree

def _build_tree(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: Optional[int],
    min_samples_split: int,
    min_samples_leaf: int,
    max_features: int,
    criterion: str
) -> Dict[str, Any]:
    """
    Recursively build a decision tree.

    Parameters:
    -----------
    X : np.ndarray
        Training data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    max_depth : int, optional
        The maximum depth of the tree.
    min_samples_split : int
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int
        The minimum number of samples required to be at a leaf node.
    max_features : int
        The number of features to consider when looking for the best split.
    criterion : str
        The function to measure the quality of a split.

    Returns:
    --------
    Dict[str, Any]
        A dictionary representing the built tree.
    """
    n_samples, n_features = X.shape

    # Check if splitting is needed
    if (max_depth is not None and max_depth <= 0) or \
       n_samples < min_samples_split or \
       len(np.unique(y)) == 1:
        return {'leaf': True, 'value': np.mean(y)}

    # Find the best split
    best_split = _find_best_split(X, y, max_features, criterion)

    if best_split is None:
        return {'leaf': True, 'value': np.mean(y)}

    # Split the data
    left_indices = X[:, best_split['feature']] <= best_split['threshold']
    right_indices = ~left_indices

    # Recursively build left and right subtrees
    left_subtree = _build_tree(
        X[left_indices], y[left_indices],
        max_depth=max_depth - 1 if max_depth is not None else None,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion
    )

    right_subtree = _build_tree(
        X[right_indices], y[right_indices],
        max_depth=max_depth - 1 if max_depth is not None else None,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion
    )

    return {
        'leaf': False,
        'feature': best_split['feature'],
        'threshold': best_split['threshold'],
        'left': left_subtree,
        'right': right_subtree
    }

def _find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    max_features: int,
    criterion: str
) -> Optional[Dict[str, Any]]:
    """
    Find the best split for a node.

    Parameters:
    -----------
    X : np.ndarray
        Training data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    max_features : int
        The number of features to consider when looking for the best split.
    criterion : str
        The function to measure the quality of a split.

    Returns:
    --------
    Optional[Dict[str, Any]]
        A dictionary representing the best split or None if no split is found.
    """
    n_samples, n_features = X.shape
    best_split = None
    best_value = -np.inf

    # Randomly select features to consider
    feature_indices = np.random.choice(n_features, max_features, replace=False)

    for feature in feature_indices:
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_indices = X[:, feature] <= threshold
            right_indices = ~left_indices

            if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                continue

            if criterion == 'gini':
                value = _gini_impurity(y[left_indices], y[right_indices])
            elif criterion == 'entropy':
                value = _information_gain(y[left_indices], y[right_indices])
            else:
                raise ValueError("Invalid criterion.")

            if value > best_value:
                best_value = value
                best_split = {
                    'feature': feature,
                    'threshold': threshold
                }

    return best_split

def _gini_impurity(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """
    Calculate the Gini impurity for a split.

    Parameters:
    -----------
    y_left : np.ndarray
        Target values of the left split.
    y_right : np.ndarray
        Target values of the right split.

    Returns:
    --------
    float
        The Gini impurity.
    """
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right

    gini_left = 1 - np.sum((np.bincount(y_left) / n_left) ** 2)
    gini_right = 1 - np.sum((np.bincount(y_right) / n_right) ** 2)

    return (n_left / n_total) * gini_left + (n_right / n_total) * gini_right

def _information_gain(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """
    Calculate the information gain for a split.

    Parameters:
    -----------
    y_left : np.ndarray
        Target values of the left split.
    y_right : np.ndarray
        Target values of the right split.

    Returns:
    --------
    float
        The information gain.
    """
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right

    entropy_left = _entropy(y_left)
    entropy_right = _entropy(y_right)

    return (n_left / n_total) * entropy_left + (n_right / n_total) * entropy_right

def _entropy(y: np.ndarray) -> float:
    """
    Calculate the entropy of a set of target values.

    Parameters:
    -----------
    y : np.ndarray
        Target values.

    Returns:
    --------
    float
        The entropy.
    """
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def _predict(trees: list, X: np.ndarray) -> np.ndarray:
    """
    Predict the target values for a given set of input data.

    Parameters:
    -----------
    trees : list
        A list of fitted decision trees.
    X : np.ndarray
        Input data of shape (n_samples, n_features).

    Returns:
    --------
    np.ndarray
        Predicted target values of shape (n_samples,).
    """
    predictions = np.zeros((X.shape[0], len(trees)))
    for i, tree in enumerate(trees):
        predictions[:, i] = _predict_single_tree(tree, X)
    return np.mean(predictions, axis=1)

def _predict_single_tree(tree: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """
    Predict the target values for a given set of input data using a single decision tree.

    Parameters:
    -----------
    tree : Dict[str, Any]
        A fitted decision tree.
    X : np.ndarray
        Input data of shape (n_samples, n_features).

    Returns:
    --------
    np.ndarray
        Predicted target values of shape (n_samples,).
    """
    predictions = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        node = tree
        while not node['leaf']:
            if X[i, node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        predictions[i] = node['value']
    return predictions

################################################################################
# gradient_boosting
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def gradient_boosting_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    min_samples_split: int = 2,
    loss: str = 'mse',
    validation_fraction: float = 0.1,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    early_stopping_rounds: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Fit a gradient boosting model to the data.

    Parameters:
    -----------
    X : np.ndarray
        Training data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    n_estimators : int, optional
        Number of boosting stages to perform.
    learning_rate : float, optional
        Shrinkage factor for each tree's contribution.
    max_depth : int, optional
        Maximum depth of the individual regression estimators.
    min_samples_split : int, optional
        Minimum number of samples required to split an internal node.
    loss : str or callable, optional
        Loss function to be optimized. 'mse', 'mae', or custom callable.
    validation_fraction : float, optional
        Fraction of data to use for early stopping validation.
    tol : float, optional
        Tolerance for the loss improvement.
    random_state : int or None, optional
        Random seed for reproducibility.
    normalizer : callable, optional
        Function to normalize the input data.
    metric : str or callable, optional
        Metric to evaluate the model. 'mse', 'mae', 'r2', or custom callable.
    early_stopping_rounds : int or None, optional
        Number of rounds to wait before stopping if no improvement.
    verbose : bool, optional
        Whether to print progress information.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the fitted model and related information.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_normalized = normalizer(X)

    # Initialize model components
    model = _initialize_model(n_estimators, learning_rate, max_depth,
                             min_samples_split, random_state)

    # Fit the model
    result = _fit_model(X_normalized, y, model, loss, validation_fraction,
                        tol, early_stopping_rounds, verbose)

    # Prepare output
    output = {
        'result': result,
        'metrics': _compute_metrics(result['y_pred'], y, metric),
        'params_used': {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'loss': loss,
            'validation_fraction': validation_fraction,
            'tol': tol,
            'random_state': random_state
        },
        'warnings': []
    }

    return output

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
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

def _initialize_model(
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    min_samples_split: int,
    random_state: Optional[int]
) -> Dict[str, Any]:
    """Initialize the gradient boosting model components."""
    if random_state is not None:
        np.random.seed(random_state)

    return {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'trees': [],
        'random_state': random_state
    }

def _fit_model(
    X: np.ndarray,
    y: np.ndarray,
    model: Dict[str, Any],
    loss: Union[str, Callable],
    validation_fraction: float,
    tol: float,
    early_stopping_rounds: Optional[int],
    verbose: bool
) -> Dict[str, Any]:
    """Fit the gradient boosting model to the data."""
    n_samples = X.shape[0]
    validation_size = int(n_samples * validation_fraction)

    if validation_size > 0:
        X_train, X_val = np.split(X, [n_samples - validation_size])
        y_train, y_val = np.split(y, [n_samples - validation_size])
    else:
        X_train, y_train = X, y
        X_val, y_val = None, None

    # Initialize predictions
    y_pred = np.full(y.shape, np.mean(y), dtype=np.float64)

    # Gradient boosting loop
    best_loss = float('inf')
    no_improvement_count = 0

    for i in range(model['n_estimators']):
        # Compute gradients
        if loss == 'mse':
            gradients = _compute_mse_gradients(y_train, y_pred)
        elif loss == 'mae':
            gradients = _compute_mae_gradients(y_train, y_pred)
        elif callable(loss):
            gradients = loss(y_train, y_pred)
        else:
            raise ValueError(f"Unsupported loss function: {loss}")

        # Fit a tree to the gradients
        tree = _fit_decision_tree(X_train, gradients,
                                 max_depth=model['max_depth'],
                                 min_samples_split=model['min_samples_split'])

        # Update predictions
        y_pred += model['learning_rate'] * tree.predict(X)

        # Evaluate on validation set if available
        current_loss = _compute_validation_loss(y_val, y_pred[-validation_size:], loss)

        if verbose:
            print(f"Boosting iteration {i + 1}/{model['n_estimators']}, "
                  f"validation loss: {current_loss:.4f}")

        # Check for early stopping
        if current_loss < best_loss - tol:
            best_loss = current_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if (early_stopping_rounds is not None and
            no_improvement_count >= early_stopping_rounds):
            if verbose:
                print(f"Early stopping at iteration {i + 1}")
            model['n_estimators'] = i
            break

        # Store the tree
        model['trees'].append(tree)

    return {
        'y_pred': y_pred,
        'model': model
    }

def _compute_mse_gradients(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute gradients for mean squared error loss."""
    return y_pred - y_true

def _compute_mae_gradients(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute gradients for mean absolute error loss."""
    return np.sign(y_pred - y_true)

def _fit_decision_tree(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int,
    min_samples_split: int
) -> Any:
    """Fit a decision tree to the data."""
    # This is a placeholder for the actual decision tree implementation
    # In a real implementation, this would be replaced with an actual decision tree fitting function
    class DummyTree:
        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.zeros(X.shape[0])

    return DummyTree()

def _compute_validation_loss(
    y_true: Optional[np.ndarray],
    y_pred: np.ndarray,
    loss: Union[str, Callable]
) -> float:
    """Compute validation loss."""
    if y_true is None or len(y_pred) == 0:
        return float('inf')

    if loss == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif loss == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif callable(loss):
        return loss(y_true, y_pred)
    else:
        raise ValueError(f"Unsupported loss function: {loss}")

def _compute_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}

    if metric == 'mse' or (callable(metric) and not hasattr(metric, '__name__')):
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    if metric == 'mae' or (callable(metric) and not hasattr(metric, '__name__')):
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    if metric == 'r2' or (callable(metric) and not hasattr(metric, '__name__')):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)

    if callable(metric):
        metrics['custom'] = metric(y_true, y_pred)

    return metrics

################################################################################
# adaboost
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

def default_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Default metric: Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def default_base_estimator(X: np.ndarray, y: np.ndarray) -> Callable:
    """Default base estimator: Decision Tree."""
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(max_depth=1)

def compute_weights(y: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute sample weights for AdaBoost."""
    if weights is None:
        return np.ones(len(y)) / len(y)
    return weights

def update_weights(weights: np.ndarray, errors: np.ndarray) -> np.ndarray:
    """Update sample weights based on classification errors."""
    epsilon = 1e-10
    errors = np.clip(errors, epsilon, 1 - epsilon)
    return weights * np.exp(-np.log(1 / errors - 1) * (y_pred == y))

def compute_estimator_weight(errors: np.ndarray) -> float:
    """Compute the weight for a new estimator."""
    epsilon = 1e-10
    errors = np.clip(errors, epsilon, 1 - epsilon)
    return 0.5 * np.log((1 - errors) / (errors + epsilon))

def adaboost_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 50,
    learning_rate: float = 1.0,
    metric: Callable = default_metric,
    base_estimator: Optional[Callable] = None,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Fit an AdaBoost model.

    Parameters:
    - X: Training data of shape (n_samples, n_features)
    - y: Target values of shape (n_samples,)
    - n_estimators: Number of estimators in the ensemble
    - learning_rate: Learning rate shrinks the contribution of each estimator
    - metric: Metric function to evaluate performance
    - base_estimator: Base estimator to use (default is Decision Tree)
    - random_state: Random seed for reproducibility

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings
    """
    validate_inputs(X, y)

    if base_estimator is None:
        base_estimator = default_base_estimator(X, y)

    n_samples = X.shape[0]
    weights = compute_weights(y)
    estimators = []
    estimator_weights = []

    for _ in range(n_estimators):
        # Fit base estimator
        estimator = base_estimator.fit(X, y, sample_weight=weights)
        estimators.append(estimator)

        # Predict with current estimator
        y_pred = estimator.predict(X)

        # Compute errors and estimator weight
        incorrect = (y_pred != y)
        errors = np.dot(weights, incorrect) / np.sum(weights)
        estimator_weight = compute_estimator_weight(errors)

        # Update weights
        weights = update_weights(weights, errors)
        weights /= np.sum(weights)  # Normalize

        estimator_weights.append(estimator_weight * learning_rate)

    # Compute final predictions
    y_pred_final = np.zeros_like(y, dtype=float)
    for estimator, weight in zip(estimators, estimator_weights):
        y_pred_final += weight * estimator.predict(X)

    # Compute metrics
    final_metric = metric(y, y_pred_final)

    return {
        "result": {"estimators": estimators, "estimator_weights": estimator_weights},
        "metrics": {"final_metric": final_metric},
        "params_used": {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "metric": metric.__name__ if hasattr(metric, '__name__') else "custom",
            "base_estimator": base_estimator.__class__.__name__ if hasattr(base_estimator, '__class__') else "custom"
        },
        "warnings": []
    }

# Example usage:
"""
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, size=100)

result = adaboost_fit(X, y)
"""

################################################################################
# bagging
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    max_samples: float = 1.0,
    max_features: float = 1.0,
    bootstrap: bool = True
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive")
    if not (0 < max_samples <= 1):
        raise ValueError("max_samples must be in (0, 1]")
    if not (0 < max_features <= 1):
        raise ValueError("max_features must be in (0, 1]")

def _compute_sample_indices(
    n_samples: int,
    max_samples: float,
    bootstrap: bool
) -> np.ndarray:
    """Compute random sample indices for a single estimator."""
    n_samples = int(max_samples * n_samples)
    if bootstrap:
        return np.random.choice(n_samples, size=n_samples, replace=True)
    else:
        return np.random.choice(n_samples, size=n_samples, replace=False)

def _compute_feature_indices(
    n_features: int,
    max_features: float
) -> np.ndarray:
    """Compute random feature indices for a single estimator."""
    n_features = int(max_features * n_features)
    return np.random.choice(n_features, size=n_features, replace=False)

def _fit_single_estimator(
    X: np.ndarray,
    y: np.ndarray,
    sample_indices: np.ndarray,
    feature_indices: np.ndarray,
    tree_constructor: Callable
) -> Any:
    """Fit a single decision tree estimator."""
    X_subset = X[sample_indices, :][:, feature_indices]
    y_subset = y[sample_indices]
    return tree_constructor(X_subset, y_subset)

def bagging_fit(
    X: np.ndarray,
    y: np.ndarray,
    tree_constructor: Callable,
    n_estimators: int = 100,
    max_samples: float = 1.0,
    max_features: float = 1.0,
    bootstrap: bool = True,
    metric: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Fit a bagging ensemble of decision trees.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    tree_constructor : Callable
        Function to construct a single decision tree
    n_estimators : int, optional
        Number of trees in the ensemble (default: 100)
    max_samples : float, optional
        Fraction of samples to draw for each tree (default: 1.0)
    max_features : float, optional
        Fraction of features to draw for each tree (default: 1.0)
    bootstrap : bool, optional
        Whether to use bootstrap sampling (default: True)
    metric : Callable, optional
        Metric function to evaluate the ensemble (default: None)

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the fitted ensemble and evaluation results
    """
    _validate_inputs(X, y, n_estimators, max_samples, max_features, bootstrap)

    n_samples, n_features = X.shape
    estimators = []

    for _ in range(n_estimators):
        sample_indices = _compute_sample_indices(n_samples, max_samples, bootstrap)
        feature_indices = _compute_feature_indices(n_features, max_features)
        estimator = _fit_single_estimator(X, y, sample_indices, feature_indices, tree_constructor)
        estimators.append(estimator)

    result = {
        "result": {"estimators": estimators},
        "metrics": {},
        "params_used": {
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "max_features": max_features,
            "bootstrap": bootstrap
        },
        "warnings": []
    }

    if metric is not None:
        predictions = np.array([estimator.predict(X) for estimator in estimators]).mean(axis=0)
        result["metrics"]["ensemble_metric"] = metric(y, predictions)

    return result

# Example usage:
"""
def simple_tree_constructor(X: np.ndarray, y: np.ndarray) -> Any:
    # This is a placeholder for an actual tree constructor
    class SimpleTree:
        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.mean(y)
    return SimpleTree()

X_example = np.random.rand(100, 5)
y_example = np.random.rand(100)

result = bagging_fit(
    X=X_example,
    y=y_example,
    tree_constructor=simple_tree_constructor
)
"""

################################################################################
# boosting
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def boosting_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    loss: str = 'mse',
    metric: Union[str, Callable] = 'mse',
    normalizer: Optional[Callable] = None,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Fit a boosting ensemble of decision trees.

    Parameters:
    -----------
    X : np.ndarray
        Training data of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    n_estimators : int, default=100
        Number of boosting stages to be run.
    learning_rate : float, default=0.1
        Shrinkage factor for each tree's contribution.
    loss : str, default='mse'
        Loss function to be optimized. Options: 'mse', 'mae', 'huber'
    metric : str or callable, default='mse'
        Metric to evaluate the quality of splits. Options: 'mse', 'mae', custom callable
    normalizer : callable, default=None
        Function to normalize the input data. If None, no normalization is applied.
    random_state : int, default=None
        Random seed for reproducibility.

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': trained model object
        - 'metrics': evaluation metrics
        - 'params_used': parameters used for fitting
        - 'warnings': any warnings encountered

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = boosting_fit(X, y, n_estimators=50, learning_rate=0.2)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set random state if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Initialize parameters
    params_used = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'loss': loss,
        'metric': metric,
        'normalizer': normalizer is not None
    }

    # Normalize data if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Initialize model components
    ensemble = []
    residuals = y.copy()

    metrics = {
        'train_error': [],
        'validation_error': []  # Placeholder for validation error
    }

    warnings = []

    # Boosting loop
    for i in range(n_estimators):
        # Fit weak learner on current residuals
        tree = _fit_weak_learner(
            X_normalized,
            residuals,
            loss=loss,
            metric=metric
        )

        # Update predictions and residuals
        predictions = _predict_weak_learner(tree, X_normalized)
        ensemble.append(predictions * learning_rate)

        # Update residuals
        residuals -= predictions * learning_rate

        # Calculate and store metrics
        train_error = _calculate_metric(y, np.sum(ensemble, axis=0), metric)
        metrics['train_error'].append(train_error)

    # Prepare result dictionary
    result = {
        'ensemble': np.array(ensemble),
        'normalized_features': X_normalized
    }

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
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

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to input data if specified."""
    if normalizer is None:
        return X
    try:
        return normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _fit_weak_learner(
    X: np.ndarray,
    y: np.ndarray,
    loss: str = 'mse',
    metric: Union[str, Callable] = 'mse'
) -> Dict[str, Any]:
    """Fit a single weak learner (decision tree) to the data."""
    # In a real implementation, this would be replaced with actual decision tree fitting
    # For this example, we'll return a mock weak learner

    if loss not in ['mse', 'mae', 'huber']:
        raise ValueError(f"Unsupported loss function: {loss}")

    if isinstance(metric, str) and metric not in ['mse', 'mae']:
        raise ValueError(f"Unsupported metric: {metric}")

    # Mock weak learner implementation
    tree = {
        'loss': loss,
        'metric': metric,
        'predictions': np.random.rand(X.shape[0])  # Mock predictions
    }

    return tree

def _predict_weak_learner(tree: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Make predictions using a weak learner."""
    return tree['predictions']

def _calculate_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> float:
    """Calculate evaluation metric."""
    if isinstance(metric, str):
        if metric == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            return np.mean(np.abs(y_true - y_pred))
    else:
        try:
            return metric(y_true, y_pred)
        except Exception as e:
            raise ValueError(f"Custom metric failed: {str(e)}")

    raise ValueError(f"Unsupported metric: {metric}")
