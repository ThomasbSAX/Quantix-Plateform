"""
Quantix – Module modeles_hierarchiques
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# arbre_decision
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

def normalize_data(X: np.ndarray, y: np.ndarray, normalization: str = "standard") -> tuple:
    """Normalize input data based on specified method."""
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
        X_q75, X_q25 = np.percentile(X, [75, 25], axis=0)
        X_iqr = X_q75 - X_q25
        X_iqr[X_iqr == 0] = 1.0
        X_normalized = (X - X_median) / X_iqr
        y_median = np.median(y)
        y_q75, y_q25 = np.percentile(y, [75, 25])
        if y_q75 == y_q25:
            y_normalized = np.zeros_like(y)
        else:
            y_iqr = y_q75 - y_q25
            y_normalized = (y - y_median) / y_iqr
        return X_normalized, y_normalized
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str = "mse") -> float:
    """Compute specified metric between true and predicted values."""
    if callable(metric):
        return metric(y_true, y_pred)
    elif metric == "mse":
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

def compute_distance(X1: np.ndarray, X2: np.ndarray, distance: str = "euclidean") -> np.ndarray:
    """Compute pairwise distances between two sets of points."""
    if callable(distance):
        return distance(X1, X2)
    elif distance == "euclidean":
        return np.sqrt(np.sum((X1[:, np.newaxis] - X2) ** 2, axis=2))
    elif distance == "manhattan":
        return np.sum(np.abs(X1[:, np.newaxis] - X2), axis=2)
    elif distance == "cosine":
        dot_products = np.dot(X1, X2.T)
        norms1 = np.linalg.norm(X1, axis=1)[:, np.newaxis]
        norms2 = np.linalg.norm(X2, axis=1)[np.newaxis, :]
        return 1 - (dot_products / (norms1 * norms2))
    elif distance == "minkowski":
        p = 3
        return np.sum(np.abs(X1[:, np.newaxis] - X2) ** p, axis=2) ** (1/p)
    else:
        raise ValueError(f"Unknown distance metric: {distance}")

def fit_decision_tree(X: np.ndarray, y: np.ndarray,
                     max_depth: int = None,
                     min_samples_split: int = 2,
                     metric: Union[str, Callable] = "mse",
                     distance: Union[str, Callable] = "euclidean") -> Dict[str, Any]:
    """Fit a decision tree to the data."""
    # Base case: if max_depth is reached or not enough samples, return leaf
    if (max_depth is not None and max_depth <= 0) or len(y) < min_samples_split:
        return {
            "type": "leaf",
            "value": np.mean(y) if len(y) > 0 else 0,
            "samples": len(y)
        }

    # Find best split
    best_split = None
    best_metric = float('inf')

    for feature_idx in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_idx])
        for threshold in thresholds:
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            left_y = y[left_mask]
            right_y = y[right_mask]

            # Calculate metric for this split
            left_pred = np.mean(left_y)
            right_pred = np.mean(right_y)

            current_metric = (compute_metric(y[left_mask], left_pred * np.ones_like(left_y), metric) *
                             len(left_y) +
                             compute_metric(y[right_mask], right_pred * np.ones_like(right_y), metric) *
                             len(right_y)) / len(y)

            if current_metric < best_metric:
                best_metric = current_metric
                best_split = {
                    "feature": feature_idx,
                    "threshold": threshold,
                    "left_mask": left_mask,
                    "right_mask": right_mask
                }

    # If no split improved the metric, return leaf
    if best_split is None:
        return {
            "type": "leaf",
            "value": np.mean(y) if len(y) > 0 else 0,
            "samples": len(y)
        }

    # Recursively fit left and right subtrees
    left_subtree = fit_decision_tree(
        X[best_split["left_mask"]],
        y[best_split["left_mask"]],
        max_depth=(max_depth - 1) if max_depth is not None else None,
        min_samples_split=min_samples_split,
        metric=metric,
        distance=distance
    )

    right_subtree = fit_decision_tree(
        X[best_split["right_mask"]],
        y[best_split["right_mask"]],
        max_depth=(max_depth - 1) if max_depth is not None else None,
        min_samples_split=min_samples_split,
        metric=metric,
        distance=distance
    )

    return {
        "type": "node",
        "feature": best_split["feature"],
        "threshold": best_split["threshold"],
        "left": left_subtree,
        "right": right_subtree,
        "samples": len(y)
    }

def predict_decision_tree(tree: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Predict using a fitted decision tree."""
    predictions = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        node = tree
        while node["type"] == "node":
            if X[i, node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        predictions[i] = node["value"]

    return predictions

def arbre_decision_fit(X: np.ndarray, y: np.ndarray,
                      normalization: str = "standard",
                      metric: Union[str, Callable] = "mse",
                      distance: Union[str, Callable] = "euclidean",
                      max_depth: int = None,
                      min_samples_split: int = 2) -> Dict[str, Any]:
    """
    Fit a hierarchical decision tree model.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    normalization : str or callable, optional
        Normalization method to apply (default: "standard")
    metric : str or callable, optional
        Metric to optimize (default: "mse")
    distance : str or callable, optional
        Distance metric for hierarchical clustering (default: "euclidean")
    max_depth : int, optional
        Maximum depth of the tree (default: None)
    min_samples_split : int, optional
        Minimum number of samples required to split a node (default: 2)

    Returns:
    --------
    dict
        Dictionary containing the fitted model and related information

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> model = arbre_decision_fit(X, y)
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = normalize_data(X, y, normalization)

    # Fit decision tree
    tree = fit_decision_tree(
        X_norm,
        y_norm,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        metric=metric,
        distance=distance
    )

    # Make predictions on training data for metrics
    y_pred = predict_decision_tree(tree, X_norm)

    # Compute metrics
    metrics = {
        "train_metric": compute_metric(y_norm, y_pred, metric),
    }

    # Prepare output
    result = {
        "model": tree,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric if not callable(metric) else "custom",
            "distance": distance if not callable(distance) else "custom",
            "max_depth": max_depth,
            "min_samples_split": min_samples_split
        },
        "warnings": []
    }

    return result

def arbre_decision_compute(model: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """
    Compute predictions using a fitted decision tree model.

    Parameters:
    -----------
    model : dict
        Fitted model returned by arbre_decision_fit
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)

    Returns:
    --------
    np.ndarray
        Predicted values of shape (n_samples,)

    Example:
    --------
    >>> X_test = np.random.rand(10, 5)
    >>> predictions = arbre_decision_compute(model, X_test)
    """
    return predict_decision_tree(model["model"], X)

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
    max_features: Union[int, float] = "auto",
    bootstrap: bool = True,
    criterion: str = "mse",
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    distance_metric: str = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Fit a random forest model with hierarchical structure.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,) or (n_samples, n_outputs).
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : Optional[int], default=None
        Maximum depth of the tree. If None, nodes are expanded until all leaves are pure.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    max_features : Union[int, float], default="auto"
        Number of features to consider when looking for the best split.
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.
    criterion : str, default="mse"
        Function to measure the quality of a split. Supported criteria are "mse" for regression and "gini" or "entropy" for classification.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], default=None
        Function to normalize the input features.
    distance_metric : str, default="euclidean"
        Distance metric used for hierarchical clustering within trees.
    solver : str, default="closed_form"
        Solver to use for optimization. Supported solvers are "closed_form", "gradient_descent", "newton", and "coordinate_descent".
    regularization : Optional[str], default=None
        Regularization type. Supported types are "l1", "l2", and "elasticnet".
    tol : float, default=1e-4
        Tolerance for the optimization.
    max_iter : int, default=1000
        Maximum number of iterations for the solver.
    random_state : Optional[int], default=None
        Random seed for reproducibility.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if a normalizer is provided
    if normalizer is not None:
        X = normalizer(X)

    # Initialize the random forest
    forest = _initialize_forest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        criterion=criterion,
        distance_metric=distance_metric,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state
    )

    # Fit the forest
    _fit_forest(forest, X, y)

    # Calculate metrics
    metrics = _calculate_metrics(X, y, forest, criterion)

    return {
        "result": forest,
        "metrics": metrics,
        "params_used": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "bootstrap": bootstrap,
            "criterion": criterion,
            "distance_metric": distance_metric,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter,
            "random_state": random_state
        },
        "warnings": []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate the input data."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1 and y.ndim != 2:
        raise ValueError("y must be a 1D or 2D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input data contains NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input data contains infinite values.")

def _initialize_forest(
    n_estimators: int,
    max_depth: Optional[int],
    min_samples_split: int,
    min_samples_leaf: int,
    max_features: Union[int, float],
    bootstrap: bool,
    criterion: str,
    distance_metric: str,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    random_state: Optional[int]
) -> Dict[str, Any]:
    """Initialize the random forest structure."""
    if max_features == "auto":
        max_features = int(np.sqrt(X.shape[1]))

    forest = {
        "trees": [],
        "params": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "bootstrap": bootstrap,
            "criterion": criterion,
            "distance_metric": distance_metric,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter,
            "random_state": random_state
        }
    }

    return forest

def _fit_forest(forest: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> None:
    """Fit the random forest to the data."""
    params = forest["params"]
    n_estimators = params["n_estimators"]

    for _ in range(n_estimators):
        tree = _build_tree(X, y, params)
        forest["trees"].append(tree)

def _build_tree(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """Build a single decision tree."""
    tree = {
        "nodes": [],
        "params": params
    }

    # Implement tree building logic here
    # This is a placeholder for the actual implementation

    return tree

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    forest: Dict[str, Any],
    criterion: str
) -> Dict[str, float]:
    """Calculate the metrics for the random forest."""
    y_pred = _predict(forest, X)

    if criterion == "mse":
        mse = np.mean((y - y_pred) ** 2)
        return {"mse": mse}
    elif criterion == "mae":
        mae = np.mean(np.abs(y - y_pred))
        return {"mae": mae}
    elif criterion == "r2":
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return {"r2": r2}
    else:
        raise ValueError(f"Unsupported criterion: {criterion}")

def _predict(forest: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Predict using the random forest."""
    y_preds = []

    for tree in forest["trees"]:
        y_pred = _predict_tree(tree, X)
        y_preds.append(y_pred)

    return np.mean(y_preds, axis=0)

def _predict_tree(tree: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Predict using a single decision tree."""
    # Implement tree prediction logic here
    # This is a placeholder for the actual implementation
    return np.zeros(X.shape[0])

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
    criterion: Optional[Callable] = None,
    normalizer: Optional[Callable] = None,
    solver: str = 'gradient_descent',
    reg_type: Optional[str] = None,
    tol: float = 1e-4,
    max_depth: int = 3,
    min_samples_split: int = 2,
    random_state: Optional[int] = None
) -> Dict:
    """
    Fit a gradient boosting model to the given data.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    n_estimators : int, optional
        Number of boosting stages to perform (default=100)
    learning_rate : float, optional
        Shrinkage factor for each tree (default=0.1)
    loss : str or callable, optional
        Loss function to be optimized ('mse', 'mae', etc.) (default='mse')
    criterion : callable, optional
        Custom loss function if not using built-in ones
    normalizer : callable, optional
        Normalization function to apply to features
    solver : str, optional
        Solver type ('gradient_descent', 'newton', etc.) (default='gradient_descent')
    reg_type : str, optional
        Regularization type ('l1', 'l2', 'elasticnet') (default=None)
    tol : float, optional
        Tolerance for early stopping (default=1e-4)
    max_depth : int, optional
        Maximum depth of individual trees (default=3)
    min_samples_split : int, optional
        Minimum number of samples required to split a node (default=2)
    random_state : int, optional
        Random seed for reproducibility (default=None)

    Returns:
    --------
    Dict containing:
        - 'result': fitted model parameters
        - 'metrics': computed metrics
        - 'params_used': actual parameters used
        - 'warnings': any warnings generated
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Initialize results dictionary
    result = {
        'result': {},
        'metrics': {},
        'params_used': locals(),
        'warnings': []
    }

    # Apply normalization if specified
    if normalizer is not None:
        X = _apply_normalization(X, normalizer)

    # Initialize model components
    model = {
        'trees': [],
        'loss_history': []
    }

    # Main boosting loop
    for i in range(n_estimators):
        # Compute residuals/pseudo-residuals based on loss function
        if criterion is None:
            residuals = _compute_residuals(y, model['trees'], loss)
        else:
            residuals = criterion(y, [tree.predict(X) for tree in model['trees']])

        # Fit weak learner to residuals
        tree = _fit_weak_learner(
            X, residuals,
            solver=solver,
            reg_type=reg_type,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )

        # Update model with new tree
        model['trees'].append(tree)

        # Compute loss for current iteration
        current_loss = _compute_loss(y, model['trees'], loss)
        model['loss_history'].append(current_loss)

        # Check for early stopping
        if i > 0 and abs(model['loss_history'][-2] - current_loss) < tol:
            result['warnings'].append(f'Early stopping at iteration {i}')
            break

    # Compute final metrics
    result['metrics'] = _compute_metrics(y, model['trees'], loss)
    result['result']['model'] = model

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalization(X: np.ndarray, normalizer: Callable) -> np.ndarray:
    """Apply normalization to features."""
    return normalizer(X)

def _compute_residuals(
    y: np.ndarray,
    trees: list,
    loss: str
) -> np.ndarray:
    """Compute residuals based on current model and specified loss."""
    if not trees:
        return y.copy()

    predictions = np.sum([tree.predict(X) for tree in trees], axis=0)

    if loss == 'mse':
        return y - predictions
    elif loss == 'mae':
        return np.sign(y - predictions)
    else:
        raise ValueError(f"Unsupported loss function: {loss}")

def _fit_weak_learner(
    X: np.ndarray,
    y: np.ndarray,
    solver: str = 'gradient_descent',
    reg_type: Optional[str] = None,
    max_depth: int = 3,
    min_samples_split: int = 2,
    random_state: Optional[int] = None
) -> object:
    """Fit a weak learner (decision tree) to the data."""
    # In a real implementation, this would be replaced with actual tree fitting
    class DummyTree:
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def predict(self, X_new):
            return np.zeros(X_new.shape[0])

    return DummyTree(X, y)

def _compute_loss(
    y: np.ndarray,
    trees: list,
    loss: str
) -> float:
    """Compute the specified loss function."""
    if not trees:
        return np.mean((y - np.zeros_like(y))**2)

    predictions = np.sum([tree.predict(X) for tree in trees], axis=0)

    if loss == 'mse':
        return np.mean((y - predictions)**2)
    elif loss == 'mae':
        return np.mean(np.abs(y - predictions))
    else:
        raise ValueError(f"Unsupported loss function: {loss}")

def _compute_metrics(
    y: np.ndarray,
    trees: list,
    loss: str
) -> Dict:
    """Compute various metrics for the model."""
    if not trees:
        return {'loss': np.mean((y - np.zeros_like(y))**2)}

    predictions = np.sum([tree.predict(X) for tree in trees], axis=0)

    metrics = {
        'loss': _compute_loss(y, trees, loss),
        'mse': np.mean((y - predictions)**2),
        'mae': np.mean(np.abs(y - predictions)),
    }

    if loss != 'mse':
        metrics['r2'] = 1 - np.sum((y - predictions)**2) / np.sum((y - np.mean(y))**2)

    return metrics

################################################################################
# xgboost
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
    """Normalize data based on user choice."""
    if normalization == "standard":
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X = (X - X_mean) / (X_std + 1e-8)
    elif normalization == "minmax":
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X = (X - X_min) / (X_max - X_min + 1e-8)
    elif normalization == "robust":
        X_median = np.median(X, axis=0)
        X_q75, X_q25 = np.percentile(X, [75, 25], axis=0)
        X_iqr = X_q75 - X_q25
        X = (X - X_median) / (X_iqr + 1e-8)
    return X, y

def _compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: Union[str, Callable]) -> float:
    """Compute the chosen metric."""
    if callable(metric):
        return metric(y_true, y_pred)
    elif metric == "mse":
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

def _xgboost_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.1,
    n_estimators: int = 100,
    max_depth: int = 3,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0
) -> Dict:
    """XGBoost implementation using gradient descent."""
    n_samples, n_features = X.shape
    trees = []
    residuals = y.copy()

    for _ in range(n_estimators):
        # Sample data
        sample_indices = np.random.choice(n_samples, size=int(n_samples * subsample), replace=False)
        X_sample = X[sample_indices]
        y_sample = residuals[sample_indices]

        # Fit a decision tree to the residuals
        tree = _fit_decision_tree(X_sample, y_sample, max_depth, min_child_weight, gamma)
        trees.append(tree)

        # Update residuals
        update = learning_rate * _predict_tree(X, tree)
        residuals -= update

    return {"trees": trees}

def _fit_decision_tree(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int,
    min_child_weight: float,
    gamma: float
) -> Dict:
    """Fit a decision tree to the data."""
    # Simplified implementation - in practice, use a proper tree implementation
    tree = {
        "depth": 0,
        "split_feature": None,
        "split_value": None,
        "left_child": None,
        "right_child": None,
        "value": np.mean(y)
    }
    return tree

def _predict_tree(X: np.ndarray, tree: Dict) -> np.ndarray:
    """Predict using a single decision tree."""
    return np.full(X.shape[0], tree["value"])

def xgboost_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalization: str = "none",
    metric: Union[str, Callable] = "mse",
    learning_rate: float = 0.1,
    n_estimators: int = 100,
    max_depth: int = 3,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0
) -> Dict:
    """
    Fit an XGBoost model to the data.

    Parameters:
    - X: Input features (n_samples, n_features)
    - y: Target values (n_samples,)
    - normalization: Data normalization method
    - metric: Evaluation metric
    - learning_rate: Learning rate for gradient boosting
    - n_estimators: Number of boosting rounds
    - max_depth: Maximum depth of each tree
    - min_child_weight: Minimum sum of instance weight needed in a child
    - gamma: Minimum loss reduction required to make a split
    - subsample: Subsample ratio of the training instances
    - colsample_bytree: Subsample ratio of features used per tree

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = _normalize_data(X.copy(), y.copy(), normalization)

    # Fit model
    result = _xgboost_gradient_descent(
        X_norm, y_norm,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree
    )

    # Make predictions
    y_pred = np.zeros_like(y_norm)
    for tree in result["trees"]:
        y_pred += _predict_tree(X_norm, tree)

    # Compute metrics
    metrics = {
        "metric": _compute_metric(y_norm, y_pred, metric)
    }

    # Prepare output
    return {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree
        },
        "warnings": []
    }

# Example usage:
"""
X = np.random.rand(100, 5)
y = np.random.rand(100)

result = xgboost_fit(
    X, y,
    normalization="standard",
    metric="mse",
    learning_rate=0.1,
    n_estimators=50
)
"""

################################################################################
# lightgbm
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> None:
    """
    Validate input data for LightGBM model.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    sample_weight : Optional[np.ndarray], default=None
        Sample weights array of shape (n_samples,)

    Raises
    ------
    ValueError
        If input validation fails
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if sample_weight is not None:
        if X.shape[0] != sample_weight.shape[0]:
            raise ValueError("sample_weight must have same length as X")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative values")

def default_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Default metric function (MSE).

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values

    Returns
    ------
    float
        Mean squared error
    """
    return np.mean((y_true - y_pred) ** 2)

def lightgbm_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = -1,
    min_child_samples: int = 20,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    metric: Union[str, Callable] = 'mse',
    normalizer: Optional[Callable] = None,
    sample_weight: Optional[np.ndarray] = None,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Fit a LightGBM model with hierarchical considerations.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    n_estimators : int, default=100
        Number of boosting iterations
    learning_rate : float, default=0.1
        Shrinkage rate for each iteration
    max_depth : int, default=-1
        Maximum tree depth (-1 means no limit)
    min_child_samples : int, default=20
        Minimum number of samples per leaf
    subsample : float, default=1.0
        Fraction of samples used for each tree
    colsample_bytree : float, default=1.0
        Fraction of features used for each tree
    metric : Union[str, Callable], default='mse'
        Evaluation metric ('mse', 'mae', etc.) or custom callable
    normalizer : Optional[Callable], default=None
        Feature normalization function
    sample_weight : Optional[np.ndarray], default=None
        Sample weights array of shape (n_samples,)
    random_state : Optional[int], default=None
        Random seed for reproducibility

    Returns
    ------
    Dict[str, Any]
        Dictionary containing:
        - 'result': fitted model parameters
        - 'metrics': evaluation metrics
        - 'params_used': actual parameters used
        - 'warnings': any warnings generated

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = lightgbm_fit(X, y)
    """
    # Validate inputs
    validate_inputs(X, y, sample_weight)

    # Initialize warnings list
    warnings = []

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize features if normalizer provided
    X_normalized = X.copy()
    if normalizer is not None:
        try:
            X_normalized = normalizer(X)
        except Exception as e:
            warnings.append(f"Normalization failed: {str(e)}")
            X_normalized = X.copy()

    # Initialize model parameters
    params_used = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'min_child_samples': min_child_samples,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'metric': metric,
        'normalizer': normalizer is not None,
        'sample_weight': sample_weight is not None
    }

    # Initialize model results
    result = {
        'trees': [],
        'feature_importances': np.zeros(X.shape[1])
    }

    # Initialize metrics
    metrics = {}

    # Main training loop
    for i in range(n_estimators):
        # Get subsample of data
        if subsample < 1.0:
            n_samples = int(X_normalized.shape[0] * subsample)
            indices = np.random.choice(X_normalized.shape[0], n_samples, replace=False)
        else:
            indices = np.arange(X_normalized.shape[0])

        X_sub = X_normalized[indices]
        y_sub = y[indices]

        if sample_weight is not None:
            w_sub = sample_weight[indices]
        else:
            w_sub = np.ones(len(y_sub))

        # Fit a single tree (simplified implementation)
        tree = _fit_single_tree(X_sub, y_sub, w_sub,
                              max_depth=max_depth,
                              min_child_samples=min_child_samples,
                              colsample_bytree=colsample_bytree)

        # Update feature importances
        result['feature_importances'] += tree['importance']

        # Make predictions and update metrics
        y_pred = _predict(X_normalized, [tree])
        if isinstance(metric, str):
            current_metric = _get_metric_function(metric)(y, y_pred)
        else:
            current_metric = metric(y, y_pred)

        metrics[f'iteration_{i}'] = current_metric

    # Normalize feature importances
    if result['feature_importances'].sum() > 0:
        result['feature_importances'] /= result['feature_importances'].sum()

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

def _fit_single_tree(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    max_depth: int = -1,
    min_child_samples: int = 20,
    colsample_bytree: float = 1.0
) -> Dict[str, Any]:
    """
    Fit a single decision tree for LightGBM.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    sample_weight : np.ndarray
        Sample weights array of shape (n_samples,)
    max_depth : int, default=-1
        Maximum tree depth (-1 means no limit)
    min_child_samples : int, default=20
        Minimum number of samples per leaf
    colsample_bytree : float, default=1.0
        Fraction of features used for this tree

    Returns
    ------
    Dict[str, Any]
        Dictionary containing:
        - 'tree_structure': tree nodes and splits
        - 'importance': feature importances for this tree
    """
    # Simplified implementation of a single decision tree
    n_features = X.shape[1]
    if colsample_bytree < 1.0:
        n_features_used = int(n_features * colsample_bytree)
        feature_indices = np.random.choice(n_features, n_features_used, replace=False)
    else:
        feature_indices = np.arange(n_features)

    # Initialize tree structure
    tree_structure = {
        'nodes': [],
        'importance': np.zeros(n_features)
    }

    # Recursive tree building (simplified)
    def build_node(depth: int, indices: np.ndarray):
        if len(indices) < min_child_samples or (max_depth >= 0 and depth >= max_depth):
            # Create leaf node
            node = {
                'is_leaf': True,
                'indices': indices,
                'value': np.average(y[indices], weights=sample_weight[indices])
            }
            tree_structure['nodes'].append(node)
            return

        # Find best split (simplified)
        best_split = _find_best_split(X[indices], y[indices],
                                    sample_weight[indices], feature_indices)

        if best_split is None:
            # Create leaf node
            node = {
                'is_leaf': True,
                'indices': indices,
                'value': np.average(y[indices], weights=sample_weight[indices])
            }
            tree_structure['nodes'].append(node)
            return

        # Update feature importance
        tree_structure['importance'][best_split['feature']] += 1

        # Create split node
        node = {
            'is_leaf': False,
            'feature': best_split['feature'],
            'threshold': best_split['threshold'],
            'left_indices': indices[best_split['left_mask']],
            'right_indices': indices[~best_split['left_mask']]
        }
        tree_structure['nodes'].append(node)

        # Recursively build left and right children
        build_node(depth + 1, node['left_indices'])
        build_node(depth + 1, node['right_indices'])

    # Start building the tree
    build_node(0, np.arange(X.shape[0]))

    return {
        'tree_structure': tree_structure,
        'importance': tree_structure['importance']
    }

def _find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    feature_indices: np.ndarray
) -> Optional[Dict[str, Any]]:
    """
    Find the best split for a node.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    sample_weight : np.ndarray
        Sample weights array of shape (n_samples,)
    feature_indices : np.ndarray
        Indices of features to consider

    Returns
    ------
    Optional[Dict[str, Any]]
        Dictionary containing:
        - 'feature': best feature index
        - 'threshold': best threshold value
        - 'left_mask': mask for left child samples
        None if no good split is found
    """
    best_split = None
    best_score = float('inf')

    for feature_idx in feature_indices:
        # Get unique values and sort
        values = X[:, feature_idx]
        unique_values = np.unique(values)
        if len(unique_values) < 2:
            continue

        # Try all possible splits
        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i+1]) / 2
            left_mask = values <= threshold

            # Calculate weighted variance reduction
            left_weight = sample_weight[left_mask].sum()
            right_weight = sample_weight[~left_mask].sum()

            if left_weight == 0 or right_weight == 0:
                continue

            left_value = np.average(y[left_mask], weights=sample_weight[left_mask])
            right_value = np.average(y[~left_mask], weights=sample_weight[~left_mask])

            left_var = np.sum(sample_weight[left_mask] * (y[left_mask] - left_value) ** 2)
            right_var = np.sum(sample_weight[~left_mask] * (y[~left_mask] - right_value) ** 2)

            total_var = left_var + right_var
            weighted_var_reduction = (left_weight * right_weight / (left_weight + right_weight)) * total_var

            if weighted_var_reduction < best_score:
                best_score = weighted_var_reduction
                best_split = {
                    'feature': feature_idx,
                    'threshold': threshold,
                    'left_mask': left_mask
                }

    return best_split

def _predict(
    X: np.ndarray,
    trees: list
) -> np.ndarray:
    """
    Make predictions using the fitted LightGBM model.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    trees : list
        List of fitted tree structures

    Returns
    ------
    np.ndarray
        Predicted values array of shape (n_samples,)
    """
    y_pred = np.zeros(X.shape[0])

    for tree in trees:
        # Traverse each tree to get predictions
        for node in tree['tree_structure']['nodes']:
            if node['is_leaf']:
                y_pred[node['indices']] += node['value']
            else:
                mask = X[:, node['feature']] <= node['threshold']
                left_indices = np.where(mask)[0]
                right_indices = np.where(~mask)[0]

                # Recursively find leaf nodes
                _traverse_tree(tree['tree_structure'], X, y_pred,
                             left_indices, node['left_indices'])
                _traverse_tree(tree['tree_structure'], X, y_pred,
                             right_indices, node['right_indices'])

    return y_pred

def _traverse_tree(
    tree_structure: Dict[str, Any],
    X: np.ndarray,
    y_pred: np.ndarray,
    indices: np.ndarray,
    node_indices: np.ndarray
) -> None:
    """
    Recursively traverse the tree to find leaf nodes.

    Parameters
    ----------
    tree_structure : Dict[str, Any]
        Tree structure dictionary
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y_pred : np.ndarray
        Predicted values array to update
    indices : np.ndarray
        Current sample indices
    node_indices : np.ndarray
        Indices of nodes to consider
    """
    for i in range(len(tree_structure['nodes'])):
        if i not in node_indices:
            continue

        node = tree_structure['nodes'][i]
        if node['is_leaf']:
            y_pred[indices] += node['value']
        else:
            mask = X[indices, node['feature']] <= node['threshold']
            left_indices = indices[mask]
            right_indices = indices[~mask]

            if len(left_indices) > 0:
                _traverse_tree(tree_structure, X, y_pred,
                             left_indices, node['left_indices'])
            if len(right_indices) > 0:
                _traverse_tree(tree_structure, X, y_pred,
                             right_indices, node['right_indices'])

def _get_metric_function(metric_name: str) -> Callable:
    """
    Get metric function based on name.

    Parameters
    ----------
    metric_name : str
        Name of the metric

    Returns
    ------
    Callable
        Metric function

    Raises
    ------
    ValueError
        If metric name is not recognized
    """
    metrics = {
        'mse': default_metric,
        # Add other metric functions as needed
    }

    if metric_name not in metrics:
        raise ValueError(f"Unknown metric: {metric_name}")

    return metrics[metric_name]

################################################################################
# catboost
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
    if normalization == "standard":
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_normalized = (X - X_mean) / (X_std + 1e-8)
        y_normalized = (y - np.mean(y)) / (np.std(y) + 1e-8)
    elif normalization == "minmax":
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_normalized = (X - X_min) / (X_max - X_min + 1e-8)
        y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-8)
    elif normalization == "robust":
        X_median = np.median(X, axis=0)
        X_q75 = np.percentile(X, 75, axis=0)
        X_q25 = np.percentile(X, 25, axis=0)
        X_normalized = (X - X_median) / (X_q75 - X_q25 + 1e-8)
        y_median = np.median(y)
        y_q75 = np.percentile(y, 75)
        y_q25 = np.percentile(y, 25)
        y_normalized = (y - y_median) / (y_q75 - y_q25 + 1e-8)
    else:
        X_normalized = X.copy()
        y_normalized = y.copy()

    return X_normalized, y_normalized

def _compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: Union[str, Callable]) -> float:
    """Compute evaluation metric."""
    if callable(metric):
        return metric(y_true, y_pred)
    elif metric == "mse":
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

def _gradient_descent(X: np.ndarray, y: np.ndarray,
                      learning_rate: float = 0.01,
                      n_iter: int = 1000,
                      tol: float = 1e-4) -> np.ndarray:
    """Gradient descent solver."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(n_iter):
        gradients = (1 / n_samples) * X.T @ (X @ weights + bias - y)
        weights -= learning_rate * gradients
        bias -= learning_rate * np.mean(X @ weights + bias - y)

        if np.linalg.norm(gradients) < tol:
            break

    return np.concatenate([weights, [bias]])

def _newton_method(X: np.ndarray, y: np.ndarray,
                   n_iter: int = 100,
                   tol: float = 1e-4) -> np.ndarray:
    """Newton method solver."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features + 1)

    for _ in range(n_iter):
        predictions = X @ weights[:-1] + weights[-1]
        residuals = y - predictions

        # Compute Hessian and gradient
        hessian = (1 / n_samples) * X.T @ X
        gradient = -(1 / n_samples) * X.T @ residuals

        # Update weights using Newton's method
        delta = np.linalg.solve(hessian, gradient)
        weights -= delta

        if np.linalg.norm(delta) < tol:
            break

    return weights

def _coordinate_descent(X: np.ndarray, y: np.ndarray,
                        n_iter: int = 100,
                        tol: float = 1e-4) -> np.ndarray:
    """Coordinate descent solver."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features + 1)

    for _ in range(n_iter):
        old_weights = weights.copy()

        for i in range(n_features + 1):
            if i < n_features:
                X_i = X[:, i]
            else:
                X_i = np.ones(n_samples)

            # Compute residual without current feature
            residual = y - (X @ weights[:-1] + weights[-1])

            # Compute correlation
            corr = X_i.T @ residual

            # Update weight
            if i < n_features:
                weights[i] = corr / (X_i.T @ X_i + 1e-8)
            else:
                weights[-1] = np.mean(residual)

        if np.linalg.norm(weights - old_weights) < tol:
            break

    return weights

def catboost_fit(X: np.ndarray, y: np.ndarray,
                 normalization: str = "standard",
                 metric: Union[str, Callable] = "mse",
                 solver: str = "gradient_descent",
                 learning_rate: float = 0.01,
                 n_iter: int = 1000,
                 tol: float = 1e-4) -> Dict:
    """
    Fit a CatBoost model with hierarchical structure.

    Parameters:
    -----------
    X : np.ndarray
        Input features (n_samples, n_features)
    y : np.ndarray
        Target values (n_samples,)
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable
        Evaluation metric ('mse', 'mae', 'r2', 'logloss') or custom function
    solver : str
        Solver method ('gradient_descent', 'newton', 'coordinate_descent')
    learning_rate : float
        Learning rate for gradient descent (default: 0.01)
    n_iter : int
        Maximum number of iterations (default: 1000)
    tol : float
        Tolerance for convergence (default: 1e-4)

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = catboost_fit(X, y, normalization="standard", metric="mse")
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = _normalize_data(X, y, normalization)

    # Choose solver
    if solver == "gradient_descent":
        weights = _gradient_descent(X_norm, y_norm,
                                   learning_rate=learning_rate,
                                   n_iter=n_iter,
                                   tol=tol)
    elif solver == "newton":
        weights = _newton_method(X_norm, y_norm,
                                n_iter=n_iter,
                                tol=tol)
    elif solver == "coordinate_descent":
        weights = _coordinate_descent(X_norm, y_norm,
                                     n_iter=n_iter,
                                     tol=tol)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Make predictions
    y_pred = X_norm @ weights[:-1] + weights[-1]

    # Compute metrics
    main_metric = _compute_metric(y_norm, y_pred, metric)

    # Prepare output
    result = {
        "result": {
            "weights": weights,
            "predictions": y_pred
        },
        "metrics": {
            "main_metric": main_metric,
            "mse": _compute_metric(y_norm, y_pred, "mse"),
            "mae": _compute_metric(y_norm, y_pred, "mae")
        },
        "params_used": {
            "normalization": normalization,
            "metric": metric,
            "solver": solver,
            "learning_rate": learning_rate,
            "n_iter": n_iter,
            "tol": tol
        },
        "warnings": []
    }

    return result

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
    max_features: Union[int, float] = "auto",
    bootstrap: bool = True,
    criterion: str = "mse",
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = None,
    normalization: str = "none",
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    random_state: Optional[int] = None,
    n_jobs: int = -1
) -> Dict[str, Any]:
    """
    Fit a random forest model to the given data.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    n_estimators : int, optional
        Number of trees in the forest. Default is 100.
    max_depth : int, optional
        Maximum depth of the tree. If None, nodes are expanded until all leaves are pure.
    min_samples_split : int, optional
        Minimum number of samples required to split an internal node. Default is 2.
    min_samples_leaf : int, optional
        Minimum number of samples required to be at a leaf node. Default is 1.
    max_features : int or float, optional
        Number of features to consider when looking for the best split. Default is "auto".
    bootstrap : bool, optional
        Whether bootstrap samples are used when building trees. Default is True.
    criterion : str, optional
        Function to measure the quality of a split. Supported criteria are "mse" for regression and "gini" or "entropy" for classification. Default is "mse".
    distance_metric : Callable, optional
        Custom distance metric function. Default is None.
    normalization : str, optional
        Normalization method for features. Supported methods are "none", "standard", "minmax", and "robust". Default is "none".
    metric : str or Callable, optional
        Metric to evaluate the quality of the split. Default is "mse".
    random_state : int, optional
        Controls both the randomness of the bootstrapping of the samples and the sampling of the features. Default is None.
    n_jobs : int, optional
        Number of jobs to run in parallel. Default is -1 (all available cores).

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the fitted model, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if specified
    X_normalized = _normalize_features(X, normalization)

    # Initialize random forest parameters
    params_used = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "bootstrap": bootstrap,
        "criterion": criterion,
        "distance_metric": distance_metric.__name__ if callable(distance_metric) else distance_metric,
        "normalization": normalization,
        "metric": metric if isinstance(metric, str) else metric.__name__,
        "random_state": random_state,
        "n_jobs": n_jobs
    }

    # Fit the random forest model
    trees = []
    for i in range(n_estimators):
        tree = _fit_single_tree(
            X_normalized, y,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            criterion=criterion,
            distance_metric=distance_metric,
            random_state=random_state if random_state is not None else np.random.randint(0, 1000)
        )
        trees.append(tree)

    # Calculate metrics
    y_pred = _predict(trees, X_normalized)
    metrics = _calculate_metrics(y, y_pred, metric)

    # Return results
    return {
        "result": {"trees": trees},
        "metrics": metrics,
        "params_used": params_used,
        "warnings": []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate the input data.

    Parameters:
    -----------
    X : np.ndarray
        Input features.
    y : np.ndarray
        Target values.

    Raises:
    -------
    ValueError
        If the input data is invalid.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("The number of samples in X and y must be the same.")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains NaN or infinite values.")
    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("y contains NaN or infinite values.")

def _normalize_features(X: np.ndarray, method: str) -> np.ndarray:
    """
    Normalize the features.

    Parameters:
    -----------
    X : np.ndarray
        Input features.
    method : str
        Normalization method.

    Returns:
    --------
    np.ndarray
        Normalized features.
    """
    if method == "standard":
        X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == "minmax":
        X_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == "robust":
        X_normalized = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        X_normalized = X
    return X_normalized

def _fit_single_tree(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: Optional[int],
    min_samples_split: int,
    min_samples_leaf: int,
    max_features: Union[int, float],
    bootstrap: bool,
    criterion: str,
    distance_metric: Callable[[np.ndarray, np.ndarray], float],
    random_state: int
) -> Dict[str, Any]:
    """
    Fit a single decision tree.

    Parameters:
    -----------
    X : np.ndarray
        Input features.
    y : np.ndarray
        Target values.
    max_depth : int, optional
        Maximum depth of the tree.
    min_samples_split : int
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int
        Minimum number of samples required to be at a leaf node.
    max_features : int or float
        Number of features to consider when looking for the best split.
    bootstrap : bool
        Whether bootstrap samples are used when building trees.
    criterion : str
        Function to measure the quality of a split.
    distance_metric : Callable
        Custom distance metric function.
    random_state : int
        Controls the randomness of the bootstrapping.

    Returns:
    --------
    Dict[str, Any]
        The fitted decision tree.
    """
    # Implementation of fitting a single decision tree
    pass

def _predict(trees: list, X: np.ndarray) -> np.ndarray:
    """
    Predict using the fitted random forest.

    Parameters:
    -----------
    trees : list
        List of fitted decision trees.
    X : np.ndarray
        Input features.

    Returns:
    --------
    np.ndarray
        Predicted values.
    """
    # Implementation of prediction using the random forest
    pass

def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, metric: Union[str, Callable]) -> Dict[str, float]:
    """
    Calculate the metrics for the predictions.

    Parameters:
    -----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.
    metric : str or Callable
        Metric to evaluate the quality of the predictions.

    Returns:
    --------
    Dict[str, float]
        Dictionary of calculated metrics.
    """
    if isinstance(metric, str):
        if metric == "mse":
            return {"mse": np.mean((y_true - y_pred) ** 2)}
        elif metric == "mae":
            return {"mae": np.mean(np.abs(y_true - y_pred))}
        elif metric == "r2":
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return {"r2": 1 - (ss_res / ss_tot)}
    else:
        return {metric.__name__: metric(y_true, y_pred)}

# Example usage
if __name__ == "__main__":
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    result = random_forest_fit(X, y, n_estimators=10, max_depth=3)
    print(result)

################################################################################
# bagging
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
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

def _normalize_data(X: np.ndarray, y: np.ndarray,
                   normalization: str = 'standard',
                   custom_normalization: Optional[Callable] = None) -> tuple:
    """Normalize data according to specified method."""
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

def _compute_metric(y_true: np.ndarray, y_pred: np.ndarray,
                   metric: str = 'mse',
                   custom_metric: Optional[Callable] = None) -> float:
    """Compute specified metric between true and predicted values."""
    if custom_metric is not None:
        return custom_metric(y_true, y_pred)
    elif metric == 'mse':
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
        raise ValueError(f"Unknown metric: {metric}")

def _fit_base_model(X: np.ndarray, y: np.ndarray,
                   solver: str = 'closed_form',
                   custom_solver: Optional[Callable] = None,
                   **solver_params) -> Callable:
    """Fit a base model using specified solver."""
    if custom_solver is not None:
        return custom_solver(X, y)
    elif solver == 'closed_form':
        X_tx = np.dot(X.T, X)
        if np.linalg.det(X_tx) == 0:
            raise ValueError("Matrix is singular, cannot compute closed form solution")
        beta = np.linalg.solve(X_tx, np.dot(X.T, y))
        return lambda x: np.dot(x, beta)
    elif solver == 'gradient_descent':
        from sklearn.linear_model import SGDRegressor
        model = SGDRegressor(**solver_params)
        model.fit(X, y)
        return lambda x: model.predict(x)
    elif solver == 'newton':
        from sklearn.linear_model import Ridge
        model = Ridge(**solver_params)
        model.fit(X, y)
        return lambda x: model.predict(x)
    elif solver == 'coordinate_descent':
        from sklearn.linear_model import Lasso
        model = Lasso(**solver_params)
        model.fit(X, y)
        return lambda x: model.predict(x)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def bagging_fit(X: np.ndarray, y: np.ndarray,
               n_estimators: int = 10,
               max_samples: float = 1.0,
               max_features: float = 1.0,
               normalization: str = 'standard',
               metric: str = 'mse',
               solver: str = 'closed_form',
               random_state: Optional[int] = None,
               custom_normalization: Optional[Callable] = None,
               custom_metric: Optional[Callable] = None,
               custom_solver: Optional[Callable] = None,
               **solver_params) -> Dict[str, Any]:
    """
    Bagging ensemble method for hierarchical modeling.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    n_estimators : int, optional
        Number of base estimators in the ensemble (default: 10)
    max_samples : float, optional
        Fraction of samples to draw for each base estimator (default: 1.0)
    max_features : float, optional
        Fraction of features to draw for each base estimator (default: 1.0)
    normalization : str or callable, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') or custom function (default: 'standard')
    metric : str or callable, optional
        Evaluation metric ('mse', 'mae', 'r2', 'logloss') or custom function (default: 'mse')
    solver : str or callable, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent') or custom function (default: 'closed_form')
    random_state : int, optional
        Random seed for reproducibility (default: None)
    custom_normalization : callable, optional
        Custom normalization function
    custom_metric : callable, optional
        Custom metric function
    custom_solver : callable, optional
        Custom solver function
    **solver_params : dict
        Additional parameters for the base estimator solver

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - 'result': list of base estimators
        - 'metrics': dictionary of evaluation metrics
        - 'params_used': dictionary of parameters used
        - 'warnings': list of warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = bagging_fit(X, y, n_estimators=5, metric='r2')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Initialize random number generator
    rng = np.random.RandomState(random_state)

    # Normalize data
    X_norm, y_norm = _normalize_data(X, y, normalization, custom_normalization)

    # Initialize results
    estimators = []
    metrics = {'train': [], 'oob': []}
    warnings = []

    # Train base estimators
    for _ in range(n_estimators):
        # Sample data with replacement
        sample_indices = rng.choice(X_norm.shape[0], size=int(max_samples * X_norm.shape[0]), replace=True)
        oob_indices = np.setdiff1d(np.arange(X_norm.shape[0]), sample_indices)

        X_sample = X_norm[sample_indices]
        y_sample = y_norm[sample_indices]

        # Sample features
        if max_features < 1.0:
            feature_indices = rng.choice(X_sample.shape[1], size=int(max_features * X_sample.shape[1]), replace=False)
            X_sample = X_sample[:, feature_indices]

        # Fit base model
        try:
            estimator = _fit_base_model(X_sample, y_sample, solver, custom_solver, **solver_params)
            estimators.append((estimator, feature_indices if max_features < 1.0 else None))
        except Exception as e:
            warnings.append(f"Failed to fit estimator: {str(e)}")
            continue

        # Compute metrics
        if len(oob_indices) > 0:
            X_oob = X_norm[oob_indices]
            if max_features < 1.0:
                X_oob = X_oob[:, feature_indices]
            y_pred = estimator(X_oob)
            metrics['oob'].append(_compute_metric(y_norm[oob_indices], y_pred, metric, custom_metric))

        y_pred_train = estimator(X_sample)
        metrics['train'].append(_compute_metric(y_sample, y_pred_train, metric, custom_metric))

    # Prepare output
    return {
        'result': estimators,
        'metrics': metrics,
        'params_used': {
            'n_estimators': n_estimators,
            'max_samples': max_samples,
            'max_features': max_features,
            'normalization': normalization if custom_normalization is None else 'custom',
            'metric': metric if custom_metric is None else 'custom',
            'solver': solver if custom_solver is None else 'custom',
            **solver_params
        },
        'warnings': warnings
    }

################################################################################
# boosting
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def boosting_fit(
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
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict:
    """
    Fit a boosting model to the data.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    n_estimators : int, optional
        Number of boosting iterations.
    learning_rate : float, optional
        Shrinkage factor for each estimator.
    loss : str, optional
        Loss function to optimize ('mse', 'mae', 'huber').
    metric : str or callable, optional
        Metric to evaluate the model ('mse', 'mae', 'r2').
    normalizer : callable, optional
        Function to normalize the data.
    solver : str, optional
        Solver method ('gradient_descent', 'newton').
    regularization : str, optional
        Regularization type ('l1', 'l2', 'elasticnet').
    tol : float, optional
        Tolerance for early stopping.
    max_iter : int, optional
        Maximum number of iterations.
    random_state : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    Dict containing the fitted model, metrics, parameters used and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Initialize model components
    n_samples = X.shape[0]
    weights = np.ones(n_samples) / n_samples
    estimators = []
    metrics_history = []

    # Normalize data if specified
    X_norm, y_norm = _apply_normalization(X, y, normalizer)

    # Select loss and metric functions
    loss_func = _get_loss_function(loss)
    metric_func = _get_metric_function(metric)

    # Initialize parameters
    params_used = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'loss': loss,
        'metric': metric,
        'solver': solver,
        'regularization': regularization
    }

    # Boosting iterations
    for _ in range(n_estimators):
        # Compute residuals
        residuals = y_norm - _predict(X_norm, estimators)

        # Fit weak learner
        estimator = _fit_weak_learner(
            X_norm,
            residuals,
            solver=solver,
            regularization=regularization
        )

        # Update estimators with learning rate
        estimator *= learning_rate
        estimators.append(estimator)

        # Compute metric
        current_pred = _predict(X_norm, estimators)
        current_metric = metric_func(y_norm, current_pred)
        metrics_history.append(current_metric)

        # Check for early stopping
        if len(metrics_history) > 1 and abs(metrics_history[-2] - current_metric) < tol:
            break

    # Prepare output
    result = {
        'estimators': estimators,
        'final_prediction': _predict(X_norm, estimators)
    }

    return {
        'result': result,
        'metrics': {'history': metrics_history, 'final': metrics_history[-1]},
        'params_used': params_used,
        'warnings': _check_warnings(metrics_history)
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input data contains NaN values")

def _apply_normalization(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable]
) -> tuple:
    """Apply normalization to data if specified."""
    X_norm = X.copy()
    y_norm = y.copy()

    if normalizer is not None:
        X_norm = normalizer(X)
        y_norm = normalizer(y.reshape(-1, 1)).flatten()

    return X_norm, y_norm

def _get_loss_function(loss: str) -> Callable:
    """Return the specified loss function."""
    if loss == 'mse':
        return _mse_loss
    elif loss == 'mae':
        return _mae_loss
    elif loss == 'huber':
        return _huber_loss
    else:
        raise ValueError(f"Unknown loss function: {loss}")

def _get_metric_function(metric: Union[str, Callable]) -> Callable:
    """Return the specified metric function."""
    if isinstance(metric, str):
        if metric == 'mse':
            return _mse_metric
        elif metric == 'mae':
            return _mae_metric
        elif metric == 'r2':
            return _r2_metric
        else:
            raise ValueError(f"Unknown metric: {metric}")
    return metric

def _fit_weak_learner(
    X: np.ndarray,
    y: np.ndarray,
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None
) -> np.ndarray:
    """Fit a weak learner using the specified solver."""
    if solver == 'gradient_descent':
        return _gradient_descent(X, y, regularization)
    elif solver == 'newton':
        return _newton_method(X, y, regularization)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _predict(X: np.ndarray, estimators: list) -> np.ndarray:
    """Make predictions using the ensemble of estimators."""
    return sum(estimator.predict(X) for estimator in estimators)

def _check_warnings(metrics_history: list) -> list:
    """Check for potential warnings during training."""
    warnings = []
    if len(metrics_history) < 10:
        warnings.append("Training stopped early")
    return warnings

# Example loss functions
def _mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)

def _mae_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))

def _huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
    error = y_true - y_pred
    is_small_error = np.abs(error) < delta
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.mean(np.where(is_small_error, squared_loss, linear_loss))

# Example metric functions
def _mse_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return _mse_loss(y_true, y_pred)

def _mae_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return _mae_loss(y_true, y_pred)

def _r2_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Example solver functions
def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str] = None
) -> 'BaseEstimator':
    # Placeholder for actual implementation
    class DummyEstimator:
        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.zeros(X.shape[0])
    return DummyEstimator()

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str] = None
) -> 'BaseEstimator':
    # Placeholder for actual implementation
    class DummyEstimator:
        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.zeros(X.shape[0])
    return DummyEstimator()

################################################################################
# stacking
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def stacking_fit(
    base_models: list,
    meta_model: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    normalize: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None,
    custom_normalize: Optional[Callable] = None,
    custom_metric: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Fit a stacking ensemble model.

    Parameters:
    - base_models: List of trained base models (callables with predict method)
    - meta_model: Callable for the meta-model
    - X_train: Training features
    - y_train: Training targets
    - X_val: Validation features (optional)
    - y_val: Validation targets (optional)
    - normalize: Normalization method ('none', 'standard', 'minmax', 'robust')
    - metric: Evaluation metric ('mse', 'mae', 'r2', 'logloss') or custom callable
    - solver: Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    - regularization: Regularization type ('none', 'l1', 'l2', 'elasticnet')
    - tol: Tolerance for convergence
    - max_iter: Maximum iterations
    - random_state: Random seed
    - custom_normalize: Custom normalization function (optional)
    - custom_metric: Custom metric function (optional)

    Returns:
    - Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(X_train, y_train, X_val, y_val)

    # Normalize data
    X_train_norm, X_val_norm = _normalize_data(X_train, X_val, normalize, custom_normalize)

    # Generate meta-features
    meta_features_train = _generate_meta_features(base_models, X_train_norm)
    if X_val is not None:
        meta_features_val = _generate_meta_features(base_models, X_val_norm)
    else:
        meta_features_val = None

    # Fit meta-model
    meta_model_params = _fit_meta_model(
        meta_features_train,
        y_train,
        X_val=meta_features_val,
        y_val=y_val,
        metric=metric,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state,
        custom_metric=custom_metric
    )

    # Calculate metrics
    metrics = _calculate_metrics(
        meta_model,
        meta_features_train,
        y_train,
        X_val=meta_features_val,
        y_val=y_val,
        metric=metric,
        custom_metric=custom_metric
    )

    return {
        'result': meta_model,
        'metrics': metrics,
        'params_used': {
            'normalize': normalize,
            'metric': metric,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

def _validate_inputs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray]
) -> None:
    """Validate input dimensions and types."""
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train must have the same number of samples")
    if X_val is not None and (X_val.shape[0] != y_val.shape[0]):
        raise ValueError("X_val and y_val must have the same number of samples")
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        raise ValueError("X_train contains NaN or Inf values")
    if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
        raise ValueError("y_train contains NaN or Inf values")
    if X_val is not None and (np.any(np.isnan(X_val)) or np.any(np.isinf(X_val))):
        raise ValueError("X_val contains NaN or Inf values")
    if y_val is not None and (np.any(np.isnan(y_val)) or np.any(np.isinf(y_val))):
        raise ValueError("y_val contains NaN or Inf values")

def _normalize_data(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray],
    method: str,
    custom_func: Optional[Callable] = None
) -> tuple:
    """Normalize data using specified method."""
    if custom_func is not None:
        X_train_norm = custom_func(X_train)
        if X_val is not None:
            X_val_norm = custom_func(X_val)
        else:
            X_val_norm = None
    elif method == 'standard':
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        X_train_norm = (X_train - mean) / std
        if X_val is not None:
            X_val_norm = (X_val - mean) / std
        else:
            X_val_norm = None
    elif method == 'minmax':
        min_val = np.min(X_train, axis=0)
        max_val = np.max(X_train, axis=0)
        X_train_norm = (X_train - min_val) / (max_val - min_val)
        if X_val is not None:
            X_val_norm = (X_val - min_val) / (max_val - min_val)
        else:
            X_val_norm = None
    elif method == 'robust':
        median = np.median(X_train, axis=0)
        iqr = np.subtract(*np.percentile(X_train, [75, 25], axis=0))
        X_train_norm = (X_train - median) / iqr
        if X_val is not None:
            X_val_norm = (X_val - median) / iqr
        else:
            X_val_norm = None
    elif method == 'none':
        X_train_norm = X_train.copy()
        if X_val is not None:
            X_val_norm = X_val.copy()
        else:
            X_val_norm = None
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return X_train_norm, X_val_norm

def _generate_meta_features(
    base_models: list,
    X: np.ndarray
) -> np.ndarray:
    """Generate meta-features using base models."""
    meta_features = []
    for model in base_models:
        if hasattr(model, 'predict'):
            meta_features.append(model.predict(X))
        else:
            raise ValueError("Base models must have a predict method")
    return np.column_stack(meta_features)

def _fit_meta_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None
) -> Dict[str, Any]:
    """Fit meta-model with specified parameters."""
    if solver == 'closed_form':
        params = _fit_closed_form(X_train, y_train)
    elif solver == 'gradient_descent':
        params = _fit_gradient_descent(
            X_train, y_train,
            metric=metric,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            custom_metric=custom_metric
        )
    elif solver == 'newton':
        params = _fit_newton(
            X_train, y_train,
            metric=metric,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            custom_metric=custom_metric
        )
    elif solver == 'coordinate_descent':
        params = _fit_coordinate_descent(
            X_train, y_train,
            metric=metric,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            custom_metric=custom_metric
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    def meta_model(X: np.ndarray) -> np.ndarray:
        return _predict_meta_model(params, X)

    return {'params': params}

def _fit_closed_form(
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, Any]:
    """Fit model using closed-form solution."""
    XtX = np.dot(X.T, X)
    if not np.allclose(np.linalg.det(XtX), 0):
        XtX_inv = np.linalg.inv(XtX)
    else:
        raise ValueError("Matrix is singular")
    Xty = np.dot(X.T, y)
    coef = np.dot(XtX_inv, Xty)
    intercept = np.mean(y) - np.dot(np.mean(X, axis=0), coef)
    return {'coef': coef, 'intercept': intercept}

def _fit_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable] = 'mse',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None
) -> Dict[str, Any]:
    """Fit model using gradient descent."""
    np.random.seed(random_state)
    n_features = X.shape[1]
    coef = np.random.randn(n_features)
    intercept = 0.0
    prev_loss = float('inf')

    for _ in range(max_iter):
        gradients = _compute_gradient(X, y, coef, intercept)
        if regularization == 'l1':
            gradients += np.sign(coef)  # L1 penalty
        elif regularization == 'l2':
            gradients += 2 * coef  # L2 penalty
        elif regularization == 'elasticnet':
            gradients += np.sign(coef) + 2 * coef  # ElasticNet penalty

        coef -= gradients
        intercept -= np.mean(y - _predict_meta_model({'coef': coef, 'intercept': intercept}, X))

        current_loss = _compute_metric(
            y,
            _predict_meta_model({'coef': coef, 'intercept': intercept}, X),
            metric,
            custom_metric
        )

        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return {'coef': coef, 'intercept': intercept}

def _fit_newton(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable] = 'mse',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None
) -> Dict[str, Any]:
    """Fit model using Newton's method."""
    np.random.seed(random_state)
    n_features = X.shape[1]
    coef = np.random.randn(n_features)
    intercept = 0.0
    prev_loss = float('inf')

    for _ in range(max_iter):
        hessian = _compute_hessian(X, y, coef, intercept)
        if regularization == 'l2':
            hessian += 2 * np.eye(n_features)  # L2 penalty

        gradient = _compute_gradient(X, y, coef, intercept)
        if regularization == 'l1':
            gradient += np.sign(coef)  # L1 penalty
        elif regularization == 'elasticnet':
            gradient += np.sign(coef) + 2 * coef  # ElasticNet penalty

        hessian_inv = np.linalg.inv(hessian)
        coef -= np.dot(hessian_inv, gradient)
        intercept -= np.mean(y - _predict_meta_model({'coef': coef, 'intercept': intercept}, X))

        current_loss = _compute_metric(
            y,
            _predict_meta_model({'coef': coef, 'intercept': intercept}, X),
            metric,
            custom_metric
        )

        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return {'coef': coef, 'intercept': intercept}

def _fit_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable] = 'mse',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None
) -> Dict[str, Any]:
    """Fit model using coordinate descent."""
    np.random.seed(random_state)
    n_features = X.shape[1]
    coef = np.zeros(n_features)
    intercept = 0.0
    prev_loss = float('inf')

    for _ in range(max_iter):
        for i in range(n_features):
            X_i = X[:, i]
            residuals = y - intercept - np.dot(X, coef) + coef[i] * X_i

            if regularization == 'l1':
                coef[i] = _soft_threshold(
                    np.dot(X_i, residuals) / np.sum(X_i**2),
                    1.0
                )
            elif regularization == 'l2':
                coef[i] = np.dot(X_i, residuals) / (np.sum(X_i**2) + 2)
            elif regularization == 'elasticnet':
                coef[i] = _soft_threshold(
                    np.dot(X_i, residuals) / (np.sum(X_i**2) + 2),
                    1.0
                )
            else:
                coef[i] = np.dot(X_i, residuals) / np.sum(X_i**2)

        intercept = np.mean(y - np.dot(X, coef))

        current_loss = _compute_metric(
            y,
            _predict_meta_model({'coef': coef, 'intercept': intercept}, X),
            metric,
            custom_metric
        )

        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return {'coef': coef, 'intercept': intercept}

def _predict_meta_model(
    params: Dict[str, Any],
    X: np.ndarray
) -> np.ndarray:
    """Predict using meta-model."""
    return params['intercept'] + np.dot(X, params['coef'])

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    coef: np.ndarray,
    intercept: float
) -> np.ndarray:
    """Compute gradient for meta-model."""
    residuals = y - intercept - np.dot(X, coef)
    return -np.dot(X.T, residuals) / X.shape[0]

def _compute_hessian(
    X: np.ndarray,
    y: np.ndarray,
    coef: np.ndarray,
    intercept: float
) -> np.ndarray:
    """Compute Hessian for meta-model."""
    return np.dot(X.T, X) / X.shape[0]

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable] = None
) -> float:
    """Compute specified metric."""
    if custom_metric is not None:
        return custom_metric(y_true, y_pred)
    elif metric == 'mse':
        return np.mean((y_true - y_pred)**2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - ss_res / ss_tot
    elif metric == 'logloss':
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _calculate_metrics(
    model: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    metric: Union[str, Callable] = 'mse',
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """Calculate metrics for training and validation data."""
    train_pred = model(X_train)
    train_metric = _compute_metric(y_train, train_pred, metric, custom_metric)

    metrics = {'train': train_metric}

    if X_val is not None and y_val is not None:
        val_pred = model(X_val)
        val_metric = _compute_metric(y_val, val_pred, metric, custom_metric)
        metrics['validation'] = val_metric

    return metrics

def _soft_threshold(
    value: float,
    lambda_: float
) -> float:
    """Soft thresholding operator."""
    if value > lambda_:
        return value - lambda_
    elif value < -lambda_:
        return value + lambda_
    else:
        return 0.0

################################################################################
# hierarchical_clustering
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def hierarchical_clustering_fit(
    data: np.ndarray,
    distance_metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    linkage_method: str = 'ward',
    normalization: Optional[str] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    max_clusters: int = None,
    min_cluster_size: int = 1,
    threshold: float = 0.0
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Perform hierarchical clustering on the input data.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    distance_metric : str or callable, optional
        Distance metric to use. Can be 'euclidean', 'manhattan', 'cosine',
        or a custom callable function.
    linkage_method : str, optional
        Linkage method to use. Can be 'ward', 'complete', 'average',
        or 'single'.
    normalization : str, optional
        Normalization method to apply. Can be 'standard', 'minmax',
        or 'robust'.
    custom_distance : callable, optional
        Custom distance function if not using built-in metrics.
    max_clusters : int, optional
        Maximum number of clusters to form. If None, all samples are clustered.
    min_cluster_size : int, optional
        Minimum size of a cluster. Default is 1.
    threshold : float, optional
        Threshold for stopping the clustering process.

    Returns
    -------
    dict
        A dictionary containing:
        - 'result': The clustering result (linkage matrix).
        - 'metrics': Dictionary of computed metrics.
        - 'params_used': Parameters used in the clustering.
        - 'warnings': Any warnings generated during the process.

    Examples
    --------
    >>> data = np.random.rand(10, 5)
    >>> result = hierarchical_clustering_fit(data, distance_metric='euclidean', linkage_method='ward')
    """
    # Validate inputs
    _validate_inputs(data, distance_metric, linkage_method, normalization)

    # Normalize data if specified
    normalized_data = _apply_normalization(data, normalization)

    # Compute distance matrix
    if custom_distance is not None:
        distance_matrix = _compute_custom_distance(normalized_data, custom_distance)
    else:
        distance_matrix = _compute_distance_matrix(normalized_data, distance_metric)

    # Perform hierarchical clustering
    linkage_matrix = _perform_hierarchical_clustering(
        distance_matrix, linkage_method, max_clusters, min_cluster_size, threshold
    )

    # Compute metrics
    metrics = _compute_metrics(linkage_matrix)

    # Prepare output
    result = {
        'result': linkage_matrix,
        'metrics': metrics,
        'params_used': {
            'distance_metric': distance_metric if custom_distance is None else 'custom',
            'linkage_method': linkage_method,
            'normalization': normalization,
            'max_clusters': max_clusters,
            'min_cluster_size': min_cluster_size,
            'threshold': threshold
        },
        'warnings': []
    }

    return result

def _validate_inputs(
    data: np.ndarray,
    distance_metric: Union[str, Callable],
    linkage_method: str,
    normalization: Optional[str]
) -> None:
    """Validate the input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array.")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values.")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values.")

    valid_distance_metrics = ['euclidean', 'manhattan', 'cosine']
    if isinstance(distance_metric, str) and distance_metric not in valid_distance_metrics:
        raise ValueError(f"Invalid distance metric. Must be one of {valid_distance_metrics} or a custom callable.")

    valid_linkage_methods = ['ward', 'complete', 'average', 'single']
    if linkage_method not in valid_linkage_methods:
        raise ValueError(f"Invalid linkage method. Must be one of {valid_linkage_methods}.")

    valid_normalizations = ['standard', 'minmax', 'robust']
    if normalization is not None and normalization not in valid_normalizations:
        raise ValueError(f"Invalid normalization method. Must be one of {valid_normalizations} or None.")

def _apply_normalization(
    data: np.ndarray,
    normalization: Optional[str]
) -> np.ndarray:
    """Apply the specified normalization to the data."""
    if normalization is None:
        return data

    if normalization == 'standard':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized_data = (data - mean) / (std + 1e-8)
    elif normalization == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        normalized_data = (data - min_val) / (max_val - min_val + 1e-8)
    elif normalization == 'robust':
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        normalized_data = (data - median) / (iqr + 1e-8)

    return normalized_data

def _compute_distance_matrix(
    data: np.ndarray,
    distance_metric: str
) -> np.ndarray:
    """Compute the distance matrix using the specified metric."""
    n_samples = data.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i, n_samples):
            if distance_metric == 'euclidean':
                dist = np.linalg.norm(data[i] - data[j])
            elif distance_metric == 'manhattan':
                dist = np.sum(np.abs(data[i] - data[j]))
            elif distance_metric == 'cosine':
                dist = 1 - np.dot(data[i], data[j]) / (np.linalg.norm(data[i]) * np.linalg.norm(data[j]))
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix

def _compute_custom_distance(
    data: np.ndarray,
    custom_distance: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """Compute the distance matrix using a custom distance function."""
    n_samples = data.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i, n_samples):
            dist = custom_distance(data[i], data[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix

def _perform_hierarchical_clustering(
    distance_matrix: np.ndarray,
    linkage_method: str,
    max_clusters: Optional[int],
    min_cluster_size: int,
    threshold: float
) -> np.ndarray:
    """Perform hierarchical clustering using the specified linkage method."""
    n_samples = distance_matrix.shape[0]
    linkage_matrix = np.zeros((n_samples - 1, 4))

    # Initialize clusters
    clusters = [[i] for i in range(n_samples)]

    current_cluster_id = n_samples
    while len(clusters) > 1 and (max_clusters is None or len(clusters) > max_clusters):
        # Find the closest clusters
        min_distance = np.inf
        cluster1, cluster2 = None, None

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if linkage_method == 'ward':
                    distance = _compute_ward_distance(distance_matrix, clusters[i], clusters[j])
                elif linkage_method == 'complete':
                    distance = _compute_complete_distance(distance_matrix, clusters[i], clusters[j])
                elif linkage_method == 'average':
                    distance = _compute_average_distance(distance_matrix, clusters[i], clusters[j])
                elif linkage_method == 'single':
                    distance = _compute_single_distance(distance_matrix, clusters[i], clusters[j])

                if distance < min_distance:
                    min_distance = distance
                    cluster1, cluster2 = i, j

        if min_distance > threshold:
            break

        # Merge the closest clusters
        merged_cluster = clusters[cluster1] + clusters[cluster2]
        if len(merged_cluster) < min_cluster_size:
            break

        linkage_matrix[current_cluster_id - n_samples] = [
            cluster1, cluster2,
            len(clusters[cluster1]), len(clusters[cluster2]),
            min_distance
        ]

        clusters.pop(max(cluster1, cluster2))
        clusters.pop(min(cluster1, cluster2))
        clusters.append(merged_cluster)

        current_cluster_id += 1

    return linkage_matrix[:current_cluster_id - n_samples]

def _compute_ward_distance(
    distance_matrix: np.ndarray,
    cluster1: list,
    cluster2: list
) -> float:
    """Compute the Ward distance between two clusters."""
    n1 = len(cluster1)
    n2 = len(cluster2)

    sum1 = np.sum([distance_matrix[i, j] for i in cluster1 for j in cluster1])
    sum2 = np.sum([distance_matrix[i, j] for i in cluster2 for j in cluster2])
    sum12 = np.sum([distance_matrix[i, j] for i in cluster1 for j in cluster2])

    return (sum1 / n1 + sum2 / n2 - 2 * sum12 / (n1 + n2)) * (n1 + n2) / (n1 * n2)

def _compute_complete_distance(
    distance_matrix: np.ndarray,
    cluster1: list,
    cluster2: list
) -> float:
    """Compute the complete linkage distance between two clusters."""
    return np.max([distance_matrix[i, j] for i in cluster1 for j in cluster2])

def _compute_average_distance(
    distance_matrix: np.ndarray,
    cluster1: list,
    cluster2: list
) -> float:
    """Compute the average linkage distance between two clusters."""
    return np.mean([distance_matrix[i, j] for i in cluster1 for j in cluster2])

def _compute_single_distance(
    distance_matrix: np.ndarray,
    cluster1: list,
    cluster2: list
) -> float:
    """Compute the single linkage distance between two clusters."""
    return np.min([distance_matrix[i, j] for i in cluster1 for j in cluster2])

def _compute_metrics(
    linkage_matrix: np.ndarray
) -> Dict[str, float]:
    """Compute metrics for the clustering result."""
    if linkage_matrix.size == 0:
        return {}

    n_clusters = len(linkage_matrix) + 1
    metrics = {
        'n_clusters': n_clusters,
        'max_distance': np.max(linkage_matrix[:, 4]),
        'min_distance': np.min(linkage_matrix[:, 4])
    }

    return metrics

################################################################################
# agglomerative_clustering
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def agglomerative_clustering_fit(
    data: np.ndarray,
    n_clusters: int = None,
    linkage_criterion: str = 'ward',
    distance_metric: Union[str, Callable] = 'euclidean',
    affinity: str = 'euclidean',
    compute_full_tree: bool = False,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Perform agglomerative clustering on the input data.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_clusters : int, optional
        Number of clusters to form. If None, the full hierarchy is returned.
    linkage_criterion : str, optional
        Linkage criterion to compute distances between clusters. Options: 'ward', 'complete', 'average', 'single'.
    distance_metric : str or callable, optional
        Distance metric to use. Options: 'euclidean', 'manhattan', 'cosine'. Custom callable can also be provided.
    affinity : str, optional
        Affinity metric for computing distances between samples. Options: 'euclidean', 'manhattan', 'cosine'.
    compute_full_tree : bool, optional
        Whether to compute the full hierarchy or stop at n_clusters.
    custom_distance : callable, optional
        Custom distance function to use instead of built-in metrics.

    Returns:
    --------
    dict
        A dictionary containing the following keys:
        - 'result': The clustering result (labels or full hierarchy).
        - 'metrics': Dictionary of computed metrics.
        - 'params_used': Parameters used for the computation.
        - 'warnings': List of warnings encountered during computation.

    Example:
    --------
    >>> data = np.random.rand(10, 5)
    >>> result = agglomerative_clustering_fit(data, n_clusters=3)
    """
    # Validate inputs
    validate_inputs(data, n_clusters)

    # Initialize warnings list
    warnings = []

    # Set default parameters if not provided
    params_used = {
        'n_clusters': n_clusters,
        'linkage_criterion': linkage_criterion,
        'distance_metric': distance_metric,
        'affinity': affinity,
        'compute_full_tree': compute_full_tree
    }

    # Compute distance matrix
    if custom_distance is not None:
        distance_matrix = compute_custom_distance(data, custom_distance)
    else:
        distance_matrix = compute_distance_matrix(data, distance_metric)

    # Perform agglomerative clustering
    if n_clusters is None or compute_full_tree:
        result = compute_full_hierarchy(distance_matrix, linkage_criterion)
    else:
        result = compute_clusters(distance_matrix, n_clusters, linkage_criterion)

    # Compute metrics
    metrics = compute_metrics(data, result, distance_matrix)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

def validate_inputs(data: np.ndarray, n_clusters: Optional[int]) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array.")
    if n_clusters is not None and (n_clusters <= 0 or n_clusters > data.shape[0]):
        raise ValueError("n_clusters must be between 1 and the number of samples.")

def compute_distance_matrix(data: np.ndarray, metric: str) -> np.ndarray:
    """Compute the distance matrix using the specified metric."""
    n_samples = data.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    if metric == 'euclidean':
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance = np.linalg.norm(data[i] - data[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
    elif metric == 'manhattan':
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance = np.sum(np.abs(data[i] - data[j]))
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
    elif metric == 'cosine':
        for i in range(n_samples):
            for j in range(i, n_samples):
                dot_product = np.dot(data[i], data[j])
                norm_i = np.linalg.norm(data[i])
                norm_j = np.linalg.norm(data[j])
                distance = 1 - (dot_product / (norm_i * norm_j))
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return distance_matrix

def compute_custom_distance(data: np.ndarray, custom_func: Callable) -> np.ndarray:
    """Compute the distance matrix using a custom distance function."""
    n_samples = data.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i, n_samples):
            distance = custom_func(data[i], data[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix

def compute_full_hierarchy(distance_matrix: np.ndarray, criterion: str) -> Dict:
    """Compute the full hierarchy of clusters."""
    n_samples = distance_matrix.shape[0]
    hierarchy = []
    current_clusters = [[i] for i in range(n_samples)]

    while len(current_clusters) > 1:
        min_distance = np.inf
        cluster_pair = None

        for i in range(len(current_clusters)):
            for j in range(i + 1, len(current_clusters)):
                distance = compute_linkage_distance(
                    current_clusters[i],
                    current_clusters[j],
                    distance_matrix,
                    criterion
                )
                if distance < min_distance:
                    min_distance = distance
                    cluster_pair = (i, j)

        if cluster_pair is None:
            break

        i, j = cluster_pair
        new_cluster = current_clusters[i] + current_clusters[j]
        hierarchy.append((current_clusters[i], current_clusters[j], min_distance))
        current_clusters.pop(j)
        current_clusters.pop(i)
        current_clusters.append(new_cluster)

    return {'hierarchy': hierarchy}

def compute_clusters(distance_matrix: np.ndarray, n_clusters: int, criterion: str) -> np.ndarray:
    """Compute the clusters using agglomerative clustering."""
    n_samples = distance_matrix.shape[0]
    current_clusters = [[i] for i in range(n_samples)]
    labels = np.arange(n_samples)

    while len(current_clusters) > n_clusters:
        min_distance = np.inf
        cluster_pair = None

        for i in range(len(current_clusters)):
            for j in range(i + 1, len(current_clusters)):
                distance = compute_linkage_distance(
                    current_clusters[i],
                    current_clusters[j],
                    distance_matrix,
                    criterion
                )
                if distance < min_distance:
                    min_distance = distance
                    cluster_pair = (i, j)

        if cluster_pair is None:
            break

        i, j = cluster_pair
        new_cluster = current_clusters[i] + current_clusters[j]
        for sample in new_cluster:
            labels[sample] = len(current_clusters)
        current_clusters.pop(j)
        current_clusters.pop(i)
        current_clusters.append(new_cluster)

    return labels

def compute_linkage_distance(
    cluster1: list,
    cluster2: list,
    distance_matrix: np.ndarray,
    criterion: str
) -> float:
    """Compute the linkage distance between two clusters."""
    if criterion == 'ward':
        n1 = len(cluster1)
        n2 = len(cluster2)
        sum1 = np.sum([distance_matrix[i, j] for i in cluster1 for j in cluster1])
        sum2 = np.sum([distance_matrix[i, j] for i in cluster2 for j in cluster2])
        sum12 = np.sum([distance_matrix[i, j] for i in cluster1 for j in cluster2])
        return np.sqrt((sum1 / (2 * n1) + sum2 / (2 * n2) - sum12 / (n1 + n2)) / (1 / n1 + 1 / n2))
    elif criterion == 'complete':
        return np.max([distance_matrix[i, j] for i in cluster1 for j in cluster2])
    elif criterion == 'average':
        return np.mean([distance_matrix[i, j] for i in cluster1 for j in cluster2])
    elif criterion == 'single':
        return np.min([distance_matrix[i, j] for i in cluster1 for j in cluster2])
    else:
        raise ValueError(f"Unsupported linkage criterion: {criterion}")

def compute_metrics(data: np.ndarray, labels: Union[np.ndarray, Dict], distance_matrix: np.ndarray) -> Dict:
    """Compute metrics for the clustering result."""
    if isinstance(labels, dict):
        # Full hierarchy case
        return {'hierarchy_depth': len(labels['hierarchy'])}
    else:
        # Cluster labels case
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        metrics = {
            'n_clusters': n_clusters,
            'silhouette_score': compute_silhouette_score(data, labels, distance_matrix)
        }
        return metrics

def compute_silhouette_score(data: np.ndarray, labels: np.ndarray, distance_matrix: np.ndarray) -> float:
    """Compute the silhouette score for the clustering result."""
    n_samples = data.shape[0]
    silhouette_scores = np.zeros(n_samples)

    for i in range(n_samples):
        a = np.mean([distance_matrix[i, j] for j in range(n_samples) if labels[j] == labels[i] and i != j])
        b = np.inf
        for label in np.unique(labels):
            if label == labels[i]:
                continue
            b_temp = np.mean([distance_matrix[i, j] for j in range(n_samples) if labels[j] == label])
            if b_temp < b:
                b = b_temp
        silhouette_scores[i] = (b - a) / max(a, b)

    return np.mean(silhouette_scores)

################################################################################
# divisive_clustering
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def divisive_clustering_fit(
    data: np.ndarray,
    distance_metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    split_criterion: Union[str, Callable[[np.ndarray], float]] = 'variance',
    max_depth: int = 10,
    min_samples_split: int = 2,
    normalization: Optional[str] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_criterion: Optional[Callable[[np.ndarray], float]] = None
) -> Dict:
    """
    Perform divisive hierarchical clustering on the input data.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    distance_metric : str or callable, optional
        Distance metric to use. Can be 'euclidean', 'manhattan', 'cosine',
        or a custom callable.
    split_criterion : str or callable, optional
        Criterion to use for splitting clusters. Can be 'variance',
        or a custom callable.
    max_depth : int, optional
        Maximum depth of the clustering tree.
    min_samples_split : int, optional
        Minimum number of samples required to split a node.
    normalization : str or None, optional
        Normalization method. Can be 'standard', 'minmax', 'robust', or None.
    custom_distance : callable or None, optional
        Custom distance function if not using built-in metrics.
    custom_criterion : callable or None, optional
        Custom split criterion function if not using built-in criteria.

    Returns:
    --------
    dict
        Dictionary containing the clustering results, metrics, parameters used,
        and any warnings.
    """
    # Validate inputs
    _validate_inputs(data, distance_metric, split_criterion,
                     max_depth, min_samples_split, normalization)

    # Normalize data if specified
    normalized_data = _apply_normalization(data, normalization)

    # Initialize clustering tree
    root_node = {'data': normalized_data, 'left': None, 'right': None}

    # Build the clustering tree
    _build_tree(root_node, distance_metric, split_criterion,
                max_depth, min_samples_split, custom_distance, custom_criterion)

    # Calculate metrics
    metrics = _calculate_metrics(root_node, distance_metric)

    return {
        'result': root_node,
        'metrics': metrics,
        'params_used': {
            'distance_metric': distance_metric,
            'split_criterion': split_criterion,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'normalization': normalization
        },
        'warnings': []
    }

def _validate_inputs(
    data: np.ndarray,
    distance_metric: Union[str, Callable],
    split_criterion: Union[str, Callable],
    max_depth: int,
    min_samples_split: int,
    normalization: Optional[str]
) -> None:
    """Validate the input parameters."""
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    if max_depth <= 0:
        raise ValueError("max_depth must be positive.")
    if min_samples_split < 2:
        raise ValueError("min_samples_split must be at least 2.")
    if normalization not in [None, 'standard', 'minmax', 'robust']:
        raise ValueError("Invalid normalization method.")

def _apply_normalization(
    data: np.ndarray,
    method: Optional[str]
) -> np.ndarray:
    """Apply normalization to the data."""
    if method is None:
        return data
    elif method == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == 'robust':
        return (data - np.median(data, axis=0)) / (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0))
    else:
        raise ValueError("Invalid normalization method.")

def _build_tree(
    node: Dict,
    distance_metric: Union[str, Callable],
    split_criterion: Union[str, Callable],
    max_depth: int,
    min_samples_split: int,
    custom_distance: Optional[Callable],
    custom_criterion: Optional[Callable]
) -> None:
    """Recursively build the clustering tree."""
    if node['left'] is not None or node['right'] is not None:
        return

    if len(node['data']) <= min_samples_split or max_depth == 0:
        return

    # Choose distance and criterion functions
    dist_func = _get_distance_function(distance_metric, custom_distance)
    crit_func = _get_criterion_function(split_criterion, custom_criterion)

    # Find the best split
    best_split = _find_best_split(node['data'], dist_func, crit_func)

    if best_split is None:
        return

    # Split the node
    left_data, right_data = _split_node(node['data'], best_split)
    node['left'] = {'data': left_data, 'left': None, 'right': None}
    node['right'] = {'data': right_data, 'left': None, 'right': None}

    # Recursively build the tree
    _build_tree(node['left'], distance_metric, split_criterion,
                max_depth - 1, min_samples_split, custom_distance, custom_criterion)
    _build_tree(node['right'], distance_metric, split_criterion,
                max_depth - 1, min_samples_split, custom_distance, custom_criterion)

def _get_distance_function(
    metric: Union[str, Callable],
    custom_func: Optional[Callable]
) -> Callable:
    """Get the distance function based on the metric."""
    if callable(metric):
        return metric
    elif custom_func is not None:
        return custom_func
    elif metric == 'euclidean':
        return lambda x, y: np.linalg.norm(x - y)
    elif metric == 'manhattan':
        return lambda x, y: np.sum(np.abs(x - y))
    elif metric == 'cosine':
        return lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    else:
        raise ValueError("Invalid distance metric.")

def _get_criterion_function(
    criterion: Union[str, Callable],
    custom_func: Optional[Callable]
) -> Callable:
    """Get the split criterion function."""
    if callable(criterion):
        return criterion
    elif custom_func is not None:
        return custom_func
    elif criterion == 'variance':
        return lambda data: np.var(data, axis=0).sum()
    else:
        raise ValueError("Invalid split criterion.")

def _find_best_split(
    data: np.ndarray,
    distance_func: Callable,
    criterion_func: Callable
) -> Optional[Dict]:
    """Find the best split for the current node."""
    # Implement logic to find the best split based on distance and criterion
    pass

def _split_node(
    data: np.ndarray,
    split_info: Dict
) -> tuple:
    """Split the node into left and right children."""
    # Implement logic to split the data based on split_info
    pass

def _calculate_metrics(
    node: Dict,
    distance_metric: Union[str, Callable]
) -> Dict:
    """Calculate metrics for the clustering results."""
    # Implement logic to calculate metrics
    pass

################################################################################
# dendrogramme
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def dendrogramme_fit(
    data: np.ndarray,
    linkage_method: str = 'ward',
    metric: Union[str, Callable] = 'euclidean',
    normalization: str = 'none',
    custom_distance: Optional[Callable] = None,
    threshold: float = 0.0,
    **kwargs
) -> Dict:
    """
    Compute a hierarchical clustering dendrogram.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    linkage_method : str, optional
        Linkage method for hierarchical clustering ('ward', 'complete', 'average', 'single').
    metric : str or callable, optional
        Distance metric ('euclidean', 'manhattan', 'cosine') or custom distance function.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    custom_distance : callable, optional
        Custom distance function if metric is not predefined.
    threshold : float, optional
        Threshold for cutting the dendrogram.

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': Linkage matrix.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used in the computation.
        - 'warnings': Any warnings encountered.

    Example:
    --------
    >>> data = np.random.rand(10, 5)
    >>> result = dendrogramme_fit(data, linkage_method='ward', metric='euclidean')
    """
    # Validate input data
    _validate_input(data)

    # Normalize data if required
    normalized_data = _normalize_data(data, normalization)

    # Compute distance matrix
    distance_matrix = _compute_distance(normalized_data, metric, custom_distance)

    # Perform hierarchical clustering
    linkage_matrix = _hierarchical_clustering(distance_matrix, linkage_method)

    # Apply threshold if specified
    if threshold > 0:
        linkage_matrix = _apply_threshold(linkage_matrix, threshold)

    # Prepare output
    output = {
        'result': linkage_matrix,
        'metrics': {'distance_metric': metric, 'linkage_method': linkage_method},
        'params_used': {
            'normalization': normalization,
            'threshold': threshold
        },
        'warnings': []
    }

    return output

def _validate_input(data: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or infinite values.")

def _normalize_data(
    data: np.ndarray,
    method: str = 'none'
) -> np.ndarray:
    """Normalize data using specified method."""
    if method == 'none':
        return data
    elif method == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == 'robust':
        return (data - np.median(data, axis=0)) / (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_distance(
    data: np.ndarray,
    metric: Union[str, Callable],
    custom_distance: Optional[Callable] = None
) -> np.ndarray:
    """Compute distance matrix using specified metric."""
    n_samples = data.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    if isinstance(metric, str):
        if metric == 'euclidean':
            distance_matrix = np.sqrt(np.sum((data[:, np.newaxis] - data) ** 2, axis=2))
        elif metric == 'manhattan':
            distance_matrix = np.sum(np.abs(data[:, np.newaxis] - data), axis=2)
        elif metric == 'cosine':
            dot_products = np.dot(data, data.T)
            norms = np.sqrt(np.sum(data ** 2, axis=1))[:, np.newaxis]
            distance_matrix = 1 - (dot_products / np.dot(norms, norms.T))
        else:
            raise ValueError(f"Unknown distance metric: {metric}")
    elif callable(metric):
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance_matrix[i, j] = metric(data[i], data[j])
                distance_matrix[j, i] = distance_matrix[i, j]
    elif custom_distance is not None:
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance_matrix[i, j] = custom_distance(data[i], data[j])
                distance_matrix[j, i] = distance_matrix[i, j]
    else:
        raise ValueError("Either metric or custom_distance must be provided.")

    return distance_matrix

def _hierarchical_clustering(
    distance_matrix: np.ndarray,
    method: str = 'ward'
) -> np.ndarray:
    """Perform hierarchical clustering using specified linkage method."""
    n_samples = distance_matrix.shape[0]
    linkage_matrix = np.zeros((n_samples - 1, 4))

    # Placeholder for actual hierarchical clustering implementation
    # This would typically use a library like scipy.cluster.hierarchy.linkage
    # For the sake of this example, we return a dummy linkage matrix
    for i in range(n_samples - 1):
        linkage_matrix[i, 0] = i
        linkage_matrix[i, 1] = i + 1
        linkage_matrix[i, 2] = distance_matrix[i, i + 1]
        if method == 'ward':
            linkage_matrix[i, 3] = n_samples - i - 1
        else:
            linkage_matrix[i, 3] = distance_matrix[i, i + 1]

    return linkage_matrix

def _apply_threshold(
    linkage_matrix: np.ndarray,
    threshold: float
) -> np.ndarray:
    """Apply threshold to the linkage matrix."""
    # Placeholder for actual threshold application
    return linkage_matrix[linkage_matrix[:, 2] <= threshold]

################################################################################
# hierarchical_regression
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def hierarchical_regression_fit(
    X: np.ndarray,
    y: np.ndarray,
    hierarchy_levels: int = 1,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, str]]:
    """
    Fit a hierarchical regression model.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    hierarchy_levels : int, optional
        Number of hierarchy levels to consider.
    normalizer : Callable[[np.ndarray], np.ndarray], optional
        Function to normalize the input features.
    metric : str, optional
        Metric to evaluate the model. Options: 'mse', 'mae', 'r2', 'logloss'.
    distance : str, optional
        Distance metric for hierarchical clustering. Options: 'euclidean', 'manhattan', 'cosine', 'minkowski'.
    solver : str, optional
        Solver to use for optimization. Options: 'closed_form', 'gradient_descent', 'newton', 'coordinate_descent'.
    regularization : str, optional
        Regularization type. Options: 'none', 'l1', 'l2', 'elasticnet'.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : Callable[[np.ndarray, np.ndarray], float], optional
        Custom metric function.
    custom_distance : Callable[[np.ndarray, np.ndarray], float], optional
        Custom distance function.

    Returns
    -------
    Dict[str, Union[Dict, float, str]]
        Dictionary containing the results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = hierarchical_regression_fit(X, y, hierarchy_levels=2, metric='mse')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if normalizer is provided
    X_normalized = _apply_normalization(X, normalizer)

    # Choose metric
    if custom_metric is not None:
        metric_func = custom_metric
    else:
        metric_func = _get_metric_function(metric)

    # Choose distance
    if custom_distance is not None:
        distance_func = custom_distance
    else:
        distance_func = _get_distance_function(distance)

    # Choose solver
    if solver == 'closed_form':
        params = _closed_form_solver(X_normalized, y)
    elif solver == 'gradient_descent':
        params = _gradient_descent_solver(X_normalized, y, tol, max_iter)
    elif solver == 'newton':
        params = _newton_solver(X_normalized, y, tol, max_iter)
    elif solver == 'coordinate_descent':
        params = _coordinate_descent_solver(X_normalized, y, tol, max_iter)
    else:
        raise ValueError("Invalid solver specified.")

    # Apply regularization if needed
    if regularization is not None:
        params = _apply_regularization(params, X_normalized, y, regularization)

    # Calculate metrics
    predictions = _predict(X_normalized, params)
    metrics = {
        'metric': metric_func(y, predictions),
        'mse': _mean_squared_error(y, predictions),
        'mae': _mean_absolute_error(y, predictions),
        'r2': _r_squared(y, predictions)
    }

    # Return results
    return {
        'result': {'params': params, 'predictions': predictions},
        'metrics': metrics,
        'params_used': {
            'hierarchy_levels': hierarchy_levels,
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': _check_warnings(y, predictions)
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values.")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    """Apply normalization to the input features."""
    if normalizer is not None:
        return normalizer(X)
    return X

def _get_metric_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the metric function based on the specified metric."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared,
        'logloss': _log_loss
    }
    if metric not in metrics:
        raise ValueError(f"Invalid metric specified: {metric}")
    return metrics[metric]

def _get_distance_function(distance: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the distance function based on the specified distance."""
    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance,
        'minkowski': _minkowski_distance
    }
    if distance not in distances:
        raise ValueError(f"Invalid distance specified: {distance}")
    return distances[distance]

def _closed_form_solver(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve the regression problem using closed-form solution."""
    return np.linalg.inv(X.T @ X) @ X.T @ y

def _gradient_descent_solver(X: np.ndarray, y: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
    """Solve the regression problem using gradient descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    learning_rate = 0.01
    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ params - y) / len(y)
        new_params = params - learning_rate * gradient
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params
    return params

def _newton_solver(X: np.ndarray, y: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
    """Solve the regression problem using Newton's method."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ params - y) / len(y)
        hessian = 2 * X.T @ X / len(y)
        new_params = params - np.linalg.inv(hessian) @ gradient
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params
    return params

def _coordinate_descent_solver(X: np.ndarray, y: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
    """Solve the regression problem using coordinate descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residual = y - X @ params + params[j] * X_j
            params[j] = np.sum(X_j * residual) / np.sum(X_j ** 2)
        if np.linalg.norm(params - np.zeros(n_features)) < tol:
            break
    return params

def _apply_regularization(params: np.ndarray, X: np.ndarray, y: np.ndarray, regularization: str) -> np.ndarray:
    """Apply regularization to the parameters."""
    if regularization == 'l1':
        return _l1_regularization(params, X, y)
    elif regularization == 'l2':
        return _l2_regularization(params, X, y)
    elif regularization == 'elasticnet':
        return _elasticnet_regularization(params, X, y)
    else:
        raise ValueError("Invalid regularization type specified.")

def _l1_regularization(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Apply L1 regularization."""
    alpha = 0.1
    gradient = 2 * X.T @ (X @ params - y) / len(y)
    gradient += alpha * np.sign(params)
    return params - 0.01 * gradient

def _l2_regularization(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Apply L2 regularization."""
    alpha = 0.1
    gradient = 2 * X.T @ (X @ params - y) / len(y)
    gradient += 2 * alpha * params
    return params - 0.01 * gradient

def _elasticnet_regularization(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Apply ElasticNet regularization."""
    alpha = 0.1
    l1_ratio = 0.5
    gradient = 2 * X.T @ (X @ params - y) / len(y)
    l1_penalty = l1_ratio * alpha * np.sign(params)
    l2_penalty = (1 - l1_ratio) * alpha * params
    gradient += l1_penalty + 2 * l2_penalty
    return params - 0.01 * gradient

def _predict(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Make predictions using the fitted parameters."""
    return X @ params

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the R-squared metric."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the log loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the cosine distance."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _minkowski_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the Minkowski distance."""
    return np.sum(np.abs(a - b) ** 3) ** (1/3)

def _check_warnings(y_true: np.ndarray, y_pred: np.ndarray) -> list:
    """Check for warnings during model fitting."""
    warnings = []
    if np.any(np.isnan(y_pred)):
        warnings.append("Predictions contain NaN values.")
    if np.any(np.isinf(y_pred)):
        warnings.append("Predictions contain infinite values.")
    return warnings

################################################################################
# hierarchical_classification
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List
from enum import Enum

class Normalization(Enum):
    NONE = "none"
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"

class Metric(Enum):
    MSE = "mse"
    MAE = "mae"
    R2 = "r2"
    LOGLOSS = "logloss"

class Distance(Enum):
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"
    MINKOWSKI = "minkowski"

class Solver(Enum):
    CLOSED_FORM = "closed_form"
    GRADIENT_DESCENT = "gradient_descent"
    NEWTON = "newton"
    COORDINATE_DESCENT = "coordinate_descent"

class Regularization(Enum):
    NONE = "none"
    L1 = "l1"
    L2 = "l2"
    ELASTICNET = "elasticnet"

def hierarchical_classification_fit(
    X: np.ndarray,
    y: np.ndarray,
    hierarchy: List[List[int]],
    normalization: Union[Normalization, str] = Normalization.STANDARD,
    metric: Union[Metric, str] = Metric.MSE,
    distance: Union[Distance, str] = Distance.EUCLIDEAN,
    solver: Union[Solver, str] = Solver.GRADIENT_DESCENT,
    regularization: Union[Regularization, str] = Regularization.NONE,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict:
    """
    Fit a hierarchical classification model.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    hierarchy : List[List[int]]
        Hierarchical structure represented as a list of lists.
    normalization : Union[Normalization, str], optional
        Normalization method to apply to features.
    metric : Union[Metric, str], optional
        Metric to evaluate the model performance.
    distance : Union[Distance, str], optional
        Distance metric for hierarchical clustering.
    solver : Union[Solver, str], optional
        Solver to use for optimization.
    regularization : Union[Regularization, str], optional
        Regularization method to apply.
    alpha : float, optional
        Regularization strength.
    l1_ratio : float, optional
        ElasticNet mixing parameter (0 = L2, 1 = L1).
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for optimization convergence.
    random_state : Optional[int], optional
        Random seed for reproducibility.
    custom_metric : Optional[Callable], optional
        Custom metric function.
    custom_distance : Optional[Callable], optional
        Custom distance function.

    Returns:
    --------
    Dict
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y, hierarchy)

    # Normalize features
    X_normalized = _apply_normalization(X, normalization)

    # Initialize parameters
    params = {
        "normalization": normalization,
        "metric": metric,
        "distance": distance,
        "solver": solver,
        "regularization": regularization,
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "max_iter": max_iter,
        "tol": tol,
        "random_state": random_state
    }

    # Fit the model based on solver choice
    if solver == Solver.CLOSED_FORM:
        result = _closed_form_solution(X_normalized, y, hierarchy)
    elif solver == Solver.GRADIENT_DESCENT:
        result = _gradient_descent(X_normalized, y, hierarchy, distance, regularization,
                                  alpha, l1_ratio, max_iter, tol, random_state)
    elif solver == Solver.NEWTON:
        result = _newton_method(X_normalized, y, hierarchy, distance, regularization,
                               alpha, l1_ratio, max_iter, tol)
    elif solver == Solver.COORDINATE_DESCENT:
        result = _coordinate_descent(X_normalized, y, hierarchy, distance, regularization,
                                    alpha, l1_ratio, max_iter, tol)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(y, result["predictions"], metric, custom_metric)

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params,
        "warnings": []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray, hierarchy: List[List[int]]) -> None:
    """Validate input dimensions and types."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")
    if not isinstance(hierarchy, list) or not all(isinstance(level, list) for level in hierarchy):
        raise ValueError("hierarchy must be a list of lists")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalization(X: np.ndarray, normalization: Union[Normalization, str]) -> np.ndarray:
    """Apply feature normalization."""
    if isinstance(normalization, str):
        normalization = Normalization(normalization)

    if normalization == Normalization.NONE:
        return X
    elif normalization == Normalization.STANDARD:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif normalization == Normalization.MINMAX:
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif normalization == Normalization.ROBUST:
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _closed_form_solution(X: np.ndarray, y: np.ndarray, hierarchy: List[List[int]]) -> Dict:
    """Closed form solution for hierarchical classification."""
    # Placeholder implementation
    predictions = np.zeros_like(y)
    return {"predictions": predictions}

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    hierarchy: List[List[int]],
    distance: Union[Distance, str],
    regularization: Union[Regularization, str],
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tol: float,
    random_state: Optional[int]
) -> Dict:
    """Gradient descent solver for hierarchical classification."""
    # Placeholder implementation
    predictions = np.zeros_like(y)
    return {"predictions": predictions}

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    hierarchy: List[List[int]],
    distance: Union[Distance, str],
    regularization: Union[Regularization, str],
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tol: float
) -> Dict:
    """Newton method solver for hierarchical classification."""
    # Placeholder implementation
    predictions = np.zeros_like(y)
    return {"predictions": predictions}

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    hierarchy: List[List[int]],
    distance: Union[Distance, str],
    regularization: Union[Regularization, str],
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tol: float
) -> Dict:
    """Coordinate descent solver for hierarchical classification."""
    # Placeholder implementation
    predictions = np.zeros_like(y)
    return {"predictions": predictions}

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[Metric, str],
    custom_metric: Optional[Callable]
) -> Dict:
    """Calculate evaluation metrics."""
    if isinstance(metric, str):
        metric = Metric(metric)

    metrics = {}

    if custom_metric is not None:
        metrics["custom"] = custom_metric(y_true, y_pred)

    if metric == Metric.MSE:
        metrics["mse"] = np.mean((y_true - y_pred) ** 2)
    elif metric == Metric.MAE:
        metrics["mae"] = np.mean(np.abs(y_true - y_pred))
    elif metric == Metric.R2:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics["r2"] = 1 - (ss_res / ss_tot)
    elif metric == Metric.LOGLOSS:
        # Placeholder for log loss calculation
        pass

    return metrics

################################################################################
# ensemble_learning
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def ensemble_learning_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Fit an ensemble learning model with hierarchical structure.

    Parameters:
    - X: Input features (n_samples, n_features)
    - y: Target values (n_samples,)
    - normalizer: Callable for normalization
    - metric: Metric to optimize ('mse', 'mae', 'r2', 'logloss') or custom callable
    - distance: Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable
    - solver: Solver type ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    - regularization: Regularization type (None, 'l1', 'l2', 'elasticnet')
    - tol: Tolerance for convergence
    - max_iter: Maximum iterations
    - custom_metric: Custom metric function if needed
    - custom_distance: Custom distance function if needed

    Returns:
    - Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_norm = _apply_normalization(X, normalizer)

    # Initialize parameters
    params = {
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'max_iter': max_iter
    }

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

    # Solve the model
    if solver == 'closed_form':
        result, metrics = _solve_closed_form(X_norm, y, metric_func)
    elif solver == 'gradient_descent':
        result, metrics = _solve_gradient_descent(X_norm, y, metric_func,
                                                 distance_func, regularization,
                                                 tol, max_iter)
    elif solver == 'newton':
        result, metrics = _solve_newton(X_norm, y, metric_func,
                                       distance_func, regularization,
                                       tol, max_iter)
    elif solver == 'coordinate_descent':
        result, metrics = _solve_coordinate_descent(X_norm, y, metric_func,
                                                   distance_func, regularization,
                                                   tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

    return output

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
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

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to input data."""
    if normalizer is None:
        return X
    try:
        return normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _get_metric_function(metric_name: str) -> Callable:
    """Get metric function based on name."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared,
        'logloss': _log_loss
    }
    if metric_name not in metrics:
        raise ValueError(f"Unknown metric: {metric_name}")
    return metrics[metric_name]

def _get_distance_function(distance_name: str) -> Callable:
    """Get distance function based on name."""
    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance,
        'minkowski': _minkowski_distance
    }
    if distance_name not in distances:
        raise ValueError(f"Unknown distance: {distance_name}")
    return distances[distance_name]

def _solve_closed_form(X: np.ndarray, y: np.ndarray, metric_func: Callable) -> tuple:
    """Solve using closed form solution."""
    # Implement closed form solution
    pass

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable,
    distance_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> tuple:
    """Solve using gradient descent."""
    # Implement gradient descent
    pass

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable,
    distance_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> tuple:
    """Solve using Newton's method."""
    # Implement Newton's method
    pass

def _solve_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable,
    distance_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> tuple:
    """Solve using coordinate descent."""
    # Implement coordinate descent
    pass

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

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate log loss."""
    # Implement log loss
    pass

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine distance."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _minkowski_distance(a: np.ndarray, b: np.ndarray, p: int = 3) -> float:
    """Calculate Minkowski distance."""
    return np.sum(np.abs(a - b) ** p) ** (1/p)

################################################################################
# feature_importance
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def feature_importance_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Calculate feature importance for hierarchical models.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    normalizer : Optional[Callable]
        Function to normalize features. Default is None.
    metric : Union[str, Callable]
        Metric to evaluate model performance. Default is "mse".
    distance : Union[str, Callable]
        Distance metric for hierarchical clustering. Default is "euclidean".
    solver : str
        Solver method. Options: "closed_form", "gradient_descent". Default is "closed_form".
    regularization : Optional[str]
        Regularization type. Options: None, "l1", "l2". Default is None.
    tol : float
        Tolerance for convergence. Default is 1e-4.
    max_iter : int
        Maximum number of iterations. Default is 1000.
    custom_metric : Optional[Callable]
        Custom metric function. Default is None.
    custom_distance : Optional[Callable]
        Custom distance function. Default is None.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if normalizer is provided
    X_normalized = _apply_normalization(X, normalizer)

    # Set metric and distance functions
    metric_func = _get_metric_function(metric, custom_metric)
    distance_func = _get_distance_function(distance, custom_distance)

    # Solve the model based on the chosen solver
    if solver == "closed_form":
        params = _solve_closed_form(X_normalized, y, regularization)
    elif solver == "gradient_descent":
        params = _solve_gradient_descent(
            X_normalized, y, metric_func, tol, max_iter, regularization
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate feature importance
    importance = _calculate_feature_importance(params, X_normalized, y, metric_func)

    # Calculate metrics
    metrics = _calculate_metrics(y, X_normalized @ params, metric_func)

    # Prepare the result dictionary
    result = {
        "result": importance,
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

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to features."""
    if normalizer is None:
        return X
    return normalizer(X)

def _get_metric_function(
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Callable:
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
        raise ValueError(f"Unknown metric: {metric}")

def _get_distance_function(
    distance: Union[str, Callable],
    custom_distance: Optional[Callable]
) -> Callable:
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
        raise ValueError(f"Unknown distance: {distance}")

def _solve_closed_form(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str]
) -> np.ndarray:
    """Solve the model using closed-form solution."""
    if regularization is None:
        return np.linalg.pinv(X) @ y
    elif regularization == "l2":
        return _ridge_regression(X, y)
    else:
        raise ValueError(f"Unknown regularization: {regularization}")

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable,
    tol: float,
    max_iter: int,
    regularization: Optional[str]
) -> np.ndarray:
    """Solve the model using gradient descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = _compute_gradient(X, y, params, metric_func, regularization)
        params -= gradient
        if np.linalg.norm(gradient) < tol:
            break
    return params

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric_func: Callable,
    regularization: Optional[str]
) -> np.ndarray:
    """Compute the gradient for gradient descent."""
    predictions = X @ params
    residuals = y - predictions
    gradient = -2 * X.T @ residuals / len(y)
    if regularization == "l1":
        gradient += np.sign(params)
    elif regularization == "l2":
        gradient += 2 * params
    return gradient

def _calculate_feature_importance(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable
) -> np.ndarray:
    """Calculate feature importance based on model parameters."""
    baseline_metric = metric_func(y, np.zeros_like(y))
    importance = []
    for i in range(X.shape[1]):
        X_temp = X.copy()
        X_temp[:, i] = 0
        temp_metric = metric_func(y, X_temp @ params)
        importance.append(baseline_metric - temp_metric)
    return np.array(importance)

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable
) -> Dict[str, float]:
    """Calculate metrics for the model."""
    return {"metric": metric_func(y_true, y_pred)}

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
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _ridge_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Perform ridge regression."""
    I = np.eye(X.shape[1])
    return np.linalg.inv(X.T @ X + I) @ X.T @ y

################################################################################
# hyperparametre_tuning
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def hyperparametre_tuning_fit(
    model: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    param_grid: Dict[str, Any] = {},
    scoring: Union[str, Callable] = 'mse',
    normalizer: Optional[Callable] = None,
    solver: str = 'gradient_descent',
    n_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning for hierarchical models.

    Parameters:
    -----------
    model : Callable
        The model to tune, must be callable with parameters.
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training targets.
    X_val : Optional[np.ndarray]
        Validation features, if None use training data for validation.
    y_val : Optional[np.ndarray]
        Validation targets, if None use training data for validation.
    param_grid : Dict[str, Any]
        Dictionary of hyperparameters to tune.
    scoring : Union[str, Callable]
        Scoring metric ('mse', 'mae', 'r2') or custom callable.
    normalizer : Optional[Callable]
        Normalization function (e.g., StandardScaler, MinMaxScaler).
    solver : str
        Solver method ('gradient_descent', 'newton', 'coordinate_descent').
    n_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for convergence.
    random_state : Optional[int]
        Random seed for reproducibility.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings.
    """
    # Validate inputs
    _validate_inputs(X_train, y_train, X_val, y_val)

    # Normalize data if specified
    if normalizer is not None:
        X_train = normalizer.fit_transform(X_train)
        if X_val is not None:
            X_val = normalizer.transform(X_val)

    # Initialize best parameters and score
    best_params = None
    best_score = np.inf if scoring in ['mse', 'mae'] else -np.inf

    # Grid search over parameter grid
    for params in _generate_param_combinations(param_grid):
        try:
            # Fit model with current parameters
            fitted_model = _fit_model(model, X_train, y_train, params, solver, n_iter, tol)

            # Compute score
            current_score = _compute_score(
                fitted_model,
                X_train if X_val is None else X_val,
                y_train if y_val is None else y_val,
                scoring
            )

            # Update best parameters and score
            if (scoring in ['mse', 'mae'] and current_score < best_score) or \
               (scoring in ['r2', 'logloss'] and current_score > best_score):
                best_params = params
                best_score = current_score

        except Exception as e:
            warnings.append(f"Error with params {params}: {str(e)}")

    # Fit final model with best parameters
    final_model = _fit_model(model, X_train, y_train, best_params, solver, n_iter, tol)
    final_score = _compute_score(final_model, X_train if X_val is None else X_val,
                                y_train if y_val is None else y_val, scoring)

    return {
        "result": final_model,
        "metrics": {"score": final_score},
        "params_used": best_params,
        "warnings": warnings
    }

def _validate_inputs(X_train: np.ndarray, y_train: np.ndarray,
                    X_val: Optional[np.ndarray], y_val: Optional[np.ndarray]) -> None:
    """Validate input data dimensions and types."""
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train must have the same number of samples")
    if X_val is not None and (X_val.shape[0] != y_val.shape[0]):
        raise ValueError("X_val and y_val must have the same number of samples")
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        raise ValueError("X_train contains NaN or infinite values")
    if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
        raise ValueError("y_train contains NaN or infinite values")

def _generate_param_combinations(param_grid: Dict[str, Any]) -> Any:
    """Generate all combinations of hyperparameters from param_grid."""
    # Implementation depends on the structure of param_grid
    # This is a simplified version - actual implementation may vary
    from itertools import product
    items = sorted(param_grid.items())
    keys, values = zip(*items)
    for v in product(*values):
        params = dict(zip(keys, v))
        yield params

def _fit_model(model: Callable, X: np.ndarray, y: np.ndarray,
               params: Dict[str, Any], solver: str, n_iter: int, tol: float) -> Any:
    """Fit the model with given parameters and solver."""
    # This is a placeholder - actual implementation depends on the model
    return model(X, y, **params)

def _compute_score(model: Any, X: np.ndarray, y: np.ndarray,
                   scoring: Union[str, Callable]) -> float:
    """Compute the score for given model and data."""
    if callable(scoring):
        return scoring(model, X, y)
    elif scoring == 'mse':
        return np.mean((model.predict(X) - y) ** 2)
    elif scoring == 'mae':
        return np.mean(np.abs(model.predict(X) - y))
    elif scoring == 'r2':
        ss_res = np.sum((y - model.predict(X)) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    else:
        raise ValueError(f"Unknown scoring metric: {scoring}")

################################################################################
# cross_validation
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def cross_validation_fit(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    random_state: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform cross-validation for hierarchical models.

    Parameters:
    -----------
    model : Callable
        The hierarchical model to validate.
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    n_splits : int, optional
        Number of cross-validation splits (default: 5).
    normalizer : Callable, optional
        Function to normalize data (default: None).
    metric : str or Callable, optional
        Metric to evaluate ('mse', 'mae', 'r2', etc.) or custom callable (default: 'mse').
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', etc.) (default: 'closed_form').
    random_state : int, optional
        Random seed for reproducibility (default: None).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_norm = normalizer(X) if normalizer else X

    # Split data into folds
    folds = _create_folds(X_norm, y, n_splits, random_state)

    # Initialize results storage
    results = {
        'result': [],
        'metrics': [],
        'params_used': {
            'n_splits': n_splits,
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'solver': solver
        },
        'warnings': []
    }

    # Perform cross-validation
    for train_idx, test_idx in folds:
        X_train, X_test = X_norm[train_idx], X_norm[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit model
        fitted_model = _fit_model(model, X_train, y_train, solver=solver, **kwargs)

        # Predict and compute metric
        y_pred = _predict(fitted_model, X_test)
        current_metric = _compute_metric(y_test, y_pred, metric)

        # Store results
        results['result'].append(fitted_model)
        results['metrics'].append(current_metric)

    return results

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _create_folds(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    random_state: Optional[int]
) -> list:
    """Create cross-validation folds."""
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    fold_size = len(indices) // n_splits
    return [
        (indices[i*fold_size:(i+1)*fold_size], indices[(i+1)*fold_size:])
        for i in range(n_splits)
    ]

def _fit_model(
    model: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    solver: str,
    **kwargs
) -> Any:
    """Fit the model using specified solver."""
    if solver == 'closed_form':
        return _fit_closed_form(model, X_train, y_train)
    elif solver == 'gradient_descent':
        return _fit_gradient_descent(model, X_train, y_train, **kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _fit_closed_form(
    model: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray
) -> Any:
    """Fit model using closed-form solution."""
    # Example implementation - adjust based on actual model requirements
    XtX = np.dot(X_train.T, X_train)
    if not np.allclose(np.linalg.det(XtX), 0):
        XtX_inv = np.linalg.inv(XtX)
        weights = np.dot(np.dot(XtX_inv, X_train.T), y_train)
    else:
        weights = np.linalg.pinv(XtX) @ X_train.T @ y_train
    return {'weights': weights}

def _fit_gradient_descent(
    model: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float = 0.01,
    n_iterations: int = 1000,
    tol: float = 1e-4
) -> Any:
    """Fit model using gradient descent."""
    # Example implementation - adjust based on actual model requirements
    weights = np.zeros(X_train.shape[1])
    for _ in range(n_iterations):
        gradient = 2 * np.dot(X_train.T, (np.dot(X_train, weights) - y_train)) / len(y_train)
        new_weights = weights - learning_rate * gradient
        if np.linalg.norm(new_weights - weights) < tol:
            break
        weights = new_weights
    return {'weights': weights}

def _predict(model: Any, X_test: np.ndarray) -> np.ndarray:
    """Make predictions using fitted model."""
    return np.dot(X_test, model['weights'])

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> float:
    """Compute specified metric."""
    if callable(metric):
        return metric(y_true, y_pred)
    elif metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    else:
        raise ValueError(f"Unknown metric: {metric}")

# Example usage
if __name__ == "__main__":
    # Define a simple linear model
    def linear_model(X, y):
        return np.linalg.pinv(X.T @ X) @ X.T @ y

    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.dot(X, np.array([1.2, -3.4, 0.5, 2.1, -1.8])) + np.random.randn(100) * 0.5

    # Standard normalizer
    def standard_normalize(X):
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Run cross-validation
    results = cross_validation_fit(
        model=linear_model,
        X=X,
        y=y,
        n_splits=5,
        normalizer=standard_normalize,
        metric='mse',
        solver='closed_form'
    )
