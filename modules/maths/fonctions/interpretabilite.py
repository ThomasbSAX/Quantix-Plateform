"""
Quantix – Module interpretabilite
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# modeles_lineaires
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def modeles_lineaires_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalisation: str = "standard",
    metrique: Union[str, Callable] = "mse",
    distance: str = "euclidean",
    solveur: str = "closed_form",
    regularisation: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    alpha: float = 1.0,
    l1_ratio: Optional[float] = None,
    custom_metrique: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fit a linear model with various options for interpretability.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    normalisation : str, optional
        Normalization method: "none", "standard", "minmax", or "robust".
    metrique : str or callable, optional
        Metric to evaluate the model: "mse", "mae", "r2", or custom callable.
    distance : str, optional
        Distance metric: "euclidean", "manhattan", "cosine", or "minkowski".
    solveur : str, optional
        Solver method: "closed_form", "gradient_descent", "newton", or "coordinate_descent".
    regularisation : str, optional
        Regularization type: "none", "l1", "l2", or "elasticnet".
    tol : float, optional
        Tolerance for stopping criteria.
    max_iter : int, optional
        Maximum number of iterations.
    alpha : float, optional
        Regularization strength.
    l1_ratio : float, optional
        ElasticNet mixing parameter (0 <= l1_ratio <= 1).
    custom_metrique : callable, optional
        Custom metric function.
    custom_distance : callable, optional
        Custom distance function.

    Returns
    -------
    Dict
        Dictionary containing:
        - "result": Fitted model parameters.
        - "metrics": Computed metrics.
        - "params_used": Parameters used for fitting.
        - "warnings": Any warnings encountered.

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = modeles_lineaires_fit(X, y, normalisation="standard", metrique="mse")
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm = _apply_normalisation(X, normalisation)

    # Choose solver
    if solveur == "closed_form":
        params = _solve_closed_form(X_norm, y, regularisation, alpha, l1_ratio)
    elif solveur == "gradient_descent":
        params = _solve_gradient_descent(X_norm, y, tol, max_iter, regularisation, alpha)
    elif solveur == "newton":
        params = _solve_newton(X_norm, y, tol, max_iter)
    elif solveur == "coordinate_descent":
        params = _solve_coordinate_descent(X_norm, y, tol, max_iter, regularisation, alpha)
    else:
        raise ValueError("Invalid solver specified.")

    # Compute metrics
    metrics = _compute_metrics(X_norm, y, params, metrique, custom_metrique)

    # Prepare output
    result = {
        "result": params,
        "metrics": metrics,
        "params_used": {
            "normalisation": normalisation,
            "metrique": metrique,
            "distance": distance,
            "solveur": solveur,
            "regularisation": regularisation,
            "tol": tol,
            "max_iter": max_iter,
            "alpha": alpha,
            "l1_ratio": l1_ratio
        },
        "warnings": []
    }

    return result

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

def _apply_normalisation(X: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalisation to the data."""
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
        raise ValueError("Invalid normalisation method specified.")

def _solve_closed_form(X: np.ndarray, y: np.ndarray, reg_type: Optional[str], alpha: float,
                       l1_ratio: Optional[float]) -> np.ndarray:
    """Solve linear model using closed-form solution."""
    if reg_type is None or reg_type == "none":
        return np.linalg.pinv(X) @ y
    elif reg_type == "l2":
        return np.linalg.inv(X.T @ X + alpha * np.eye(X.shape[1])) @ X.T @ y
    elif reg_type == "l1":
        # Use coordinate descent for L1 regularization
        return _solve_coordinate_descent(X, y, 1e-4, 1000, "l1", alpha)
    elif reg_type == "elasticnet":
        if l1_ratio is None:
            raise ValueError("l1_ratio must be specified for elasticnet.")
        return _solve_coordinate_descent(X, y, 1e-4, 1000, "elasticnet", alpha, l1_ratio)
    else:
        raise ValueError("Invalid regularisation type specified.")

def _solve_gradient_descent(X: np.ndarray, y: np.ndarray, tol: float, max_iter: int,
                            reg_type: Optional[str], alpha: float) -> np.ndarray:
    """Solve linear model using gradient descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradient = X.T @ (X @ params - y) / len(y)
        if reg_type == "l2":
            gradient += 2 * alpha * params
        elif reg_type == "l1":
            gradient += alpha * np.sign(params)

        params_new = params - learning_rate * gradient

        if np.linalg.norm(params_new - params) < tol:
            break
        params = params_new

    return params

def _solve_newton(X: np.ndarray, y: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
    """Solve linear model using Newton's method."""
    n_features = X.shape[1]
    params = np.zeros(n_features)

    for _ in range(max_iter):
        gradient = X.T @ (X @ params - y) / len(y)
        hessian = X.T @ X / len(y)

        params_new = params - np.linalg.inv(hessian) @ gradient

        if np.linalg.norm(params_new - params) < tol:
            break
        params = params_new

    return params

def _solve_coordinate_descent(X: np.ndarray, y: np.ndarray, tol: float, max_iter: int,
                              reg_type: str, alpha: float, l1_ratio: Optional[float] = None) -> np.ndarray:
    """Solve linear model using coordinate descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residual = y - X @ params + params[j] * X_j

            if reg_type == "l1":
                rho = alpha
            elif reg_type == "elasticnet" and l1_ratio is not None:
                rho = alpha * l1_ratio
            else:
                rho = 0

            candidate = np.append(residual + params[j] * X_j, residual - params[j] * X_j)
            candidate = np.append(candidate, 0)

            if reg_type == "l1" or (reg_type == "elasticnet" and l1_ratio is not None):
                candidate = np.append(candidate, params[j] - rho)
                candidate = np.append(candidate, params[j] + rho)

            params[j] = candidate[np.argmin(np.abs(candidate))]

        if np.linalg.norm(X @ params - y) < tol:
            break

    return params

def _compute_metrics(X: np.ndarray, y: np.ndarray, params: np.ndarray,
                     metrique: Union[str, Callable], custom_metrique: Optional[Callable]) -> Dict:
    """Compute specified metrics for the model."""
    y_pred = X @ params
    metrics = {}

    if custom_metrique is not None:
        metrics["custom"] = custom_metrique(y, y_pred)
    else:
        if metrique == "mse":
            metrics["mse"] = np.mean((y - y_pred) ** 2)
        elif metrique == "mae":
            metrics["mae"] = np.mean(np.abs(y - y_pred))
        elif metrique == "r2":
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            metrics["r2"] = 1 - ss_res / (ss_tot + 1e-8)
        elif metrique == "logloss":
            metrics["logloss"] = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
        else:
            raise ValueError("Invalid metric specified.")

    return metrics

################################################################################
# arbres_decision
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> None:
    """Validate input data and normalizer."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")
    if normalizer is not None:
        try:
            X_normalized = normalizer(X)
        except Exception as e:
            raise ValueError(f"Normalizer function failed: {str(e)}")

def compute_criterion(
    X: np.ndarray,
    y: np.ndarray,
    criterion: str = "gini",
    custom_criterion: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
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

def find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: Optional[np.ndarray] = None,
    criterion: str = "gini",
    custom_criterion: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    max_depth: int = 10
) -> Dict[str, Any]:
    """Find the best split for a node."""
    if feature_indices is None:
        feature_indices = np.arange(X.shape[1])

    best_split = {"feature": None, "threshold": None, "value": float('inf')}

    for feature in feature_indices:
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            left_value = compute_criterion(X[left_mask], y[left_mask], criterion, custom_criterion)
            right_value = compute_criterion(X[right_mask], y[right_mask], criterion, custom_criterion)
            split_value = left_value + right_value

            if split_value < best_split["value"]:
                best_split.update({
                    "feature": feature,
                    "threshold": threshold,
                    "value": split_value
                })

    return best_split

def build_tree(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int = 10,
    min_samples_split: int = 2,
    criterion: str = "gini",
    custom_criterion: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """Recursively build a decision tree."""
    n_samples = X.shape[0]

    if (max_depth == 0 or
        n_samples < min_samples_split or
        len(np.unique(y)) == 1):
        return {"leaf": True, "value": np.bincount(y.astype(int)).argmax()}

    best_split = find_best_split(
        X, y,
        criterion=criterion,
        custom_criterion=custom_criterion,
        max_depth=max_depth
    )

    if best_split["feature"] is None:
        return {"leaf": True, "value": np.bincount(y.astype(int)).argmax()}

    left_mask = X[:, best_split["feature"]] <= best_split["threshold"]
    right_mask = ~left_mask

    left_subtree = build_tree(
        X[left_mask],
        y[left_mask],
        max_depth - 1,
        min_samples_split,
        criterion,
        custom_criterion
    )

    right_subtree = build_tree(
        X[right_mask],
        y[right_mask],
        max_depth - 1,
        min_samples_split,
        criterion,
        custom_criterion
    )

    return {
        "leaf": False,
        "feature": best_split["feature"],
        "threshold": best_split["threshold"],
        "left": left_subtree,
        "right": right_subtree
    }

def predict_tree(
    tree: Dict[str, Any],
    X: np.ndarray
) -> np.ndarray:
    """Predict using a decision tree."""
    predictions = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        node = tree
        while not node["leaf"]:
            if X[i, node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        predictions[i] = node["value"]

    return predictions

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Union[str, list] = "accuracy",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    result = {}

    if isinstance(metrics, str):
        metrics = [metrics]

    for metric in metrics:
        if metric == "accuracy":
            result["accuracy"] = np.mean(y_true == y_pred)
        elif metric == "precision":
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            result["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
        elif metric == "recall":
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            result["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        elif metric == "f1":
            precision = result.get("precision", None)
            recall = result.get("recall", None)
            if precision is not None and recall is not None:
                result["f1"] = 2 * (precision * recall) / (precision + recall)
        elif metric == "custom" and custom_metric is not None:
            result["custom"] = custom_metric(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return result

def arbres_decision_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    criterion: str = "gini",
    custom_criterion: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    max_depth: int = 10,
    min_samples_split: int = 2,
    metrics: Union[str, list] = "accuracy",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Fit a decision tree model.

    Parameters:
    - X: Input features (2D array)
    - y: Target values (1D array)
    - normalizer: Optional normalization function
    - criterion: Splitting criterion ("gini" or "entropy")
    - custom_criterion: Optional custom splitting criterion function
    - max_depth: Maximum tree depth
    - min_samples_split: Minimum samples required to split a node
    - metrics: Evaluation metrics to compute
    - custom_metric: Optional custom evaluation metric function

    Returns:
    - Dictionary containing the fitted tree, metrics, and parameters used
    """
    # Validate inputs
    validate_inputs(X, y, normalizer)

    # Normalize data if specified
    X_normalized = X if normalizer is None else normalizer(X)

    # Build the tree
    tree = build_tree(
        X_normalized,
        y,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        custom_criterion=custom_criterion
    )

    # Make predictions and compute metrics
    y_pred = predict_tree(tree, X_normalized)
    metrics_result = compute_metrics(y, y_pred, metrics, custom_metric)

    return {
        "result": tree,
        "metrics": metrics_result,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer is not None else None,
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split
        },
        "warnings": []
    }

################################################################################
# shap_values
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def shap_values_fit(
    model: Callable,
    X: np.ndarray,
    feature_names: Optional[list] = None,
    normalizer: str = "standard",
    metric: Union[str, Callable] = "mse",
    distance: str = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric_weights: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute SHAP values for a given model and dataset.

    Parameters
    ----------
    model : Callable
        The predictive model for which SHAP values are computed.
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    feature_names : Optional[list], default=None
        Names of the features.
    normalizer : str, default="standard"
        Normalization method: "none", "standard", "minmax", or "robust".
    metric : Union[str, Callable], default="mse"
        Metric to use: "mse", "mae", "r2", "logloss", or custom callable.
    distance : str, default="euclidean"
        Distance metric: "euclidean", "manhattan", "cosine", or "minkowski".
    solver : str, default="closed_form"
        Solver method: "closed_form", "gradient_descent", "newton", or "coordinate_descent".
    regularization : Optional[str], default=None
        Regularization type: "none", "l1", "l2", or "elasticnet".
    tol : float, default=1e-4
        Tolerance for convergence.
    max_iter : int, default=1000
        Maximum number of iterations for iterative solvers.
    custom_metric_weights : Optional[np.ndarray], default=None
        Custom weights for the metric.

    Returns
    -------
    Dict
        Dictionary containing:
            - "result": Computed SHAP values.
            - "metrics": Metrics computed during the process.
            - "params_used": Parameters used in the computation.
            - "warnings": Any warnings generated during the process.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> X = np.random.rand(10, 5)
    >>> model = LinearRegression()
    >>> shap_values_fit(model, X)
    """
    # Validate inputs
    _validate_inputs(X, feature_names)

    # Normalize data
    X_normalized = _normalize_data(X, normalizer)

    # Initialize parameters and results
    params_used = {
        "normalizer": normalizer,
        "metric": metric,
        "distance": distance,
        "solver": solver,
        "regularization": regularization,
        "tol": tol,
        "max_iter": max_iter
    }

    # Compute SHAP values based on solver choice
    if solver == "closed_form":
        shap_values = _compute_shap_closed_form(model, X_normalized, metric, distance)
    elif solver == "gradient_descent":
        shap_values = _compute_shap_gradient_descent(model, X_normalized, metric, distance,
                                                    tol, max_iter, regularization)
    elif solver == "newton":
        shap_values = _compute_shap_newton(model, X_normalized, metric, distance,
                                           tol, max_iter, regularization)
    elif solver == "coordinate_descent":
        shap_values = _compute_shap_coordinate_descent(model, X_normalized, metric, distance,
                                                       tol, max_iter, regularization)
    else:
        raise ValueError("Unsupported solver method.")

    # Compute metrics
    metrics = _compute_metrics(model, X_normalized, metric, custom_metric_weights)

    # Prepare warnings
    warnings = _check_warnings(shap_values, metrics)

    return {
        "result": shap_values,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

def _validate_inputs(X: np.ndarray, feature_names: Optional[list]) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X must not contain NaN or infinite values.")
    if feature_names is not None and len(feature_names) != X.shape[1]:
        raise ValueError("feature_names length must match number of features in X.")

def _normalize_data(X: np.ndarray, normalizer: str) -> np.ndarray:
    """Normalize the input data."""
    if normalizer == "none":
        return X
    elif normalizer == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif normalizer == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif normalizer == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError("Unsupported normalizer method.")

def _compute_shap_closed_form(model: Callable, X: np.ndarray,
                             metric: Union[str, Callable], distance: str) -> np.ndarray:
    """Compute SHAP values using closed-form solution."""
    # Placeholder for actual implementation
    return np.random.rand(X.shape[1])

def _compute_shap_gradient_descent(model: Callable, X: np.ndarray,
                                  metric: Union[str, Callable], distance: str,
                                  tol: float, max_iter: int,
                                  regularization: Optional[str]) -> np.ndarray:
    """Compute SHAP values using gradient descent."""
    # Placeholder for actual implementation
    return np.random.rand(X.shape[1])

def _compute_shap_newton(model: Callable, X: np.ndarray,
                         metric: Union[str, Callable], distance: str,
                         tol: float, max_iter: int,
                         regularization: Optional[str]) -> np.ndarray:
    """Compute SHAP values using Newton's method."""
    # Placeholder for actual implementation
    return np.random.rand(X.shape[1])

def _compute_shap_coordinate_descent(model: Callable, X: np.ndarray,
                                    metric: Union[str, Callable], distance: str,
                                    tol: float, max_iter: int,
                                    regularization: Optional[str]) -> np.ndarray:
    """Compute SHAP values using coordinate descent."""
    # Placeholder for actual implementation
    return np.random.rand(X.shape[1])

def _compute_metrics(model: Callable, X: np.ndarray,
                     metric: Union[str, Callable],
                     custom_metric_weights: Optional[np.ndarray]) -> Dict:
    """Compute metrics for the model."""
    # Placeholder for actual implementation
    return {"metric_value": 0.5}

def _check_warnings(shap_values: np.ndarray, metrics: Dict) -> list:
    """Check for any warnings during computation."""
    warnings = []
    if np.any(np.isnan(shap_values)):
        warnings.append("SHAP values contain NaN.")
    if metrics["metric_value"] < 0:
        warnings.append("Metric value is negative, which might indicate an issue.")
    return warnings

################################################################################
# lime_explications
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    distance_metric: Callable[[np.ndarray, np.ndarray], float],
    feature_selection: str = 'auto',
) -> None:
    """Validate input data and parameters."""
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
    normalization: str = 'standard',
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize data based on the specified method."""
    if normalization == 'none':
        return X, y
    elif normalization == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_normalized = (X - mean) / std
    elif normalization == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_normalized = (X - min_val) / (max_val - min_val + 1e-8)
    elif normalization == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_normalized = (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")
    return X_normalized, y

def _compute_distance(
    X: np.ndarray,
    x: np.ndarray,
    distance_metric: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """Compute distances between samples and the instance to explain."""
    return np.array([distance_metric(xi, x) for xi in X])

def _select_features(
    distances: np.ndarray,
    k: int = 10,
    feature_selection: str = 'auto',
) -> np.ndarray:
    """Select features based on the specified method."""
    if feature_selection == 'auto':
        k = min(k, len(distances))
        indices = np.argsort(distances)[:k]
    elif feature_selection == 'all':
        indices = np.arange(len(distances))
    else:
        raise ValueError(f"Unknown feature selection method: {feature_selection}")
    return indices

def _fit_lime_model(
    X_selected: np.ndarray,
    y_selected: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    solver: str = 'closed_form',
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Fit the LIME model using the specified solver."""
    if solver == 'closed_form':
        coefficients = np.linalg.pinv(X_selected) @ y_selected
    elif solver == 'gradient_descent':
        coefficients = _gradient_descent(X_selected, y_selected)
    elif solver == 'newton':
        coefficients = _newton_method(X_selected, y_selected)
    elif solver == 'coordinate_descent':
        coefficients = _coordinate_descent(X_selected, y_selected)
    else:
        raise ValueError(f"Unknown solver: {solver}")
    return coefficients, {'solver': solver}

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    n_iter: int = 1000,
) -> np.ndarray:
    """Gradient descent solver."""
    coefficients = np.zeros(X.shape[1])
    for _ in range(n_iter):
        gradient = 2 * X.T @ (X @ coefficients - y) / len(y)
        coefficients -= learning_rate * gradient
    return coefficients

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Newton method solver."""
    coefficients = np.zeros(X.shape[1])
    hessian = X.T @ X
    gradient = 2 * X.T @ (X @ coefficients - y)
    coefficients -= np.linalg.inv(hessian) @ gradient
    return coefficients

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    n_iter: int = 1000,
) -> np.ndarray:
    """Coordinate descent solver."""
    coefficients = np.zeros(X.shape[1])
    for _ in range(n_iter):
        for i in range(X.shape[1]):
            X_i = X[:, i]
            coefficients[i] = np.linalg.lstsq(X_i.reshape(-1, 1), y - X @ coefficients + coefficients[i] * X_i, rcond=None)[0]
    return coefficients

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
) -> Dict[str, float]:
    """Compute the specified metrics."""
    return {name: metric(y_true, y_pred) for name, metric in metrics.items()}

def lime_explications_fit(
    X: np.ndarray,
    y: np.ndarray,
    x_to_explain: np.ndarray,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = lambda a, b: np.linalg.norm(a - b),
    normalization: str = 'standard',
    feature_selection: str = 'auto',
    k: int = 10,
    solver: str = 'closed_form',
    metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = None,
) -> Dict[str, Any]:
    """
    Fit LIME explanations for a given instance.

    Parameters:
    - X: Feature matrix (n_samples, n_features)
    - y: Target values (n_samples,)
    - x_to_explain: Instance to explain (n_features,)
    - distance_metric: Distance metric function
    - normalization: Normalization method ('none', 'standard', 'minmax', 'robust')
    - feature_selection: Feature selection method ('auto', 'all')
    - k: Number of features to select
    - solver: Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    - metrics: Dictionary of metric functions to compute

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings
    """
    _validate_inputs(X, y, distance_metric, feature_selection)
    X_normalized, y_normalized = _normalize_data(X, y, normalization)
    distances = _compute_distance(X_normalized, x_to_explain, distance_metric)
    selected_indices = _select_features(distances, k, feature_selection)
    X_selected = X_normalized[selected_indices]
    y_selected = y_normalized[selected_indices]

    # Add intercept term
    X_selected_with_intercept = np.column_stack([np.ones(len(X_selected)), X_selected])
    coefficients, solver_info = _fit_lime_model(X_selected_with_intercept, y_selected, lambda a, b: np.mean((a - b)**2), solver)

    # Compute predictions
    y_pred = X_selected_with_intercept @ coefficients

    # Compute metrics if provided
    computed_metrics = {}
    if metrics is not None:
        computed_metrics = _compute_metrics(y_selected, y_pred, metrics)

    return {
        'result': coefficients,
        'metrics': computed_metrics,
        'params_used': {
            'normalization': normalization,
            'feature_selection': feature_selection,
            'k': k,
            'solver': solver_info['solver'],
        },
        'warnings': [],
    }

################################################################################
# partial_dependence_plots
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def partial_dependence_plots_fit(
    model: Callable,
    X: np.ndarray,
    features: Union[list, np.ndarray],
    feature_ranges: Optional[Dict[str, tuple]] = None,
    n_grid_points: int = 100,
    normalize: str = 'standard',
    metric: Callable = None,
    distance_metric: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: str = 'none',
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute partial dependence plots for given features of a model.

    Parameters:
    -----------
    model : Callable
        The trained model to analyze. Should have a predict method.
    X : np.ndarray
        Training data used to fit the model. Shape (n_samples, n_features).
    features : list or np.ndarray
        Features for which to compute partial dependence. Can be indices or names.
    feature_ranges : dict, optional
        Dictionary mapping features to their value ranges (min, max).
    n_grid_points : int, optional
        Number of points to evaluate on each feature's range.
    normalize : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : Callable, optional
        Custom metric function. If None, uses model's default prediction.
    distance_metric : str, optional
        Distance metric for feature space: 'euclidean', 'manhattan', etc.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', etc.
    regularization : str, optional
        Regularization type: 'none', 'l1', 'l2', or 'elasticnet'.
    tol : float, optional
        Tolerance for solver convergence.
    max_iter : int, optional
        Maximum iterations for iterative solvers.
    random_state : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': Computed partial dependence values.
        - 'metrics': Evaluation metrics.
        - 'params_used': Parameters used in computation.
        - 'warnings': Any warnings encountered.

    Example:
    --------
    >>> model = RandomForestClassifier()
    >>> X_train = np.random.rand(100, 5)
    >>> partial_dependence_plots_fit(model, X_train, features=[0, 2])
    """
    # Validate inputs
    _validate_inputs(X, features, feature_ranges)

    # Normalize data if required
    X_normalized = _normalize_data(X, normalize=normalize)

    # Prepare feature ranges
    feature_ranges = _prepare_feature_ranges(X_normalized, features, feature_ranges)

    # Compute partial dependence
    pdp_results = _compute_partial_dependence(
        model, X_normalized, features, feature_ranges,
        n_grid_points=n_grid_points, metric=metric,
        distance_metric=distance_metric, solver=solver,
        regularization=regularization, tol=tol,
        max_iter=max_iter
    )

    # Prepare output dictionary
    result_dict = {
        'result': pdp_results,
        'metrics': _compute_metrics(model, X_normalized),
        'params_used': {
            'normalize': normalize,
            'metric': metric.__name__ if metric else None,
            'distance_metric': distance_metric,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

    return result_dict

def _validate_inputs(
    X: np.ndarray,
    features: Union[list, np.ndarray],
    feature_ranges: Optional[Dict[str, tuple]]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if any(np.isnan(X).any()):
        raise ValueError("X contains NaN values")
    if any(np.isinf(X).any()):
        raise ValueError("X contains infinite values")

def _normalize_data(
    X: np.ndarray,
    normalize: str = 'standard'
) -> np.ndarray:
    """Normalize data according to specified method."""
    if normalize == 'none':
        return X
    elif normalize == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif normalize == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif normalize == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {normalize}")

def _prepare_feature_ranges(
    X: np.ndarray,
    features: Union[list, np.ndarray],
    feature_ranges: Optional[Dict[str, tuple]]
) -> Dict[int, tuple]:
    """Prepare feature ranges for partial dependence computation."""
    if feature_ranges is None:
        feature_ranges = {
            f: (np.min(X[:, f]), np.max(X[:, f]))
            for f in features
        }
    return feature_ranges

def _compute_partial_dependence(
    model: Callable,
    X: np.ndarray,
    features: Union[list, np.ndarray],
    feature_ranges: Dict[int, tuple],
    n_grid_points: int = 100,
    metric: Callable = None,
    distance_metric: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: str = 'none',
    tol: float = 1e-4,
    max_iter: int = 1000
) -> Dict[int, np.ndarray]:
    """Compute partial dependence for specified features."""
    pdp_results = {}

    for feature in features:
        min_val, max_val = feature_ranges[feature]
        grid_points = np.linspace(min_val, max_val, n_grid_points)

        # Compute partial dependence for this feature
        pdp_values = _compute_single_feature_pdp(
            model, X, feature, grid_points,
            metric=metric, distance_metric=distance_metric,
            solver=solver, regularization=regularization,
            tol=tol, max_iter=max_iter
        )

        pdp_results[feature] = pdp_values

    return pdp_results

def _compute_single_feature_pdp(
    model: Callable,
    X: np.ndarray,
    feature: int,
    grid_points: np.ndarray,
    metric: Callable = None,
    distance_metric: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: str = 'none',
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Compute partial dependence for a single feature."""
    n_samples = X.shape[0]
    pdp_values = np.zeros_like(grid_points)

    for i, value in enumerate(grid_points):
        # Create modified dataset with this feature value
        X_modified = X.copy()
        X_modified[:, feature] = value

        # Compute predictions
        if metric is None:
            predictions = model.predict(X_modified)
        else:
            predictions = _compute_custom_metric(model, X_modified, metric)

        # Average predictions
        pdp_values[i] = np.mean(predictions)

    return pdp_values

def _compute_custom_metric(
    model: Callable,
    X: np.ndarray,
    metric: Callable
) -> np.ndarray:
    """Compute custom metric for model predictions."""
    return np.array([metric(model.predict(x.reshape(1, -1))) for x in X])

def _compute_metrics(
    model: Callable,
    X: np.ndarray
) -> Dict[str, float]:
    """Compute evaluation metrics for the model."""
    predictions = model.predict(X)
    return {
        'mse': np.mean((predictions - X) ** 2),
        'mae': np.mean(np.abs(predictions - X)),
        'r2': 1 - np.sum((predictions - X) ** 2) / np.sum((X - np.mean(X)) ** 2)
    }

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
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_weights: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute feature importance using various statistical and machine learning methods.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalizer : Optional[Callable]
        Function to normalize features. Default is None.
    metric : Union[str, Callable]
        Metric to evaluate feature importance. Can be 'mse', 'mae', 'r2', 'logloss' or custom callable.
    distance : Union[str, Callable]
        Distance metric for feature importance calculation. Can be 'euclidean', 'manhattan', 'cosine', 'minkowski' or custom callable.
    solver : str
        Solver to use. Options: 'closed_form', 'gradient_descent', 'newton', 'coordinate_descent'.
    regularization : Optional[str]
        Regularization type. Options: None, 'l1', 'l2', 'elasticnet'.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_weights : Optional[np.ndarray]
        Custom weights for features.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings.

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = feature_importance_fit(X, y, normalizer=np.std, metric='r2', solver='gradient_descent')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if normalizer is provided
    X_normalized = _apply_normalization(X, normalizer)

    # Apply custom weights if provided
    if custom_weights is not None:
        X_normalized = X_normalized * custom_weights

    # Choose solver
    if solver == 'closed_form':
        coefficients = _solve_closed_form(X_normalized, y)
    elif solver == 'gradient_descent':
        coefficients = _solve_gradient_descent(X_normalized, y, metric, tol, max_iter)
    elif solver == 'newton':
        coefficients = _solve_newton(X_normalized, y, metric, tol, max_iter)
    elif solver == 'coordinate_descent':
        coefficients = _solve_coordinate_descent(X_normalized, y, metric, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if specified
    if regularization == 'l1':
        coefficients = _apply_l1_regularization(coefficients, tol)
    elif regularization == 'l2':
        coefficients = _apply_l2_regularization(coefficients, tol)
    elif regularization == 'elasticnet':
        coefficients = _apply_elasticnet_regularization(coefficients, tol)

    # Compute feature importance
    feature_importance = _compute_feature_importance(coefficients, distance)

    # Compute metrics
    metrics = _compute_metrics(X_normalized, y, coefficients, metric)

    # Prepare results
    result = {
        'result': feature_importance,
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
    """Validate input dimensions and types."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or Inf values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or Inf values")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to features."""
    if normalizer is None:
        return X
    try:
        return normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve for coefficients using closed form solution."""
    XTX = X.T @ X
    if np.linalg.det(XTX) == 0:
        raise ValueError("Matrix is singular")
    XTy = X.T @ y
    return np.linalg.solve(XTX, XTy)

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve for coefficients using gradient descent."""
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = _compute_gradient(X, y, coefficients, metric)
        coefficients -= tol * gradient
    return coefficients

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    metric: Union[str, Callable]
) -> np.ndarray:
    """Compute gradient for given metric."""
    if isinstance(metric, str):
        if metric == 'mse':
            residuals = y - X @ coefficients
            return 2 * X.T @ residuals / len(y)
        elif metric == 'mae':
            residuals = y - X @ coefficients
            return np.sign(residuals).T @ X / len(y)
        elif metric == 'r2':
            residuals = y - X @ coefficients
            return 2 * X.T @ residuals / len(y)
        elif metric == 'logloss':
            predictions = 1 / (1 + np.exp(-X @ coefficients))
            gradient = X.T @ (predictions - y)
            return gradient / len(y)
    else:
        return metric(X, y, coefficients)

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve for coefficients using Newton's method."""
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = _compute_gradient(X, y, coefficients, metric)
        hessian = _compute_hessian(X, y, coefficients, metric)
        if np.linalg.det(hessian) == 0:
            raise ValueError("Hessian is singular")
        coefficients -= np.linalg.solve(hessian, gradient)
    return coefficients

def _compute_hessian(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    metric: Union[str, Callable]
) -> np.ndarray:
    """Compute Hessian for given metric."""
    if isinstance(metric, str):
        if metric == 'mse':
            return 2 * X.T @ X / len(y)
        elif metric == 'mae':
            return np.eye(X.shape[1])  # Approximation for MAE
        elif metric == 'r2':
            return 2 * X.T @ X / len(y)
        elif metric == 'logloss':
            predictions = 1 / (1 + np.exp(-X @ coefficients))
            hessian = X.T @ np.diag(predictions * (1 - predictions)) @ X
            return hessian / len(y)
    else:
        raise NotImplementedError("Custom Hessian computation not implemented")

def _solve_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve for coefficients using coordinate descent."""
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)
    for _ in range(max_iter):
        for i in range(n_features):
            X_i = X[:, i]
            residuals = y - (X @ coefficients - X_i * coefficients[i])
            if isinstance(metric, str):
                if metric == 'mse':
                    coefficients[i] = np.sum(X_i * residuals) / np.sum(X_i ** 2)
                elif metric == 'mae':
                    coefficients[i] = np.median(residuals)  # Approximation for MAE
                elif metric == 'r2':
                    coefficients[i] = np.sum(X_i * residuals) / np.sum(X_i ** 2)
                elif metric == 'logloss':
                    raise NotImplementedError("Logloss not implemented for coordinate descent")
            else:
                raise NotImplementedError("Custom metric not implemented for coordinate descent")
    return coefficients

def _apply_l1_regularization(coefficients: np.ndarray, alpha: float) -> np.ndarray:
    """Apply L1 regularization."""
    return np.sign(coefficients) * np.maximum(np.abs(coefficients) - alpha, 0)

def _apply_l2_regularization(coefficients: np.ndarray, alpha: float) -> np.ndarray:
    """Apply L2 regularization."""
    return coefficients / (1 + alpha * np.linalg.norm(coefficients))

def _apply_elasticnet_regularization(
    coefficients: np.ndarray,
    alpha: float
) -> np.ndarray:
    """Apply elasticnet regularization."""
    l1_coeffs = _apply_l1_regularization(coefficients, alpha)
    l2_coeffs = _apply_l2_regularization(l1_coeffs, alpha)
    return l2_coeffs

def _compute_feature_importance(
    coefficients: np.ndarray,
    distance: Union[str, Callable]
) -> Dict[str, float]:
    """Compute feature importance based on coefficients and distance metric."""
    if isinstance(distance, str):
        if distance == 'euclidean':
            return {f'feature_{i}': abs(coeff) for i, coeff in enumerate(coefficients)}
        elif distance == 'manhattan':
            return {f'feature_{i}': abs(coeff) for i, coeff in enumerate(coefficients)}
        elif distance == 'cosine':
            norm = np.linalg.norm(coefficients)
            if norm == 0:
                return {f'feature_{i}': 0 for i in range(len(coefficients))}
            return {f'feature_{i}': abs(coeff) / norm for i, coeff in enumerate(coefficients)}
        elif distance == 'minkowski':
            return {f'feature_{i}': abs(coeff) for i, coeff in enumerate(coefficients)}
    else:
        return distance(coefficients)

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """Compute metrics based on predictions."""
    predictions = X @ coefficients
    if isinstance(metric, str):
        metrics = {}
        if metric == 'mse':
            metrics['mse'] = np.mean((y - predictions) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(y - predictions))
        elif metric == 'r2':
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        elif metric == 'logloss':
            metrics['logloss'] = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    else:
        metrics = metric(X, y, coefficients)
    return metrics

################################################################################
# decision_rules
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def decision_rules_fit(
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
    **kwargs
) -> Dict[str, Any]:
    """
    Fit decision rules to interpret a model.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    normalizer : Optional[Callable]
        Function to normalize features. Default is None.
    metric : Union[str, Callable]
        Metric to evaluate performance. Can be "mse", "mae", "r2", or a custom callable.
    distance : Union[str, Callable]
        Distance metric for feature space. Can be "euclidean", "manhattan", "cosine", or a custom callable.
    solver : str
        Solver to use. Options: "closed_form", "gradient_descent", "newton", "coordinate_descent".
    regularization : Optional[str]
        Regularization type. Options: None, "l1", "l2", "elasticnet".
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable]
        Custom metric function if not using built-in metrics.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if normalizer is provided
    X_normalized = _apply_normalization(X, normalizer)

    # Choose solver and fit model
    if solver == "closed_form":
        params = _solve_closed_form(X_normalized, y, regularization)
    elif solver == "gradient_descent":
        params = _solve_gradient_descent(X_normalized, y, tol, max_iter, regularization)
    elif solver == "newton":
        params = _solve_newton(X_normalized, y, tol, max_iter, regularization)
    elif solver == "coordinate_descent":
        params = _solve_coordinate_descent(X_normalized, y, tol, max_iter, regularization)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    y_pred = _predict(X_normalized, params)
    metrics = _compute_metrics(y, y_pred, metric, custom_metric)

    # Prepare output
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
        "warnings": _check_warnings(y, y_pred)
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

def _solve_closed_form(X: np.ndarray, y: np.ndarray, regularization: Optional[str]) -> np.ndarray:
    """Solve using closed-form solution."""
    if regularization is None:
        return np.linalg.pinv(X.T @ X) @ X.T @ y
    elif regularization == "l2":
        return np.linalg.inv(X.T @ X + 1e-4 * np.eye(X.shape[1])) @ X.T @ y
    else:
        raise ValueError(f"Unsupported regularization: {regularization}")

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int,
    regularization: Optional[str]
) -> np.ndarray:
    """Solve using gradient descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradient = _compute_gradient(X, y, params, regularization)
        params -= learning_rate * gradient
        if np.linalg.norm(gradient) < tol:
            break

    return params

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    regularization: Optional[str]
) -> np.ndarray:
    """Compute gradient for gradient descent."""
    residuals = X @ params - y
    gradient = X.T @ residuals / len(y)

    if regularization == "l1":
        gradient += np.sign(params)
    elif regularization == "l2":
        gradient += 2 * params

    return gradient

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int,
    regularization: Optional[str]
) -> np.ndarray:
    """Solve using Newton's method."""
    n_features = X.shape[1]
    params = np.zeros(n_features)

    for _ in range(max_iter):
        gradient = _compute_gradient(X, y, params, regularization)
        hessian = X.T @ X / len(y)

        if regularization == "l2":
            hessian += 2 * np.eye(n_features)

        params -= np.linalg.pinv(hessian) @ gradient

        if np.linalg.norm(gradient) < tol:
            break

    return params

def _solve_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int,
    regularization: Optional[str]
) -> np.ndarray:
    """Solve using coordinate descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - (X @ params - X_j * params[j])

            if regularization == "l1":
                params[j] = np.sign(X_j.T @ residuals) * np.maximum(
                    np.abs(X_j.T @ residuals) - 1, 0
                ) / (X_j.T @ X_j)
            else:
                params[j] = (X_j.T @ residuals) / (X_j.T @ X_j)

        if np.linalg.norm(_compute_gradient(X, y, params, regularization)) < tol:
            break

    return params

def _predict(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Predict using fitted parameters."""
    return X @ params

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict[str, float]:
    """Compute metrics for evaluation."""
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
    elif callable(metric):
        metrics["custom"] = metric(y_true, y_pred)

    if custom_metric is not None:
        metrics["custom"] = custom_metric(y_true, y_pred)

    return metrics

def _check_warnings(y_true: np.ndarray, y_pred: np.ndarray) -> list:
    """Check for potential warnings."""
    warnings = []

    if np.any(np.isnan(y_pred)):
        warnings.append("Predictions contain NaN values.")
    if np.any(np.isinf(y_pred)):
        warnings.append("Predictions contain infinite values.")

    return warnings

# Example usage:
"""
X = np.random.rand(100, 5)
y = np.random.rand(100)

result = decision_rules_fit(
    X, y,
    normalizer=None,
    metric="mse",
    distance="euclidean",
    solver="closed_form",
    regularization=None,
    tol=1e-4,
    max_iter=1000
)
"""

################################################################################
# saliency_maps
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    normalize: str = "standard",
    metric: Union[str, Callable] = "mse",
    distance: str = "euclidean"
) -> None:
    """
    Validate input data and parameters.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    model : Callable
        Predictive model function.
    normalize : str, optional
        Normalization method (default is "standard").
    metric : Union[str, Callable], optional
        Metric function or name (default is "mse").
    distance : str, optional
        Distance metric for saliency calculation (default is "euclidean").

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
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

    if normalize not in ["none", "standard", "minmax", "robust"]:
        raise ValueError("Invalid normalization method")
    if isinstance(metric, str) and metric not in ["mse", "mae", "r2", "logloss"]:
        raise ValueError("Invalid metric name")
    if distance not in ["euclidean", "manhattan", "cosine", "minkowski"]:
        raise ValueError("Invalid distance metric")

def normalize_data(
    X: np.ndarray,
    method: str = "standard"
) -> np.ndarray:
    """
    Normalize input data.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    method : str, optional
        Normalization method (default is "standard").

    Returns
    -------
    np.ndarray
        Normalized data.
    """
    if method == "none":
        return X

    if method == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)

    if method == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)

    if method == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)

def compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable] = "mse"
) -> float:
    """
    Compute evaluation metric.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values.
    metric : Union[str, Callable], optional
        Metric function or name (default is "mse").

    Returns
    -------
    float
        Computed metric value.
    """
    if isinstance(metric, str):
        if metric == "mse":
            return np.mean((y_true - y_pred) ** 2)
        if metric == "mae":
            return np.mean(np.abs(y_true - y_pred))
        if metric == "r2":
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / (ss_tot + 1e-8))
        if metric == "logloss":
            y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    return metric(y_true, y_pred)

def compute_distance(
    X: np.ndarray,
    distance: str = "euclidean",
    p: float = 2.0
) -> np.ndarray:
    """
    Compute distance matrix.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    distance : str, optional
        Distance metric (default is "euclidean").
    p : float, optional
        Power parameter for Minkowski distance (default is 2.0).

    Returns
    -------
    np.ndarray
        Distance matrix of shape (n_samples, n_samples).
    """
    if distance == "euclidean":
        return np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
    if distance == "manhattan":
        return np.sum(np.abs(X[:, np.newaxis] - X), axis=2)
    if distance == "cosine":
        dot_products = np.dot(X, X.T)
        norms = np.sqrt(np.sum(X ** 2, axis=1))[:, np.newaxis]
        return 1 - (dot_products / (norms * norms.T + 1e-8))
    if distance == "minkowski":
        return np.sum(np.abs(X[:, np.newaxis] - X) ** p, axis=2) ** (1/p)

def saliency_maps_fit(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    normalize: str = "standard",
    metric: Union[str, Callable] = "mse",
    distance: str = "euclidean",
    solver: str = "gradient_descent",
    regularization: Optional[str] = None,
    **solver_kwargs
) -> Dict:
    """
    Compute saliency maps for model interpretability.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    model : Callable
        Predictive model function.
    normalize : str, optional
        Normalization method (default is "standard").
    metric : Union[str, Callable], optional
        Metric function or name (default is "mse").
    distance : str, optional
        Distance metric for saliency calculation (default is "euclidean").
    solver : str, optional
        Optimization method (default is "gradient_descent").
    regularization : Optional[str], optional
        Regularization type (default is None).
    **solver_kwargs : dict
        Additional solver parameters.

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    validate_inputs(X, y, model, normalize, metric, distance)

    # Normalize data
    X_normalized = normalize_data(X, normalize)

    # Compute predictions
    y_pred = model(X_normalized)

    # Compute base metric
    base_metric = compute_metric(y, y_pred, metric)

    # Compute distance matrix
    dist_matrix = compute_distance(X_normalized, distance)

    # Initialize saliency maps
    saliency_maps = np.zeros_like(X_normalized, dtype=float)

    # Compute saliency for each feature
    for i in range(X_normalized.shape[1]):
        # Perturb feature values
        X_perturbed = X_normalized.copy()
        X_perturbed[:, i] += 1e-4

        # Compute perturbed predictions
        y_perturbed = model(X_perturbed)

        # Compute metric difference
        perturbed_metric = compute_metric(y, y_perturbed, metric)
        saliency_maps[:, i] = (base_metric - perturbed_metric) / 1e-4

    # Apply regularization if specified
    if regularization == "l1":
        saliency_maps = np.sign(saliency_maps) * np.abs(saliency_maps) - solver_kwargs.get("alpha", 0.1)
    elif regularization == "l2":
        saliency_maps = saliency_maps / (1 + solver_kwargs.get("alpha", 0.1) * np.sum(saliency_maps**2, axis=1)[:, np.newaxis])

    # Return results
    return {
        "result": saliency_maps,
        "metrics": {"base_metric": base_metric},
        "params_used": {
            "normalize": normalize,
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularization": regularization
        },
        "warnings": []
    }

# Example usage:
"""
from sklearn.linear_model import LinearRegression

X = np.random.rand(100, 5)
y = X.dot(np.array([2.0, -1.5, 3.0, 0.5, -2.0])) + np.random.normal(0, 0.1, 100)
model = LinearRegression().fit(X, y)

result = saliency_maps_fit(
    X=X,
    y=y,
    model=lambda x: model.predict(x),
    normalize="standard",
    metric="mse",
    distance="euclidean"
)
"""

################################################################################
# anchors_explications
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def anchors_explications_fit(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Callable = None,
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Fit anchor explanations for a given model.

    Parameters:
    -----------
    model : Callable
        The predictive model to explain.
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    normalizer : Callable, optional
        Function to normalize the data. Default is None.
    metric : str or Callable, optional
        Metric to evaluate explanations. Default is 'mse'.
    distance : str, optional
        Distance metric for anchor search. Default is 'euclidean'.
    solver : str, optional
        Solver to use for optimization. Default is 'closed_form'.
    regularization : str, optional
        Regularization type. Default is None.
    tol : float, optional
        Tolerance for convergence. Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    custom_metric : Callable, optional
        Custom metric function. Default is None.
    custom_distance : Callable, optional
        Custom distance function. Default is None.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Prepare metric and distance functions
    metric_func = _get_metric_function(metric, custom_metric)
    distance_func = _get_distance_function(distance, custom_distance)

    # Fit the model
    anchors = _fit_anchors(
        model, X_normalized, y,
        metric_func=metric_func,
        distance_func=distance_func,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter
    )

    # Calculate metrics
    metrics = _calculate_metrics(model, X_normalized, y, metric_func)

    # Prepare output
    result = {
        'result': anchors,
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

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to the data."""
    if normalizer is None:
        return X
    return normalizer(X)

def _get_metric_function(metric: Union[str, Callable], custom_metric: Optional[Callable]) -> Callable:
    """Get the metric function based on input."""
    if custom_metric is not None:
        return custom_metric
    metric_functions = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared,
        'logloss': _log_loss
    }
    if isinstance(metric, str):
        return metric_functions.get(metric.lower(), _mean_squared_error)
    raise ValueError(f"Unsupported metric: {metric}")

def _get_distance_function(distance: str, custom_distance: Optional[Callable]) -> Callable:
    """Get the distance function based on input."""
    if custom_distance is not None:
        return custom_distance
    distance_functions = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance,
        'minkowski': _minkowski_distance
    }
    return distance_functions.get(distance.lower(), _euclidean_distance)

def _fit_anchors(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    *,
    metric_func: Callable,
    distance_func: Callable,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Fit anchor explanations using the specified solver."""
    solvers = {
        'closed_form': _closed_form_solver,
        'gradient_descent': _gradient_descent_solver,
        'newton': _newton_solver,
        'coordinate_descent': _coordinate_descent_solver
    }
    solver_func = solvers.get(solver.lower(), _closed_form_solver)
    return solver_func(model, X, y, metric_func, distance_func, regularization, tol, max_iter)

def _calculate_metrics(model: Callable, X: np.ndarray, y: np.ndarray, metric_func: Callable) -> Dict[str, float]:
    """Calculate metrics for the model."""
    y_pred = np.array([model(x) for x in X])
    return {
        'metric_value': metric_func(y, y_pred),
        'r2_score': _r_squared(y, y_pred)
    }

# Example metric functions
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
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Example distance functions
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
    return np.sum(np.abs(a - b) ** p) ** (1 / p)

# Example solver functions
def _closed_form_solver(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable,
    distance_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Closed form solver for anchor explanations."""
    # Placeholder implementation
    return np.zeros(X.shape[1])

def _gradient_descent_solver(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable,
    distance_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Gradient descent solver for anchor explanations."""
    # Placeholder implementation
    return np.zeros(X.shape[1])

def _newton_solver(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable,
    distance_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Newton solver for anchor explanations."""
    # Placeholder implementation
    return np.zeros(X.shape[1])

def _coordinate_descent_solver(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable,
    distance_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Coordinate descent solver for anchor explanations."""
    # Placeholder implementation
    return np.zeros(X.shape[1])

################################################################################
# counterfactual_exemples
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def counterfactual_exemples_fit(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    target_value: float,
    distance_metric: str = 'euclidean',
    normalization: Optional[str] = None,
    solver: str = 'gradient_descent',
    max_iter: int = 1000,
    tol: float = 1e-4,
    metric: str = 'mse',
    custom_metric: Optional[Callable] = None,
    regularization: Optional[str] = None,
    reg_param: float = 1.0,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute counterfactual examples for a given model and target value.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    model : Callable
        The trained model with a predict method.
    target_value : float
        The desired target value for counterfactual examples.
    distance_metric : str, optional
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski').
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    solver : str, optional
        Solver to use ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for stopping criterion.
    metric : str, optional
        Metric to evaluate counterfactual examples ('mse', 'mae', 'r2', 'logloss').
    custom_metric : Callable, optional
        Custom metric function.
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet').
    reg_param : float, optional
        Regularization parameter.
    **kwargs :
        Additional solver-specific parameters.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_normalized = _apply_normalization(X, normalization)

    # Initialize counterfactual examples
    counterfactuals = np.zeros_like(X_normalized)

    # Choose distance metric function
    distance_func = _get_distance_function(distance_metric)

    # Choose solver
    if solver == 'closed_form':
        counterfactuals = _closed_form_solver(X_normalized, y, model, target_value, distance_func)
    elif solver == 'gradient_descent':
        counterfactuals = _gradient_descent_solver(
            X_normalized, y, model, target_value, distance_func,
            max_iter=max_iter, tol=tol, reg_type=regularization, reg_param=reg_param,
            **kwargs
        )
    elif solver == 'newton':
        counterfactuals = _newton_solver(
            X_normalized, y, model, target_value, distance_func,
            max_iter=max_iter, tol=tol, reg_type=regularization, reg_param=reg_param,
            **kwargs
        )
    elif solver == 'coordinate_descent':
        counterfactuals = _coordinate_descent_solver(
            X_normalized, y, model, target_value, distance_func,
            max_iter=max_iter, tol=tol, reg_type=regularization, reg_param=reg_param,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(
        X_normalized, counterfactuals, y, target_value,
        metric=metric, custom_metric=custom_metric
    )

    # Return results
    return {
        'result': counterfactuals,
        'metrics': metrics,
        'params_used': {
            'distance_metric': distance_metric,
            'normalization': normalization,
            'solver': solver,
            'max_iter': max_iter,
            'tol': tol,
            'metric': metric,
            'regularization': regularization,
            'reg_param': reg_param
        },
        'warnings': _check_warnings(counterfactuals, metrics)
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

def _apply_normalization(X: np.ndarray, method: Optional[str]) -> np.ndarray:
    """Apply normalization to input data."""
    if method is None or method == 'none':
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

def _get_distance_function(metric: str) -> Callable:
    """Get distance function based on metric name."""
    if metric == 'euclidean':
        return lambda a, b: np.linalg.norm(a - b)
    elif metric == 'manhattan':
        return lambda a, b: np.sum(np.abs(a - b))
    elif metric == 'cosine':
        return lambda a, b: 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    elif metric == 'minkowski':
        return lambda a, b: np.sum(np.abs(a - b)**3)**(1/3)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _closed_form_solver(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    target_value: float,
    distance_func: Callable
) -> np.ndarray:
    """Closed form solution for counterfactual examples."""
    # This is a placeholder - actual implementation would depend on the model
    return np.zeros_like(X)

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    target_value: float,
    distance_func: Callable,
    max_iter: int = 1000,
    tol: float = 1e-4,
    reg_type: Optional[str] = None,
    reg_param: float = 1.0,
    **kwargs
) -> np.ndarray:
    """Gradient descent solver for counterfactual examples."""
    # This is a placeholder - actual implementation would depend on the model
    return np.zeros_like(X)

def _newton_solver(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    target_value: float,
    distance_func: Callable,
    max_iter: int = 1000,
    tol: float = 1e-4,
    reg_type: Optional[str] = None,
    reg_param: float = 1.0,
    **kwargs
) -> np.ndarray:
    """Newton's method solver for counterfactual examples."""
    # This is a placeholder - actual implementation would depend on the model
    return np.zeros_like(X)

def _coordinate_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    target_value: float,
    distance_func: Callable,
    max_iter: int = 1000,
    tol: float = 1e-4,
    reg_type: Optional[str] = None,
    reg_param: float = 1.0,
    **kwargs
) -> np.ndarray:
    """Coordinate descent solver for counterfactual examples."""
    # This is a placeholder - actual implementation would depend on the model
    return np.zeros_like(X)

def _compute_metrics(
    X: np.ndarray,
    counterfactuals: np.ndarray,
    y: np.ndarray,
    target_value: float,
    metric: str = 'mse',
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """Compute metrics for counterfactual examples."""
    metrics = {}

    if custom_metric is not None:
        metrics['custom'] = custom_metric(counterfactuals, y)

    if metric == 'mse':
        metrics['mse'] = np.mean((counterfactuals - target_value) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(counterfactuals - target_value))
    elif metric == 'r2':
        ss_res = np.sum((y - counterfactuals) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)
    elif metric == 'logloss':
        metrics['logloss'] = -np.mean(y * np.log(counterfactuals) + (1 - y) * np.log(1 - counterfactuals))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

def _check_warnings(counterfactuals: np.ndarray, metrics: Dict[str, float]) -> list:
    """Check for potential warnings in the results."""
    warnings = []

    if np.any(np.isnan(counterfactuals)):
        warnings.append("Counterfactual examples contain NaN values")

    if np.any(np.isinf(counterfactuals)):
        warnings.append("Counterfactual examples contain infinite values")

    if 'mse' in metrics and metrics['mse'] > 1e6:
        warnings.append("High MSE in counterfactual examples")

    return warnings

################################################################################
# attention_mechanisms
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    attention_weights: Optional[np.ndarray] = None
) -> None:
    """Validate input dimensions and types."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if attention_weights is not None and not isinstance(attention_weights, np.ndarray):
        raise TypeError("attention_weights must be a numpy array or None")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y is not None and X.shape[0] != len(y):
        raise ValueError("X and y must have the same number of samples")
    if attention_weights is not None:
        if X.shape[1] != attention_weights.shape[0]:
            raise ValueError("X and attention_weights dimensions are incompatible")

def _normalize_data(
    X: np.ndarray,
    method: str = "standard",
    custom_func: Optional[Callable] = None
) -> np.ndarray:
    """Normalize input data using specified method."""
    if custom_func is not None:
        return custom_func(X)

    X_norm = X.copy()
    if method == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_norm = (X - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - median) / (iqr + 1e-8)
    return X_norm

def _compute_attention_scores(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    attention_weights: Optional[np.ndarray] = None,
    distance_metric: str = "euclidean",
    custom_distance: Optional[Callable] = None
) -> np.ndarray:
    """Compute attention scores based on specified distance metric."""
    if custom_distance is not None:
        return custom_distance(X, y)

    n_samples = X.shape[0]
    attention_scores = np.zeros((n_samples, X.shape[1]))

    for i in range(n_samples):
        if distance_metric == "euclidean":
            distances = np.linalg.norm(X - X[i], axis=1)
        elif distance_metric == "manhattan":
            distances = np.sum(np.abs(X - X[i]), axis=1)
        elif distance_metric == "cosine":
            distances = 1 - np.dot(X, X[i]) / (np.linalg.norm(X, axis=1) * np.linalg.norm(X[i]))
        elif distance_metric == "minkowski":
            distances = np.sum(np.abs(X - X[i])**3, axis=1)**(1/3)

        if y is not None:
            # Weight by target similarity
            target_diff = np.abs(y - y[i])
            distances = distances * (1 + 0.1 * target_diff)

        if attention_weights is not None:
            distances = distances * (1 + 0.1 * attention_weights)

        # Softmax to get probabilities
        exp_distances = np.exp(-distances)
        attention_scores[i] = exp_distances / np.sum(exp_distances)

    return attention_scores

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_names: Union[str, list] = "mse",
    custom_metrics: Optional[Dict[str, Callable]] = None
) -> Dict[str, float]:
    """Compute specified metrics between true and predicted values."""
    if isinstance(metric_names, str):
        metric_names = [metric_names]

    metrics = {}

    if "mse" in metric_names:
        metrics["mse"] = np.mean((y_true - y_pred) ** 2)
    if "mae" in metric_names:
        metrics["mae"] = np.mean(np.abs(y_true - y_pred))
    if "r2" in metric_names:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics["r2"] = 1 - (ss_res / ss_tot)

    if custom_metrics is not None:
        for name, func in custom_metrics.items():
            metrics[name] = func(y_true, y_pred)

    return metrics

def attention_mechanisms_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    attention_weights: Optional[np.ndarray] = None,
    normalization: str = "standard",
    custom_normalize: Optional[Callable] = None,
    distance_metric: str = "euclidean",
    custom_distance: Optional[Callable] = None,
    metrics: Union[str, list] = "mse",
    custom_metrics: Optional[Dict[str, Callable]] = None,
    **kwargs
) -> Dict:
    """Main function to compute attention mechanisms and interpretability metrics.

    Example:
        >>> X = np.random.rand(100, 5)
        >>> y = np.random.rand(100)
        >>> result = attention_mechanisms_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y, attention_weights)

    # Normalize data
    X_norm = _normalize_data(X, normalization, custom_normalize)

    # Compute attention scores
    attention_scores = _compute_attention_scores(
        X_norm, y, attention_weights,
        distance_metric, custom_distance
    )

    # Compute predictions (weighted average)
    y_pred = np.dot(attention_scores, X) if y is None else np.dot(attention_scores, X).mean(axis=0)

    # Compute metrics
    result_metrics = {}
    if y is not None:
        result_metrics = _compute_metrics(y, y_pred, metrics, custom_metrics)

    return {
        "result": {
            "attention_scores": attention_scores,
            "predictions": y_pred if y is not None else None
        },
        "metrics": result_metrics,
        "params_used": {
            "normalization": normalization if custom_normalize is None else "custom",
            "distance_metric": distance_metric if custom_distance is None else "custom",
            "metrics": metrics
        },
        "warnings": []
    }

################################################################################
# sparse_models
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def sparse_models_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'coordinate_descent',
    regularization: str = 'l1',
    tol: float = 1e-4,
    max_iter: int = 1000,
    alpha: float = 1.0,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Fit sparse models for interpretability.

    Parameters:
    - X: Input features (n_samples, n_features)
    - y: Target values (n_samples,)
    - normalizer: Function to normalize features
    - metric: Metric for evaluation ('mse', 'mae', 'r2', or custom callable)
    - distance: Distance metric ('euclidean', 'manhattan', etc., or custom callable)
    - solver: Solver method ('closed_form', 'gradient_descent', etc.)
    - regularization: Regularization type ('none', 'l1', 'l2', etc.)
    - tol: Tolerance for convergence
    - max_iter: Maximum iterations
    - alpha: Regularization strength
    - custom_metric: Custom metric function
    - custom_distance: Custom distance function

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features
    X_normalized = normalizer(X)

    # Choose solver and fit model
    if solver == 'coordinate_descent':
        coefs, metrics = _coordinate_descent(
            X_normalized, y,
            metric=metric,
            distance=distance,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter,
            alpha=alpha,
            custom_metric=custom_metric,
            custom_distance=custom_distance
        )
    elif solver == 'gradient_descent':
        coefs, metrics = _gradient_descent(
            X_normalized, y,
            metric=metric,
            distance=distance,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter,
            alpha=alpha,
            custom_metric=custom_metric,
            custom_distance=custom_distance
        )
    else:
        raise ValueError(f"Unsupported solver: {solver}")

    # Prepare output
    result = {
        'result': {'coefficients': coefs},
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter,
            'alpha': alpha
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    regularization: str,
    tol: float,
    max_iter: int,
    alpha: float,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]],
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> tuple:
    """Coordinate descent solver for sparse models."""
    n_samples, n_features = X.shape
    coefs = np.zeros(n_features)
    metrics_history = []

    for _ in range(max_iter):
        old_coefs = coefs.copy()

        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - np.dot(X, coefs) + coefs[j] * X_j

            if regularization == 'l1':
                coefs[j] = _soft_threshold(residuals.dot(X_j), alpha * np.linalg.norm(X_j, 1))
            elif regularization == 'l2':
                coefs[j] = residuals.dot(X_j) / (np.linalg.norm(X_j, 2)**2 + alpha)
            else:
                coefs[j] = residuals.dot(X_j) / np.linalg.norm(X_j, 2)**2

        # Calculate metrics
        y_pred = np.dot(X, coefs)
        if custom_metric:
            current_metric = custom_metric(y, y_pred)
        else:
            if metric == 'mse':
                current_metric = np.mean((y - y_pred) ** 2)
            elif metric == 'mae':
                current_metric = np.mean(np.abs(y - y_pred))
            elif metric == 'r2':
                current_metric = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        metrics_history.append(current_metric)

        if np.linalg.norm(coefs - old_coefs) < tol:
            break

    return coefs, {'history': metrics_history}

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    regularization: str,
    tol: float,
    max_iter: int,
    alpha: float,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]],
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> tuple:
    """Gradient descent solver for sparse models."""
    n_samples, n_features = X.shape
    coefs = np.zeros(n_features)
    metrics_history = []
    learning_rate = 0.1

    for _ in range(max_iter):
        old_coefs = coefs.copy()

        gradient = np.dot(X.T, (np.dot(X, coefs) - y)) / n_samples

        if regularization == 'l1':
            gradient += alpha * np.sign(coefs)
        elif regularization == 'l2':
            gradient += 2 * alpha * coefs
        elif regularization == 'elasticnet':
            gradient += alpha * (np.sign(coefs) + 2 * coefs)

        coefs -= learning_rate * gradient

        # Calculate metrics
        y_pred = np.dot(X, coefs)
        if custom_metric:
            current_metric = custom_metric(y, y_pred)
        else:
            if metric == 'mse':
                current_metric = np.mean((y - y_pred) ** 2)
            elif metric == 'mae':
                current_metric = np.mean(np.abs(y - y_pred))
            elif metric == 'r2':
                current_metric = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        metrics_history.append(current_metric)

        if np.linalg.norm(coefs - old_coefs) < tol:
            break

    return coefs, {'history': metrics_history}

def _soft_threshold(rho: float, lambda_: float) -> float:
    """Soft thresholding operator."""
    if rho < -lambda_:
        return rho + lambda_
    elif rho > lambda_:
        return rho - lambda_
    else:
        return 0.0

################################################################################
# bayesian_rules
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def bayesian_rules_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
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
    Fit Bayesian rules model with configurable parameters.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize features. Default is identity.
    metric : str
        Metric to evaluate model performance. Options: 'mse', 'mae', 'r2'.
    distance : str
        Distance metric for rule generation. Options: 'euclidean', 'manhattan', 'cosine'.
    solver : str
        Solver method. Options: 'closed_form', 'gradient_descent'.
    regularization : Optional[str]
        Regularization type. Options: None, 'l1', 'l2'.
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
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = bayesian_rules_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features
    X_norm = normalizer(X)

    # Set metric and distance functions
    metric_func, distance_func = _set_metric_distance(metric, distance, custom_metric, custom_distance)

    # Solve model
    params = _solve_model(
        X_norm, y,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter
    )

    # Calculate metrics
    metrics = _calculate_metrics(X_norm, y, params, metric_func)

    # Prepare output
    result = {
        'result': {'params': params},
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
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
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _set_metric_distance(
    metric: str,
    distance: str,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]],
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> tuple:
    """Set metric and distance functions."""
    if custom_metric is not None:
        metric_func = custom_metric
    else:
        metric_func = _get_metric_function(metric)

    if custom_distance is not None:
        distance_func = custom_distance
    else:
        distance_func = _get_distance_function(distance)

    return metric_func, distance_func

def _get_metric_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get metric function based on input string."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _get_distance_function(distance: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get distance function based on input string."""
    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance
    }
    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")
    return distances[distance]

def _solve_model(
    X: np.ndarray,
    y: np.ndarray,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict:
    """Solve model with specified solver and regularization."""
    if solver == 'closed_form':
        params = _solve_closed_form(X, y, regularization)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(X, y, regularization, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    return params

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Calculate metrics for the model."""
    y_pred = _predict(X, params)
    return {'metric': metric_func(y, y_pred)}

def _solve_closed_form(X: np.ndarray, y: np.ndarray, regularization: Optional[str]) -> Dict:
    """Solve model using closed form solution."""
    if regularization is None:
        params = np.linalg.pinv(X) @ y
    elif regularization == 'l2':
        params = _ridge_regression(X, y)
    else:
        raise ValueError(f"Unknown regularization: {regularization}")
    return {'coefficients': params}

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict:
    """Solve model using gradient descent."""
    # Simplified implementation for demonstration
    n_features = X.shape[1]
    params = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ params - y) / len(y)
        if regularization == 'l2':
            gradient += 2 * params
        params -= tol * gradient
    return {'coefficients': params}

def _predict(X: np.ndarray, params: Dict) -> np.ndarray:
    """Make predictions using model parameters."""
    return X @ params['coefficients']

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
    return np.linalg.inv(X.T @ X + 1e-4 * np.eye(X.shape[1])) @ X.T @ y

################################################################################
# surrogate_models
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def surrogate_models_fit(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "linear",
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Fit surrogate models for interpretability.

    Parameters:
    - X: Input features (n_samples, n_features)
    - y: Target values (n_samples,)
    - model_type: Type of surrogate model ("linear", "tree")
    - normalizer: Normalization function
    - metric: Metric for evaluation ("mse", "mae", "r2", custom callable)
    - distance: Distance metric ("euclidean", "manhattan", etc.)
    - solver: Solver method
    - regularization: Regularization type ("l1", "l2", "elasticnet")
    - tol: Tolerance for convergence
    - max_iter: Maximum iterations
    - custom_metric: Custom metric function

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_norm = _apply_normalization(X, normalizer)

    # Choose model and fit
    if model_type == "linear":
        result = _fit_linear_model(X_norm, y, solver, regularization, tol, max_iter)
    else:
        raise ValueError("Unsupported model type")

    # Compute metrics
    metrics = _compute_metrics(y, result["predictions"], metric, custom_metric)

    return {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "model_type": model_type,
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
    """Apply normalization to input features."""
    if normalizer is None:
        return X
    return normalizer(X)

def _fit_linear_model(
    X: np.ndarray,
    y: np.ndarray,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Fit a linear model with specified solver and regularization."""
    if solver == "closed_form":
        params = _solve_closed_form(X, y, regularization)
    else:
        raise ValueError("Unsupported solver")

    predictions = X @ params

    return {
        "params": params,
        "predictions": predictions
    }

def _solve_closed_form(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str]
) -> np.ndarray:
    """Solve linear regression in closed form with optional regularization."""
    if regularization is None:
        return np.linalg.pinv(X.T @ X) @ X.T @ y
    elif regularization == "l2":
        return np.linalg.pinv(X.T @ X + 1e-4 * np.eye(X.shape[1])) @ X.T @ y
    else:
        raise ValueError("Unsupported regularization type")

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
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
            raise ValueError("Unsupported metric")

    if custom_metric is not None:
        metrics["custom"] = custom_metric(y_true, y_pred)

    return metrics

# Example usage:
"""
X_example = np.random.rand(100, 5)
y_example = X_example @ np.array([1.2, -0.8, 0.5, 0.3, -0.1]) + np.random.normal(0, 0.1, 100)

result = surrogate_models_fit(
    X=X_example,
    y=y_example,
    model_type="linear",
    normalizer=None,
    metric="mse"
)
"""

################################################################################
# global_explanations
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray],
    metric: Callable[[np.ndarray, np.ndarray], float]
) -> None:
    """
    Validate input data and functions.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalizer : Callable[[np.ndarray], np.ndarray]
        Normalization function
    metric : Callable[[np.ndarray, np.ndarray], float]
        Metric function

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
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def compute_statistic(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray],
    distance_metric: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, Any]:
    """
    Compute global explanation statistics.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalizer : Callable[[np.ndarray], np.ndarray]
        Normalization function
    distance_metric : Callable[[np.ndarray, np.ndarray], float]
        Distance metric function

    Returns
    -------
    Dict[str, Any]
        Dictionary containing computed statistics
    """
    X_norm = normalizer(X)
    n_samples, n_features = X.shape

    # Compute pairwise distances
    distance_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            dist = distance_metric(X_norm[i], X_norm[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    # Compute feature importance based on distances
    feature_importance = np.zeros(n_features)
    for f in range(n_features):
        feature_diff = X_norm[:, f].reshape(-1, 1) - X_norm[:, f]
        feature_importance[f] = np.mean(np.abs(feature_diff))

    return {
        "distance_matrix": distance_matrix,
        "feature_importance": feature_importance
    }

def estimate_parameters(
    X: np.ndarray,
    y: np.ndarray,
    solver: Callable[[np.ndarray, np.ndarray], np.ndarray],
    regularizer: Optional[Callable[[np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Estimate model parameters.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    solver : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Solver function
    regularizer : Optional[Callable[[np.ndarray], float]]
        Regularization function

    Returns
    -------
    Dict[str, Any]
        Dictionary containing estimated parameters
    """
    params = solver(X, y)

    if regularizer is not None:
        penalty = regularizer(params)
    else:
        penalty = 0.0

    return {
        "params": params,
        "penalty": penalty
    }

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    custom_metrics: Optional[Dict[str, Callable]] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True target values of shape (n_samples,)
    y_pred : np.ndarray
        Predicted target values of shape (n_samples,)
    metric : Callable[[np.ndarray, np.ndarray], float]
        Primary metric function
    custom_metrics : Optional[Dict[str, Callable]]
        Dictionary of additional metrics

    Returns
    -------
    Dict[str, float]
        Dictionary containing computed metrics
    """
    metrics = {
        "primary_metric": metric(y_true, y_pred)
    }

    if custom_metrics is not None:
        for name, func in custom_metrics.items():
            metrics[name] = func(y_true, y_pred)

    return metrics

def global_explanations_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: Callable[[np.ndarray, np.ndarray], float] = lambda y_true, y_pred: np.mean((y_true - y_pred)**2),
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.linalg.norm(x - y),
    solver: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda X, y: np.linalg.pinv(X) @ y,
    regularizer: Optional[Callable[[np.ndarray], float]] = None,
    custom_metrics: Optional[Dict[str, Callable]] = None
) -> Dict[str, Any]:
    """
    Compute global explanations for a given dataset.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalizer : Callable[[np.ndarray], np.ndarray], optional
        Normalization function, by default identity
    metric : Callable[[np.ndarray, np.ndarray], float], optional
        Primary evaluation metric, by default MSE
    distance_metric : Callable[[np.ndarray, np.ndarray], float], optional
        Distance metric function, by default Euclidean distance
    solver : Callable[[np.ndarray, np.ndarray], np.ndarray], optional
        Solver function, by default pseudo-inverse
    regularizer : Optional[Callable[[np.ndarray], float]], optional
        Regularization function, by default None
    custom_metrics : Optional[Dict[str, Callable]], optional
        Additional metrics to compute, by default None

    Returns
    -------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = global_explanations_fit(X, y)
    """
    # Validate inputs
    validate_inputs(X, y, normalizer, metric)

    # Compute statistics
    stats = compute_statistic(X, y, normalizer, distance_metric)

    # Estimate parameters
    params = estimate_parameters(X, y, solver, regularizer)

    # Compute predictions
    y_pred = X @ params["params"]

    # Compute metrics
    metrics = compute_metrics(y, y_pred, metric, custom_metrics)

    return {
        "result": {
            "statistics": stats,
            "parameters": params["params"],
            "penalty": params["penalty"]
        },
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if hasattr(normalizer, '__name__') else "custom",
            "metric": metric.__name__ if hasattr(metric, '__name__') else "custom",
            "distance_metric": distance_metric.__name__ if hasattr(distance_metric, '__name__') else "custom",
            "solver": solver.__name__ if hasattr(solver, '__name__') else "custom",
            "regularizer": regularizer.__name__ if hasattr(regularizer, '__name__') else "custom"
        },
        "warnings": []
    }

################################################################################
# local_explanations
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def local_explanations_fit(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[list] = None,
    target_name: str = "target",
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    distance_metric: str = "euclidean",
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute local explanations for a given dataset.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    feature_names : Optional[list], default=None
        List of feature names.
    target_name : str, default="target"
        Name of the target variable.
    normalizer : Callable[[np.ndarray], np.ndarray], default=lambda x: x
        Function to normalize the features.
    distance_metric : str, default="euclidean"
        Distance metric for local explanations ("euclidean", "manhattan", "cosine", "minkowski").
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]], default="mse"
        Metric to evaluate the explanations ("mse", "mae", "r2", "logloss").
    solver : str, default="closed_form"
        Solver to use for fitting ("closed_form", "gradient_descent", "newton", "coordinate_descent").
    regularization : Optional[str], default=None
        Regularization type ("none", "l1", "l2", "elasticnet").
    alpha : float, default=1.0
        Regularization strength.
    max_iter : int, default=1000
        Maximum number of iterations for iterative solvers.
    tol : float, default=1e-4
        Tolerance for convergence.
    random_state : Optional[int], default=None
        Random seed for reproducibility.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]], default=None
        Custom distance function.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], default=None
        Custom metric function.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features
    X_normalized = normalizer(X)

    # Choose distance metric
    if custom_distance is not None:
        distance_func = custom_distance
    else:
        distance_func = _get_distance_metric(distance_metric)

    # Choose metric
    if custom_metric is not None:
        metric_func = custom_metric
    else:
        metric_func = _get_metric(metric)

    # Choose solver
    if solver == "closed_form":
        coefficients = _closed_form_solver(X_normalized, y)
    elif solver == "gradient_descent":
        coefficients = _gradient_descent_solver(X_normalized, y, alpha, max_iter, tol, random_state)
    elif solver == "newton":
        coefficients = _newton_solver(X_normalized, y, alpha, max_iter, tol, random_state)
    elif solver == "coordinate_descent":
        coefficients = _coordinate_descent_solver(X_normalized, y, alpha, max_iter, tol, random_state)
    else:
        raise ValueError("Invalid solver specified.")

    # Apply regularization if needed
    if regularization == "l1":
        coefficients = _apply_l1_regularization(coefficients, alpha)
    elif regularization == "l2":
        coefficients = _apply_l2_regularization(coefficients, alpha)
    elif regularization == "elasticnet":
        coefficients = _apply_elasticnet_regularization(coefficients, alpha)

    # Compute metrics
    predictions = np.dot(X_normalized, coefficients)
    mse = _compute_mse(y, predictions)
    mae = _compute_mae(y, predictions)
    r2 = _compute_r2(y, predictions)

    metrics = {
        "mse": mse,
        "mae": mae,
        "r2": r2
    }

    # Prepare output
    result = {
        "coefficients": coefficients,
        "feature_names": feature_names if feature_names is not None else [f"feature_{i}" for i in range(X.shape[1])],
        "target_name": target_name
    }

    params_used = {
        "normalizer": normalizer.__name__ if hasattr(normalizer, '__name__') else "custom",
        "distance_metric": distance_metric,
        "metric": metric if isinstance(metric, str) else "custom",
        "solver": solver,
        "regularization": regularization,
        "alpha": alpha,
        "max_iter": max_iter,
        "tol": tol
    }

    warnings = []

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
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

def _get_distance_metric(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the distance metric function."""
    if metric == "euclidean":
        return lambda a, b: np.linalg.norm(a - b)
    elif metric == "manhattan":
        return lambda a, b: np.sum(np.abs(a - b))
    elif metric == "cosine":
        return lambda a, b: 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    elif metric == "minkowski":
        return lambda a, b: np.sum(np.abs(a - b) ** 3) ** (1/3)
    else:
        raise ValueError("Invalid distance metric specified.")

def _get_metric(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the metric function."""
    if metric == "mse":
        return _compute_mse
    elif metric == "mae":
        return _compute_mae
    elif metric == "r2":
        return _compute_r2
    elif metric == "logloss":
        return _compute_logloss
    else:
        raise ValueError("Invalid metric specified.")

def _closed_form_solver(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed form solution for linear regression."""
    return np.linalg.inv(X.T @ X) @ X.T @ y

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float,
    random_state: Optional[int]
) -> np.ndarray:
    """Gradient descent solver for linear regression."""
    if random_state is not None:
        np.random.seed(random_state)
    coefficients = np.random.randn(X.shape[1])
    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ coefficients - y) / len(y)
        new_coefficients = coefficients - alpha * gradient
        if np.linalg.norm(new_coefficients - coefficients) < tol:
            break
        coefficients = new_coefficients
    return coefficients

def _newton_solver(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float,
    random_state: Optional[int]
) -> np.ndarray:
    """Newton's method solver for linear regression."""
    if random_state is not None:
        np.random.seed(random_state)
    coefficients = np.random.randn(X.shape[1])
    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ coefficients - y) / len(y)
        hessian = 2 * X.T @ X / len(y)
        new_coefficients = coefficients - np.linalg.inv(hessian) @ gradient
        if np.linalg.norm(new_coefficients - coefficients) < tol:
            break
        coefficients = new_coefficients
    return coefficients

def _coordinate_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float,
    random_state: Optional[int]
) -> np.ndarray:
    """Coordinate descent solver for linear regression."""
    if random_state is not None:
        np.random.seed(random_state)
    coefficients = np.zeros(X.shape[1])
    for _ in range(max_iter):
        for j in range(X.shape[1]):
            X_j = X[:, j]
            residuals = y - X @ coefficients + coefficients[j] * X_j
            coefficients[j] = np.sum(X_j * residuals) / (np.sum(X_j ** 2) + alpha)
        if np.linalg.norm(coefficients - np.zeros_like(coefficients)) < tol:
            break
    return coefficients

def _apply_l1_regularization(coefficients: np.ndarray, alpha: float) -> np.ndarray:
    """Apply L1 regularization."""
    return np.sign(coefficients) * np.maximum(np.abs(coefficients) - alpha, 0)

def _apply_l2_regularization(coefficients: np.ndarray, alpha: float) -> np.ndarray:
    """Apply L2 regularization."""
    return coefficients / (1 + alpha * len(coefficients))

def _apply_elasticnet_regularization(coefficients: np.ndarray, alpha: float) -> np.ndarray:
    """Apply elastic net regularization."""
    l1_coeffs = _apply_l1_regularization(coefficients, alpha / 2)
    l2_coeffs = _apply_l2_regularization(coefficients, alpha / 2)
    return (l1_coeffs + l2_coeffs) / 2

def _compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def _compute_logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute log loss."""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
