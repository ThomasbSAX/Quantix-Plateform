"""
Quantix – Module classification_probabiliste
Généré automatiquement
Date: 2026-01-06
"""

################################################################################
# modèle_bayésien
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

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
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize data using specified method."""
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

def _compute_priors(y: np.ndarray) -> np.ndarray:
    """Compute class priors."""
    classes, counts = np.unique(y, return_counts=True)
    return counts / y.shape[0]

def _compute_likelihoods(X: np.ndarray, y: np.ndarray,
                         distance_metric: str = 'euclidean',
                         custom_distance: Optional[Callable] = None) -> Dict:
    """Compute likelihoods for each class."""
    classes = np.unique(y)
    likelihoods = {}
    for cls in classes:
        X_cls = X[y == cls]
        if distance_metric == 'custom' and custom_distance is not None:
            distances = np.array([custom_distance(x, X_cls) for x in X])
        else:
            if distance_metric == 'euclidean':
                distances = np.linalg.norm(X[:, np.newaxis] - X_cls, axis=2)
            elif distance_metric == 'manhattan':
                distances = np.sum(np.abs(X[:, np.newaxis] - X_cls), axis=2)
            elif distance_metric == 'cosine':
                distances = 1 - np.dot(X, X_cls.T) / (
                    np.linalg.norm(X, axis=1)[:, np.newaxis] *
                    np.linalg.norm(X_cls.T, axis=0))
            elif distance_metric == 'minkowski':
                distances = np.sum(np.abs(X[:, np.newaxis] - X_cls)**3, axis=2)**(1/3)
            else:
                raise ValueError(f"Unknown distance metric: {distance_metric}")
        likelihoods[cls] = distances
    return likelihoods

def _compute_posterior(X: np.ndarray, y: np.ndarray,
                       priors: np.ndarray,
                       likelihoods: Dict,
                       distance_metric: str = 'euclidean') -> np.ndarray:
    """Compute posterior probabilities."""
    classes = np.unique(y)
    posteriors = np.zeros((X.shape[0], len(classes)))
    for i, cls in enumerate(classes):
        if distance_metric == 'custom':
            # For custom distances, we need to handle the likelihood computation differently
            # This is a simplified approach and might need adjustment based on actual custom distance
            posteriors[:, i] = priors[i] * np.exp(-likelihoods[cls])
        else:
            posteriors[:, i] = priors[i] * np.exp(-likelihoods[cls].mean(axis=1))
    # Normalize to get probabilities
    sum_posteriors = posteriors.sum(axis=1, keepdims=True)
    return posteriors / (sum_posteriors + 1e-8)

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     metric_names: list = ['accuracy'],
                     custom_metrics: Optional[Dict[str, Callable]] = None) -> Dict:
    """Compute classification metrics."""
    metrics = {}
    if 'accuracy' in metric_names:
        metrics['accuracy'] = np.mean(y_true == y_pred.argmax(axis=1))
    if 'logloss' in metric_names:
        metrics['logloss'] = -np.mean(y_true * np.log(y_pred + 1e-8))
    if custom_metrics is not None:
        for name, func in custom_metrics.items():
            metrics[name] = func(y_true, y_pred)
    return metrics

def modèle_bayésien_fit(X: np.ndarray,
                        y: np.ndarray,
                        normalization: str = 'standard',
                        distance_metric: str = 'euclidean',
                        custom_distance: Optional[Callable] = None,
                        metrics: list = ['accuracy', 'logloss'],
                        custom_metrics: Optional[Dict[str, Callable]] = None) -> Dict:
    """
    Fit a Bayesian classification model.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    distance_metric : str, optional
        Distance metric for likelihood computation ('euclidean', 'manhattan',
        'cosine', 'minkowski', or 'custom')
    custom_distance : Callable, optional
        Custom distance function if distance_metric='custom'
    metrics : list, optional
        List of metric names to compute ('accuracy', 'logloss')
    custom_metrics : Dict[str, Callable], optional
        Dictionary of custom metric functions

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': predicted probabilities
        - 'metrics': computed metrics
        - 'params_used': parameters used in the computation
        - 'warnings': any warnings generated

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = modèle_bayésien_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm = _normalize_data(X, normalization)

    # Compute priors
    priors = _compute_priors(y)

    # Compute likelihoods
    likelihoods = _compute_likelihoods(X_norm, y,
                                      distance_metric=distance_metric,
                                      custom_distance=custom_distance)

    # Compute posterior probabilities
    y_pred = _compute_posterior(X_norm, y,
                               priors=priors,
                               likelihoods=likelihoods,
                               distance_metric=distance_metric)

    # Compute metrics
    metrics_result = _compute_metrics(y, y_pred,
                                     metric_names=metrics,
                                     custom_metrics=custom_metrics)

    # Prepare output
    result = {
        'result': y_pred,
        'metrics': metrics_result,
        'params_used': {
            'normalization': normalization,
            'distance_metric': distance_metric,
            'metrics': metrics
        },
        'warnings': []
    }

    return result

################################################################################
# naive_bayes
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def naive_bayes_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    prior: Optional[np.ndarray] = None,
    smoothing: float = 1e-9,
    metric: Union[str, Callable] = "logloss",
    distance: str = "euclidean",
    normalize: bool = True,
    normalizer: Optional[Callable] = None
) -> Dict:
    """
    Fit a Naive Bayes classifier.

    Parameters:
    -----------
    X : np.ndarray
        Training data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    prior : Optional[np.ndarray], default=None
        Prior probabilities of the classes. If None, computed from data.
    smoothing : float, default=1e-9
        Smoothing parameter to avoid zero probabilities.
    metric : Union[str, Callable], default="logloss"
        Metric to evaluate the model. Can be "logloss" or a custom callable.
    distance : str, default="euclidean"
        Distance metric for feature computation. Options: "euclidean", "manhattan".
    normalize : bool, default=True
        Whether to normalize features.
    normalizer : Optional[Callable], default=None
        Custom normalizer function. If None, uses standard scaling.

    Returns:
    --------
    Dict
        Dictionary containing fitted model parameters and metrics.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Compute class priors
    classes = np.unique(y)
    n_classes = len(classes)

    if prior is None:
        prior = np.bincount(y) / len(y)
    else:
        if len(prior) != n_classes:
            raise ValueError("Prior length must match number of classes.")
        prior = np.asarray(prior)
        prior /= prior.sum()

    # Compute class conditional probabilities
    likelihoods = _compute_likelihoods(X, y, classes, smoothing=smoothing,
                                      distance=distance, normalize=normalize,
                                      normalizer=normalizer)

    # Compute metrics
    metrics = _compute_metrics(likelihoods, prior, metric=metric)

    # Prepare output
    result = {
        "result": {
            "classes": classes,
            "prior": prior,
            "likelihoods": likelihoods
        },
        "metrics": metrics,
        "params_used": {
            "smoothing": smoothing,
            "metric": metric,
            "distance": distance,
            "normalize": normalize
        },
        "warnings": []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values.")

def _compute_likelihoods(
    X: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    *,
    smoothing: float = 1e-9,
    distance: str = "euclidean",
    normalize: bool = True,
    normalizer: Optional[Callable] = None
) -> Dict:
    """Compute class conditional probabilities."""
    likelihoods = {}
    n_features = X.shape[1]

    for cls in classes:
        X_cls = X[y == cls]
        if normalize and normalizer is None:
            X_cls = (X_cls - np.mean(X_cls, axis=0)) / (np.std(X_cls, axis=0) + 1e-8)
        elif normalize and normalizer is not None:
            X_cls = normalizer(X_cls)

        # Compute feature statistics
        means = np.mean(X_cls, axis=0)
        variances = np.var(X_cls, axis=0)

        # Store likelihood parameters
        likelihoods[cls] = {
            "mean": means,
            "variance": variances + smoothing
        }

    return likelihoods

def _compute_metrics(
    likelihoods: Dict,
    prior: np.ndarray,
    metric: Union[str, Callable] = "logloss"
) -> Dict:
    """Compute evaluation metrics."""
    metrics = {}

    if metric == "logloss":
        # Compute log loss (cross-entropy)
        metrics["logloss"] = _compute_log_loss(likelihoods, prior)
    elif callable(metric):
        metrics["custom_metric"] = metric(likelihoods, prior)
    else:
        raise ValueError("Invalid metric specified.")

    return metrics

def _compute_log_loss(likelihoods: Dict, prior: np.ndarray) -> float:
    """Compute log loss (cross-entropy)."""
    # This is a simplified version - actual implementation would need test data
    log_loss = 0.0
    for cls, params in likelihoods.items():
        # Compute class probability (simplified)
        log_prior = np.log(prior[cls])
        # Compute likelihood terms (simplified)
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * params["variance"]))
        log_loss += log_prior + log_likelihood

    return -log_loss  # Return negative for minimization purposes

# Example usage:
"""
X_train = np.random.rand(100, 5)
y_train = np.random.randint(0, 2, size=100)

result = naive_bayes_fit(
    X_train,
    y_train,
    prior=None,
    smoothing=1e-9,
    metric="logloss",
    distance="euclidean",
    normalize=True
)
"""

################################################################################
# régression_logistique
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

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

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize data using specified method."""
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

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute sigmoid function."""
    return 1 / (1 + np.exp(-z))

def _compute_loss(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """Compute specified loss metric."""
    if metric == "logloss":
        return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
    elif metric == "mse":
        return np.mean((y_true - y_pred) ** 2)
    elif metric == "mae":
        return np.mean(np.abs(y_true - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _gradient_descent(X: np.ndarray, y: np.ndarray,
                      learning_rate: float = 0.01,
                      max_iter: int = 1000,
                      tol: float = 1e-4) -> np.ndarray:
    """Perform gradient descent optimization."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(max_iter):
        linear_model = np.dot(X, weights) + bias
        y_pred = _sigmoid(linear_model)

        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)

        weights -= learning_rate * dw
        bias -= learning_rate * db

        if np.linalg.norm(dw) < tol and np.abs(db) < tol:
            break

    return np.concatenate([weights, [bias]])

def _newton_method(X: np.ndarray, y: np.ndarray,
                   max_iter: int = 100,
                   tol: float = 1e-4) -> np.ndarray:
    """Perform Newton's method optimization."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features + 1)

    for _ in range(max_iter):
        linear_model = np.dot(X, weights[:-1]) + weights[-1]
        y_pred = _sigmoid(linear_model)

        hessian = (1 / n_samples) * np.dot(X.T, X * y_pred * (1 - y_pred))
        gradient = (1 / n_samples) * np.dot(X.T, (y_pred - y))

        hessian[-1, :] = np.dot(X.T, y_pred * (1 - y_pred))
        hessian[:, -1] = np.dot(X.T, y_pred * (1 - y_pred))
        hessian[-1, -1] = np.sum(y_pred * (1 - y_pred))

        weights -= np.linalg.solve(hessian, gradient)

        if np.linalg.norm(gradient) < tol:
            break

    return weights

def régression_logistique_fit(X: np.ndarray, y: np.ndarray,
                             normalisation: str = "standard",
                             solveur: str = "gradient_descent",
                             métrique: str = "logloss",
                             learning_rate: float = 0.01,
                             max_iter: int = 1000,
                             tol: float = 1e-4) -> Dict:
    """Fit logistic regression model with specified parameters.

    Args:
        X: Input features (n_samples, n_features)
        y: Target values (n_samples,)
        normalisation: Normalization method ("none", "standard", "minmax", "robust")
        solveur: Optimization algorithm ("gradient_descent", "newton")
        métrique: Loss metric to compute
        learning_rate: Learning rate for gradient descent
        max_iter: Maximum number of iterations
        tol: Tolerance for stopping criteria

    Returns:
        Dictionary containing results, metrics, parameters used and warnings
    """
    _validate_inputs(X, y)

    # Add bias term to X
    X = np.hstack([X, np.ones((X.shape[0], 1))])

    # Normalize data
    X_normalized = _normalize_data(X, normalisation)

    # Choose solver
    if solveur == "gradient_descent":
        weights = _gradient_descent(X_normalized, y,
                                   learning_rate=learning_rate,
                                   max_iter=max_iter,
                                   tol=tol)
    elif solveur == "newton":
        weights = _newton_method(X_normalized, y,
                                max_iter=max_iter,
                                tol=tol)
    else:
        raise ValueError(f"Unknown solver: {solveur}")

    # Compute predictions
    y_pred = _sigmoid(np.dot(X_normalized, weights))

    # Compute metrics
    metrics = {
        "logloss": _compute_loss(y, y_pred, "logloss"),
        "mse": _compute_loss(y, y_pred, "mse"),
        "mae": _compute_loss(y, y_pred, "mae")
    }

    return {
        "result": {
            "weights": weights,
            "predictions": y_pred
        },
        "metrics": {métrique: metrics[métrique]},
        "params_used": {
            "normalisation": normalisation,
            "solveur": solveur,
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "tol": tol
        },
        "warnings": []
    }

# Exemple d'utilisation:
"""
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 1])
result = régression_logistique_fit(X, y)
"""

################################################################################
# arbre_de_decision
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
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

def compute_criterion(
    X: np.ndarray,
    y: np.ndarray,
    criterion: str = "gini",
    custom_criterion: Optional[Callable] = None
) -> np.ndarray:
    """Compute the splitting criterion for decision tree nodes."""
    if custom_criterion is not None:
        return custom_criterion(X, y)

    if criterion == "gini":
        unique_classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini
    elif criterion == "entropy":
        unique_classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

def find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: Optional[np.ndarray] = None,
    max_depth: int = 5,
    min_samples_split: int = 2
) -> Dict[str, Any]:
    """Find the best split for a node in the decision tree."""
    if feature_indices is None:
        feature_indices = np.arange(X.shape[1])

    best_split = {
        "feature": None,
        "threshold": None,
        "left_indices": None,
        "right_indices": None,
        "improvement": -np.inf
    }

    for feature in feature_indices:
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_indices = X[:, feature] <= threshold
            right_indices = ~left_indices

            if np.sum(left_indices) < min_samples_split or np.sum(right_indices) < min_samples_split:
                continue

            parent_criterion = compute_criterion(X, y)
            left_criterion = compute_criterion(X[left_indices], y[left_indices])
            right_criterion = compute_criterion(X[right_indices], y[right_indices])

            improvement = parent_criterion - (left_criterion * np.sum(left_indices) + right_criterion * np.sum(right_indices)) / len(y)

            if improvement > best_split["improvement"]:
                best_split.update({
                    "feature": feature,
                    "threshold": threshold,
                    "left_indices": left_indices,
                    "right_indices": right_indices,
                    "improvement": improvement
                })

    return best_split

def build_tree(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int = 5,
    min_samples_split: int = 2,
    criterion: str = "gini",
    custom_criterion: Optional[Callable] = None,
    feature_indices: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Recursively build the decision tree."""
    if feature_indices is None:
        feature_indices = np.arange(X.shape[1])

    node = {
        "is_leaf": False,
        "feature": None,
        "threshold": None,
        "left": None,
        "right": None,
        "class_distribution": np.bincount(y.astype(int), minlength=np.max(y) + 1)
    }

    if len(np.unique(y)) == 1 or len(feature_indices) == 0 or max_depth <= 0:
        node["is_leaf"] = True
        return node

    best_split = find_best_split(X, y, feature_indices, max_depth, min_samples_split)

    if best_split["improvement"] <= 0:
        node["is_leaf"] = True
        return node

    node.update({
        "feature": best_split["feature"],
        "threshold": best_split["threshold"]
    })

    left_feature_indices = np.delete(feature_indices, node["feature"])
    right_feature_indices = np.delete(feature_indices, node["feature"])

    node["left"] = build_tree(
        X[best_split["left_indices"]],
        y[best_split["left_indices"]],
        max_depth - 1,
        min_samples_split,
        criterion,
        custom_criterion,
        left_feature_indices
    )

    node["right"] = build_tree(
        X[best_split["right_indices"]],
        y[best_split["right_indices"]],
        max_depth - 1,
        min_samples_split,
        criterion,
        custom_criterion,
        right_feature_indices
    )

    return node

def predict_sample(node: Dict[str, Any], x: np.ndarray) -> int:
    """Predict the class for a single sample."""
    if node["is_leaf"]:
        return np.argmax(node["class_distribution"])

    if x[node["feature"]] <= node["threshold"]:
        return predict_sample(node["left"], x)
    else:
        return predict_sample(node["right"], x)

def arbre_de_decision_fit(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int = 5,
    min_samples_split: int = 2,
    criterion: str = "gini",
    custom_criterion: Optional[Callable] = None,
    normalizer: Callable = lambda x: x
) -> Dict[str, Any]:
    """
    Fit a decision tree classifier.

    Parameters:
    - X: Input features (2D array)
    - y: Target labels (1D array)
    - max_depth: Maximum depth of the tree
    - min_samples_split: Minimum number of samples required to split a node
    - criterion: Splitting criterion ("gini" or "entropy")
    - custom_criterion: Custom splitting criterion function
    - normalizer: Function to normalize input features

    Returns:
    - Dictionary containing the fitted tree and other information
    """
    validate_inputs(X, y)

    X_normalized = normalizer(X)
    tree = build_tree(
        X_normalized,
        y,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        custom_criterion=custom_criterion
    )

    return {
        "result": tree,
        "metrics": {},
        "params_used": {
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "criterion": criterion,
            "normalizer": normalizer.__name__ if hasattr(normalizer, "__name__") else "custom"
        },
        "warnings": []
    }

def arbre_de_decision_predict(
    tree: Dict[str, Any],
    X: np.ndarray,
    normalizer: Callable = lambda x: x
) -> np.ndarray:
    """
    Predict class labels for samples using a fitted decision tree.

    Parameters:
    - tree: Fitted decision tree
    - X: Input features (2D array)
    - normalizer: Function to normalize input features

    Returns:
    - Predicted class labels (1D array)
    """
    X_normalized = normalizer(X)
    return np.array([predict_sample(tree, x) for x in X_normalized])

################################################################################
# forêt_aleatoire
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

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize data using specified method."""
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

def _calculate_metric(y_true: np.ndarray, y_pred: np.ndarray,
                      metric: Union[str, Callable]) -> float:
    """Calculate specified metric between true and predicted values."""
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

def _calculate_distance(x1: np.ndarray, x2: np.ndarray,
                        distance: str) -> float:
    """Calculate specified distance between two vectors."""
    if distance == "euclidean":
        return np.linalg.norm(x1 - x2)
    elif distance == "manhattan":
        return np.sum(np.abs(x1 - x2))
    elif distance == "cosine":
        return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    elif distance == "minkowski":
        return np.sum(np.abs(x1 - x2) ** 3) ** (1/3)
    else:
        raise ValueError(f"Unknown distance: {distance}")

def _bootstrap_sample(X: np.ndarray, y: np.ndarray,
                      sample_size: int) -> tuple:
    """Create a bootstrap sample from the data."""
    indices = np.random.choice(X.shape[0], size=sample_size, replace=True)
    return X[indices], y[indices]

def _grow_tree(X: np.ndarray, y: np.ndarray,
               max_depth: int, min_samples_split: int,
               distance: str) -> Dict:
    """Grow a single decision tree."""
    def _split_node(X_node: np.ndarray, y_node: np.ndarray,
                    depth: int) -> Dict:
        if (depth >= max_depth or len(y_node) < min_samples_split):
            return {"is_leaf": True, "value": np.mean(y_node)}

        best_split = None
        best_gain = -np.inf

        for i in range(X_node.shape[1]):
            thresholds = np.unique(X_node[:, i])
            for threshold in thresholds:
                left_mask = X_node[:, i] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < min_samples_split or np.sum(right_mask) < min_samples_split:
                    continue

                left_var = np.var(y_node[left_mask])
                right_var = np.var(y_node[right_mask])
                gain = left_var + right_var

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        "feature": i,
                        "threshold": threshold,
                        "left_mask": left_mask,
                        "right_mask": right_mask
                    }

        if best_split is None:
            return {"is_leaf": True, "value": np.mean(y_node)}

        left_child = _split_node(X_node[best_split["left_mask"]],
                                y_node[best_split["left_mask"]], depth + 1)
        right_child = _split_node(X_node[best_split["right_mask"]],
                                 y_node[best_split["right_mask"]], depth + 1)

        return {
            "is_leaf": False,
            "feature": best_split["feature"],
            "threshold": best_split["threshold"],
            "left_child": left_child,
            "right_child": right_child
        }

    return _split_node(X, y, 0)

def forêt_aleatoire_fit(X: np.ndarray, y: np.ndarray,
                        n_estimators: int = 100,
                        max_depth: int = None,
                        min_samples_split: int = 2,
                        normalization: str = "none",
                        distance: str = "euclidean") -> Dict[str, Any]:
    """
    Fit a random forest classifier.

    Parameters:
    -----------
    X : np.ndarray
        Training data of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    n_estimators : int, optional
        Number of trees in the forest (default: 100)
    max_depth : int, optional
        Maximum depth of the trees (default: None)
    min_samples_split : int, optional
        Minimum number of samples required to split a node (default: 2)
    normalization : str, optional
        Normalization method ("none", "standard", "minmax", "robust") (default: "none")
    distance : str, optional
        Distance metric ("euclidean", "manhattan", "cosine", "minkowski") (default: "euclidean")

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the fitted model and related information
    """
    _validate_inputs(X, y)
    X_normalized = _normalize_data(X, normalization)

    forest = []
    for _ in range(n_estimators):
        X_sample, y_sample = _bootstrap_sample(X_normalized, y,
                                              sample_size=X.shape[0])
        tree = _grow_tree(X_sample, y_sample,
                         max_depth if max_depth is not None else np.inf,
                         min_samples_split, distance)
        forest.append(tree)

    return {
        "result": {"forest": forest},
        "metrics": {},
        "params_used": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "normalization": normalization,
            "distance": distance
        },
        "warnings": []
    }

def forêt_aleatoire_compute(model: Dict[str, Any],
                            X: np.ndarray,
                            metric: Union[str, Callable] = "mse") -> Dict[str, Any]:
    """
    Compute predictions and metrics using a fitted random forest model.

    Parameters:
    -----------
    model : Dict[str, Any]
        Fitted random forest model
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    metric : Union[str, Callable], optional
        Metric to compute ("mse", "mae", "r2", "logloss" or custom callable) (default: "mse")

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing predictions and computed metrics
    """
    _validate_inputs(X, np.zeros(X.shape[0]))

    predictions = []
    for tree in model["result"]["forest"]:
        def _predict_tree(x: np.ndarray, node: Dict) -> float:
            if node["is_leaf"]:
                return node["value"]
            if x[node["feature"]] <= node["threshold"]:
                return _predict_tree(x, node["left_child"])
            else:
                return _predict_tree(x, node["right_child"])

        tree_preds = np.array([_predict_tree(x, tree) for x in X])
        predictions.append(tree_preds)

    avg_predictions = np.mean(predictions, axis=0)
    computed_metric = _calculate_metric(np.zeros(X.shape[0]), avg_predictions, metric)

    return {
        "result": {"predictions": avg_predictions},
        "metrics": {metric: computed_metric if isinstance(metric, str) else "custom"},
        "params_used": {},
        "warnings": []
    }

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
    max_depth: int = 3,
    min_samples_split: int = 2,
    loss: str = 'deviance',
    criterion: str = 'friedman_mse',
    subsample: float = 1.0,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Fit a gradient boosting model for probabilistic classification.

    Parameters
    ----------
    X : np.ndarray
        Training data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    n_estimators : int, optional
        Number of boosting stages to be run. Default is 100.
    learning_rate : float, optional
        Learning rate shrinks the contribution of each tree. Default is 0.1.
    max_depth : int, optional
        Maximum depth of the individual regression estimators. Default is 3.
    min_samples_split : int, optional
        Minimum number of samples required to split an internal node. Default is 2.
    loss : str, optional
        Loss function to be optimized. 'deviance' for classification. Default is 'deviance'.
    criterion : str, optional
        Function to measure the quality of a split. 'friedman_mse' for MSE. Default is 'friedman_mse'.
    subsample : float, optional
        The fraction of samples to be used for fitting the individual base learners. Default is 1.0.
    random_state : int, optional
        Controls both the randomness of the bootstrapping of the samples used when building trees and the sampling of the features to consider when looking for the best split at each node. Default is None.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict]]
        A dictionary containing the fitted model and metrics.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Initialize random state
    rng = np.random.RandomState(random_state)

    # Initialize predictions with the mean of y for regression or log-odds for classification
    if loss == 'deviance':
        initial_pred = np.log(np.mean(y) / (1 - np.mean(y)))
    else:
        initial_pred = np.mean(y)

    # Initialize model components
    estimators = []
    train_errors = []

    for i in range(n_estimators):
        # Compute pseudo-residuals
        residuals = _compute_residuals(y, initial_pred, loss)

        # Fit a weak learner (decision tree) to the residuals
        estimator = _fit_weak_learner(
            X, residuals,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
            subsample=subsample,
            random_state=rng
        )

        # Update predictions
        initial_pred += learning_rate * estimator.predict(X)

        # Compute training error
        train_error = _compute_metric(y, initial_pred, loss)
        train_errors.append(train_error)

        # Store the estimator
        estimators.append(estimator)

    return {
        'result': {'estimators': estimators, 'initial_prediction': initial_pred},
        'metrics': {'train_errors': train_errors},
        'params_used': {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'loss': loss,
            'criterion': criterion,
            'subsample': subsample,
            'random_state': random_state
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
        raise ValueError("X and y must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("X and y must not contain infinite values.")

def _compute_residuals(y: np.ndarray, pred: np.ndarray, loss: str) -> np.ndarray:
    """Compute the residuals based on the specified loss function."""
    if loss == 'deviance':
        # For logistic regression, residuals are the negative gradient of the loss
        return y - _sigmoid(pred)
    else:
        # For other losses, residuals are the difference between y and pred
        return y - pred

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-z))

def _fit_weak_learner(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int = 3,
    min_samples_split: int = 2,
    criterion: str = 'friedman_mse',
    subsample: float = 1.0,
    random_state: Optional[np.random.RandomState] = None
) -> 'DecisionTreeRegressor':
    """Fit a weak learner (decision tree) to the data."""
    from sklearn.tree import DecisionTreeRegressor

    n_samples = X.shape[0]
    if subsample < 1.0:
        n_samples = int(n_samples * subsample)

    if random_state is not None:
        indices = random_state.choice(n_samples, size=n_samples, replace=False)
    else:
        indices = np.random.choice(n_samples, size=n_samples, replace=False)

    X_subsampled = X[indices]
    y_subsampled = y[indices]

    estimator = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        random_state=random_state
    )
    estimator.fit(X_subsampled, y_subsampled)

    return estimator

def _compute_metric(y: np.ndarray, pred: np.ndarray, loss: str) -> float:
    """Compute the metric based on the specified loss function."""
    if loss == 'deviance':
        # Log loss for classification
        return -np.mean(y * np.log(pred + 1e-15) + (1 - y) * np.log(1 - pred + 1e-15))
    else:
        # Mean squared error for regression
        return np.mean((y - pred) ** 2)

################################################################################
# k_plus_proches_voisins
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def k_plus_proches_voisins_fit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int = 5,
    distance_metric: Union[str, Callable] = 'euclidean',
    normalization: Optional[str] = None,
    weight_function: Union[str, Callable] = 'uniform',
    custom_distance: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Fit the k-nearest neighbors model for probabilistic classification.

    Parameters:
    -----------
    X_train : np.ndarray
        Training data of shape (n_samples, n_features).
    y_train : np.ndarray
        Target values of shape (n_samples,).
    k : int, optional
        Number of neighbors to consider. Default is 5.
    distance_metric : str or callable, optional
        Distance metric to use. Can be 'euclidean', 'manhattan', 'cosine',
        or a custom callable. Default is 'euclidean'.
    normalization : str, optional
        Normalization method. Can be 'standard', 'minmax', or None.
    weight_function : str or callable, optional
        Weight function for neighbors. Can be 'uniform', 'distance',
        or a custom callable. Default is 'uniform'.
    custom_distance : callable, optional
        Custom distance function if not using built-in metrics.
    **kwargs :
        Additional keyword arguments for specific distance metrics.

    Returns:
    --------
    Dict
        Dictionary containing the fitted model and metadata.
    """
    # Validate inputs
    _validate_inputs(X_train, y_train)

    # Normalize data if specified
    X_normalized = _apply_normalization(X_train, normalization)

    # Prepare the model dictionary
    model = {
        'X_train': X_normalized,
        'y_train': y_train,
        'k': k,
        'distance_metric': distance_metric,
        'normalization': normalization,
        'weight_function': weight_function
    }

    return {
        'result': model,
        'metrics': {},
        'params_used': kwargs,
        'warnings': []
    }

def k_plus_proches_voisins_compute(
    X_test: np.ndarray,
    model: Dict,
    distance_metric: Union[str, Callable] = 'euclidean',
    weight_function: Union[str, Callable] = 'uniform'
) -> Dict:
    """
    Compute predictions using the fitted k-nearest neighbors model.

    Parameters:
    -----------
    X_test : np.ndarray
        Test data of shape (n_samples, n_features).
    model : Dict
        Fitted model dictionary from k_plus_proches_voisins_fit.
    distance_metric : str or callable, optional
        Distance metric to use. Default is 'euclidean'.
    weight_function : str or callable, optional
        Weight function for neighbors. Default is 'uniform'.

    Returns:
    --------
    Dict
        Dictionary containing predictions and metadata.
    """
    # Validate inputs
    _validate_inputs(X_test, model['y_train'])

    # Normalize test data using the same method as training
    X_test_normalized = _apply_normalization(X_test, model['normalization'])

    # Compute distances
    distances = _compute_distances(
        X_test_normalized,
        model['X_train'],
        distance_metric
    )

    # Find k nearest neighbors
    knn_indices = _find_knn(distances, model['k'])

    # Get predictions
    predictions = _get_predictions(
        knn_indices,
        model['y_train'],
        weight_function
    )

    return {
        'result': predictions,
        'metrics': {},
        'params_used': {'distance_metric': distance_metric, 'weight_function': weight_function},
        'warnings': []
    }

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

def _apply_normalization(X: np.ndarray, method: Optional[str]) -> np.ndarray:
    """Apply specified normalization to the data."""
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

def _compute_distances(
    X_test: np.ndarray,
    X_train: np.ndarray,
    metric: Union[str, Callable]
) -> np.ndarray:
    """Compute distances between test and training data."""
    if callable(metric):
        return metric(X_test, X_train)
    elif metric == 'euclidean':
        return np.sqrt(np.sum((X_test[:, np.newaxis, :] - X_train) ** 2, axis=2))
    elif metric == 'manhattan':
        return np.sum(np.abs(X_test[:, np.newaxis, :] - X_train), axis=2)
    elif metric == 'cosine':
        dot_products = np.dot(X_test, X_train.T)
        norms = np.linalg.norm(X_test, axis=1)[:, np.newaxis] * np.linalg.norm(X_train, axis=1)
        return 1 - dot_products / (norms + 1e-8)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _find_knn(distances: np.ndarray, k: int) -> np.ndarray:
    """Find indices of k nearest neighbors."""
    return np.argpartition(distances, k, axis=1)[:, :k]

def _get_predictions(
    knn_indices: np.ndarray,
    y_train: np.ndarray,
    weight_function: Union[str, Callable]
) -> np.ndarray:
    """Get predictions based on k nearest neighbors."""
    if callable(weight_function):
        weights = weight_function(knn_indices)
    elif weight_function == 'uniform':
        weights = np.ones_like(knn_indices, dtype=float)
    elif weight_function == 'distance':
        # Assuming distances are precomputed and available
        raise NotImplementedError("Distance weighting requires precomputed distances.")
    else:
        raise ValueError(f"Unknown weight function: {weight_function}")

    # For probabilistic classification, we can return class probabilities
    unique_classes = np.unique(y_train)
    n_classes = len(unique_classes)
    class_probs = np.zeros((knn_indices.shape[0], n_classes))

    for i in range(knn_indices.shape[0]):
        neighbors = y_train[knn_indices[i]]
        for j, cls in enumerate(unique_classes):
            class_mask = (neighbors == cls)
            class_probs[i, j] = np.sum(weights[i][class_mask]) / np.sum(weights[i])

    return class_probs

################################################################################
# réseau_de_neurones
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def réseau_de_neurones_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalisation: str = 'standard',
    métrique: Union[str, Callable] = 'logloss',
    solveur: str = 'gradient_descent',
    régularisation: Optional[str] = None,
    tolérance: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    batch_size: Optional[int] = None,
    activation: str = 'sigmoid',
    layers: tuple = (10,),
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Fit a neural network model for probabilistic classification.

    Parameters:
    -----------
    X : np.ndarray
        Training data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,) or (n_samples, n_classes).
    normalisation : str
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    métrique : str or callable
        Metric to evaluate the model: 'mse', 'mae', 'r2', 'logloss', or custom.
    solveur : str
        Solver method: 'gradient_descent', 'newton', or 'coordinate_descent'.
    régularisation : str, optional
        Regularization type: 'none', 'l1', 'l2', or 'elasticnet'.
    tolérance : float
        Tolerance for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    learning_rate : float
        Learning rate for gradient descent.
    batch_size : int, optional
        Batch size for mini-batch gradient descent.
    activation : str
        Activation function: 'sigmoid', 'relu', or 'tanh'.
    layers : tuple
        Architecture of the neural network (number of neurons per layer).
    random_state : int, optional
        Random seed for reproducibility.
    custom_metric : callable, optional
        Custom metric function.

    Returns:
    --------
    dict
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validation des entrées
    _validate_inputs(X, y)

    # Normalisation des données
    X_normalized = _normaliser(X, normalisation)

    # Initialisation des paramètres
    params = _initialiser_paramètres(layers, X.shape[1], random_state)

    # Choix du solveur
    if solveur == 'gradient_descent':
        params = _gradient_descent(
            X_normalized, y, params, métrique, régularisation,
            tolérance, max_iter, learning_rate, batch_size
        )
    elif solveur == 'newton':
        params = _newton_method(
            X_normalized, y, params, métrique, régularisation,
            tolérance, max_iter
        )
    elif solveur == 'coordinate_descent':
        params = _coordinate_descent(
            X_normalized, y, params, métrique, régularisation,
            tolérance, max_iter
        )

    # Calcul des métriques
    metrics = _calculer_métriques(X_normalized, y, params, métrique, custom_metric)

    # Retour des résultats
    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalisation': normalisation,
            'métrique': métrique,
            'solveur': solveur,
            'régularisation': régularisation,
            'tolérance': tolérance,
            'max_iter': max_iter,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'activation': activation,
            'layers': layers
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim not in (1, 2) or (y.ndim == 2 and y.shape[1] != 1):
        raise ValueError("y must be a 1D or 2D array with one column.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")

def _normaliser(X: np.ndarray, méthode: str) -> np.ndarray:
    """Normalize the input data."""
    if méthode == 'none':
        return X
    elif méthode == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif méthode == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif méthode == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {méthode}")

def _initialiser_paramètres(layers: tuple, n_features: int, random_state: Optional[int] = None) -> Dict:
    """Initialize the parameters of the neural network."""
    if random_state is not None:
        np.random.seed(random_state)
    params = {}
    prev_layer_size = n_features
    for i, layer_size in enumerate(layers):
        params[f'W{i}'] = np.random.randn(prev_layer_size, layer_size) * 0.01
        params[f'b{i}'] = np.zeros((1, layer_size))
        prev_layer_size = layer_size
    return params

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict,
    métrique: Union[str, Callable],
    régularisation: Optional[str],
    tolérance: float,
    max_iter: int,
    learning_rate: float,
    batch_size: Optional[int]
) -> Dict:
    """Perform gradient descent optimization."""
    n_samples = X.shape[0]
    if batch_size is None:
        batch_size = n_samples
    for _ in range(max_iter):
        # Mini-batch gradient descent
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]
            grads = _calculer_gradients(X_batch, y_batch, params, métrique, régularisation)
            _mettre_a_jour_paramètres(params, grads, learning_rate)
    return params

def _calculer_gradients(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict,
    métrique: Union[str, Callable],
    régularisation: Optional[str]
) -> Dict:
    """Calculate the gradients of the loss function."""
    grads = {}
    # Forward pass
    A = X
    for i in range(len(params) // 2):
        W = params[f'W{i}']
        b = params[f'b{i}']
        Z = np.dot(A, W) + b
        A = _activation_function(Z, 'sigmoid')
    # Backward pass (simplified for example)
    dA = A - y
    for i in reversed(range(len(params) // 2)):
        W = params[f'W{i}']
        dZ = dA * _activation_derivative(A, 'sigmoid')
        grads[f'dW{i}'] = np.dot(X.T, dZ) / X.shape[0]
        grads[f'db{i}'] = np.sum(dZ, axis=0) / X.shape[0]
        dA = np.dot(dZ, W.T)
    return grads

def _mettre_a_jour_paramètres(params: Dict, grads: Dict, learning_rate: float) -> None:
    """Update the parameters using gradients."""
    for key in params:
        params[key] -= learning_rate * grads[f'd{key}']

def _calculer_métriques(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict,
    métrique: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Calculate the metrics for the model."""
    metrics = {}
    A = X
    for i in range(len(params) // 2):
        W = params[f'W{i}']
        b = params[f'b{i}']
        Z = np.dot(A, W) + b
        A = _activation_function(Z, 'sigmoid')
    if métrique == 'logloss':
        metrics['logloss'] = -np.mean(y * np.log(A + 1e-8) + (1 - y) * np.log(1 - A + 1e-8))
    elif métrique == 'mse':
        metrics['mse'] = np.mean((y - A) ** 2)
    elif métrique == 'mae':
        metrics['mae'] = np.mean(np.abs(y - A))
    elif métrique == 'r2':
        ss_res = np.sum((y - A) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics['r2'] = 1 - ss_res / (ss_tot + 1e-8)
    if custom_metric is not None:
        metrics['custom'] = custom_metric(y, A)
    return metrics

def _activation_function(Z: np.ndarray, activation: str) -> np.ndarray:
    """Apply the activation function."""
    if activation == 'sigmoid':
        return 1 / (1 + np.exp(-Z))
    elif activation == 'relu':
        return np.maximum(0, Z)
    elif activation == 'tanh':
        return np.tanh(Z)
    else:
        raise ValueError(f"Unknown activation function: {activation}")

def _activation_derivative(A: np.ndarray, activation: str) -> np.ndarray:
    """Calculate the derivative of the activation function."""
    if activation == 'sigmoid':
        return A * (1 - A)
    elif activation == 'relu':
        return (A > 0).astype(float)
    elif activation == 'tanh':
        return 1 - A ** 2
    else:
        raise ValueError(f"Unknown activation function: {activation}")

################################################################################
# svm_lineaire
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

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

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize data using specified method."""
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

def _compute_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray,
                      C: float, penalty: str) -> np.ndarray:
    """Compute gradient for linear SVM."""
    n_samples = X.shape[0]
    yXw = y * (X @ w)
    hinge_loss_grad = -y * X * (yXw >= 1).astype(float)
    penalty_grad = np.zeros_like(w)

    if penalty == "l2":
        penalty_grad = 2 * w
    elif penalty == "l1":
        penalty_grad = np.sign(w)
    elif penalty == "elasticnet":
        penalty_grad = 2 * w + np.sign(w)

    return (hinge_loss_grad.sum(axis=0) / n_samples) + C * penalty_grad

def _compute_hessian(X: np.ndarray, y: np.ndarray, w: np.ndarray,
                     C: float, penalty: str) -> np.ndarray:
    """Compute Hessian for linear SVM."""
    n_samples = X.shape[0]
    yXw = y * (X @ w)
    hinge_loss_hess = np.zeros((n_samples, X.shape[1], X.shape[1]))

    for i in range(n_samples):
        if yXw[i] >= 1:
            hinge_loss_hess[i] = X[i, :][:, np.newaxis] * X[i, :]

    penalty_hess = np.zeros_like(hinge_loss_hess[0])

    if penalty == "l2":
        penalty_hess = 2 * np.eye(X.shape[1])
    elif penalty == "elasticnet":
        penalty_hess = 2 * np.eye(X.shape[1])

    return (hinge_loss_hess.sum(axis=0) / n_samples) + C * penalty_hess

def _gradient_descent(X: np.ndarray, y: np.ndarray,
                      w_init: np.ndarray, C: float, penalty: str,
                      tol: float = 1e-4, max_iter: int = 1000) -> np.ndarray:
    """Gradient descent solver for linear SVM."""
    w = w_init.copy()
    prev_w = np.zeros_like(w)
    iter_count = 0

    while np.linalg.norm(w - prev_w) > tol and iter_count < max_iter:
        prev_w = w.copy()
        grad = _compute_gradient(X, y, w, C, penalty)
        w -= 0.1 * grad
        iter_count += 1

    return w

def _newton_method(X: np.ndarray, y: np.ndarray,
                   w_init: np.ndarray, C: float, penalty: str,
                   tol: float = 1e-4, max_iter: int = 100) -> np.ndarray:
    """Newton method solver for linear SVM."""
    w = w_init.copy()
    prev_w = np.zeros_like(w)
    iter_count = 0

    while np.linalg.norm(w - prev_w) > tol and iter_count < max_iter:
        prev_w = w.copy()
        grad = _compute_gradient(X, y, w, C, penalty)
        hess = _compute_hessian(X, y, w, C, penalty)
        w -= np.linalg.solve(hess, grad)
        iter_count += 1

    return w

def svm_lineaire_fit(X: np.ndarray, y: np.ndarray,
                     C: float = 1.0, penalty: str = "l2",
                     solver: str = "gradient_descent",
                     normalization: str = "standard",
                     tol: float = 1e-4,
                     max_iter: int = 1000) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Fit a linear SVM model to the data.

    Parameters
    ----------
    X : np.ndarray
        Training data of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    C : float, optional
        Regularization parameter, by default 1.0
    penalty : str, optional
        Penalty type ('none', 'l1', 'l2', 'elasticnet'), by default "l2"
    solver : str, optional
        Solver type ('gradient_descent', 'newton'), by default "gradient_descent"
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust'), by default "standard"
    tol : float, optional
        Tolerance for stopping criteria, by default 1e-4
    max_iter : int, optional
        Maximum number of iterations, by default 1000

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict]]
        Dictionary containing:
        - "result": Fitted weights
        - "metrics": Computed metrics
        - "params_used": Parameters used for fitting
        - "warnings": Any warnings encountered

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100) * 2 - 1
    >>> result = svm_lineaire_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm = _normalize_data(X, normalization)

    # Initialize weights
    w_init = np.zeros(X_norm.shape[1])

    # Choose solver
    if solver == "gradient_descent":
        w = _gradient_descent(X_norm, y, w_init, C, penalty, tol, max_iter)
    elif solver == "newton":
        w = _newton_method(X_norm, y, w_init, C, penalty, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    y_pred = np.sign(X_norm @ w)
    accuracy = np.mean(y_pred == y)

    # Prepare output
    result = {
        "result": w,
        "metrics": {"accuracy": accuracy},
        "params_used": {
            "C": C,
            "penalty": penalty,
            "solver": solver,
            "normalization": normalization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    return result

################################################################################
# svm_rbf
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
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

def normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """Normalize data using specified method."""
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

def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    """Compute RBF kernel between two matrices."""
    sq_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * sq_dists)

def compute_dual_objective(K: np.ndarray, y: np.ndarray, C: float) -> tuple:
    """Compute dual objective function and its gradient."""
    n_samples = K.shape[0]
    alpha = np.zeros(n_samples)

    def objective(alpha: np.ndarray) -> float:
        return 0.5 * np.dot(alpha, np.dot(alpha, K)) - np.sum(alpha)

    def gradient(alpha: np.ndarray) -> np.ndarray:
        return np.dot(alpha, K) - y

    return objective, gradient

def solve_dual_problem(K: np.ndarray, y: np.ndarray, C: float,
                       solver: str = 'gradient_descent',
                       tol: float = 1e-4,
                       max_iter: int = 1000) -> np.ndarray:
    """Solve dual SVM problem using specified solver."""
    objective, grad = compute_dual_objective(K, y, C)
    alpha = np.zeros(K.shape[0])

    if solver == 'gradient_descent':
        for _ in range(max_iter):
            grad_val = grad(alpha)
            alpha -= 0.01 * grad_val
            alpha = np.clip(alpha, 0, C)
            if np.linalg.norm(grad_val) < tol:
                break
    else:
        raise ValueError(f"Unknown solver: {solver}")

    return alpha

def compute_decision_function(X: np.ndarray, X_train: np.ndarray,
                             y_train: np.ndarray, alpha: np.ndarray,
                             gamma: float) -> np.ndarray:
    """Compute decision function for new data points."""
    K = rbf_kernel(X, X_train, gamma)
    return np.dot(alpha * y_train, K)

def compute_probabilities(decision: np.ndarray) -> np.ndarray:
    """Compute probabilities using logistic function."""
    return 1 / (1 + np.exp(-decision))

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metric: Union[str, Callable]) -> Dict[str, float]:
    """Compute specified metrics."""
    if metric == 'accuracy':
        return {'accuracy': np.mean(y_true == y_pred)}
    elif callable(metric):
        return {f'custom_metric': metric(y_true, y_pred)}
    else:
        raise ValueError(f"Unknown metric: {metric}")

def svm_rbf_fit(X: np.ndarray, y: np.ndarray,
                gamma: float = 1.0,
                C: float = 1.0,
                normalization: str = 'standard',
                solver: str = 'gradient_descent',
                metric: Union[str, Callable] = 'accuracy',
                tol: float = 1e-4,
                max_iter: int = 1000) -> Dict:
    """Fit SVM with RBF kernel and return results."""
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_norm = normalize_data(X, normalization)

    # Compute kernel matrix
    K = rbf_kernel(X_norm, X_norm, gamma)

    # Solve dual problem
    alpha = solve_dual_problem(K, y, C, solver, tol, max_iter)

    # Compute decision function
    decision = compute_decision_function(X_norm, X_norm, y, alpha, gamma)

    # Compute probabilities
    probas = compute_probabilities(decision)

    # Compute metrics
    y_pred = (probas > 0.5).astype(int)
    metrics = compute_metrics(y, y_pred, metric)

    return {
        'result': probas,
        'metrics': metrics,
        'params_used': {
            'gamma': gamma,
            'C': C,
            'normalization': normalization,
            'solver': solver,
            'metric': metric.__name__ if callable(metric) else metric
        },
        'warnings': []
    }

# Example usage:
"""
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, size=100)

result = svm_rbf_fit(X, y,
                    gamma=0.5,
                    C=1.0,
                    normalization='standard',
                    solver='gradient_descent',
                    metric='accuracy')
"""

################################################################################
# k_means
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_inputs(X: np.ndarray, n_clusters: int = 8,
                   random_state: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Validate input data and parameters for k-means clustering.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_clusters : int, optional
        Number of clusters to form (default is 8)
    random_state : int or None, optional
        Seed for random number generation (default is None)

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing validated inputs and parameters

    Raises
    ------
    ValueError
        If input validation fails
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X must be a 2D numpy array")
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    if random_state is not None and (not isinstance(random_state, int) or random_state < 0):
        raise ValueError("random_state must be non-negative integer or None")

    return {
        'X': X,
        'n_clusters': n_clusters,
        'random_state': random_state
    }

def initialize_centroids(X: np.ndarray, n_clusters: int,
                        init_method: str = 'random',
                        random_state: Optional[int] = None) -> np.ndarray:
    """
    Initialize cluster centroids using specified method.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_clusters : int
        Number of clusters to form
    init_method : str, optional
        Initialization method ('random' or 'k-means++') (default is 'random')
    random_state : int or None, optional
        Seed for random number generation (default is None)

    Returns
    -------
    np.ndarray
        Initial centroids of shape (n_clusters, n_features)
    """
    rng = np.random.RandomState(random_state)

    if init_method == 'random':
        indices = rng.permutation(X.shape[0])[:n_clusters]
        return X[indices]

    elif init_method == 'k-means++':
        centroids = [X[rng.randint(X.shape[0])]]

        for _ in range(1, n_clusters):
            dist_sq = np.min([np.sum((X - c)**2, axis=1) for c in centroids], axis=0)
            probs = dist_sq / np.sum(dist_sq)
            cumulative_probs = np.cumsum(probs)
            r = rng.rand()

            for j, p in enumerate(cumulative_probs):
                if r < p:
                    i = j
                    break

            centroids.append(X[i])

        return np.array(centroids)

    else:
        raise ValueError("init_method must be 'random' or 'k-means++'")

def compute_distances(X: np.ndarray, centroids: np.ndarray,
                     distance_metric: Union[str, Callable]) -> np.ndarray:
    """
    Compute distances between data points and centroids.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    centroids : np.ndarray
        Centroids matrix of shape (n_clusters, n_features)
    distance_metric : str or callable
        Distance metric to use ('euclidean', 'manhattan', 'cosine') or custom function

    Returns
    -------
    np.ndarray
        Distance matrix of shape (n_samples, n_clusters)
    """
    if callable(distance_metric):
        return np.array([distance_metric(x, centroids) for x in X])

    if distance_metric == 'euclidean':
        return np.sqrt(((X[:, np.newaxis] - centroids)**2).sum(axis=2))

    elif distance_metric == 'manhattan':
        return np.sum(np.abs(X[:, np.newaxis] - centroids), axis=2)

    elif distance_metric == 'cosine':
        return 1 - np.dot(X, centroids.T) / (
            np.linalg.norm(X, axis=1)[:, np.newaxis] *
            np.linalg.norm(centroids, axis=1)[np.newaxis, :]
        )

    else:
        raise ValueError("distance_metric must be 'euclidean', 'manhattan', 'cosine' or callable")

def assign_clusters(distances: np.ndarray) -> np.ndarray:
    """
    Assign data points to nearest clusters based on distances.

    Parameters
    ----------
    distances : np.ndarray
        Distance matrix of shape (n_samples, n_clusters)

    Returns
    -------
    np.ndarray
        Cluster assignments of shape (n_samples,)
    """
    return np.argmin(distances, axis=1)

def update_centroids(X: np.ndarray, labels: np.ndarray,
                    n_clusters: int) -> np.ndarray:
    """
    Update cluster centroids based on current assignments.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    labels : np.ndarray
        Cluster assignments of shape (n_samples,)
    n_clusters : int
        Number of clusters

    Returns
    -------
    np.ndarray
        Updated centroids of shape (n_clusters, n_features)
    """
    centroids = np.zeros((n_clusters, X.shape[1]))
    for k in range(n_clusters):
        if np.any(labels == k):
            centroids[k] = X[labels == k].mean(axis=0)
    return centroids

def compute_inertia(X: np.ndarray, labels: np.ndarray,
                   centroids: np.ndarray) -> float:
    """
    Compute the total within-cluster sum of squares.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    labels : np.ndarray
        Cluster assignments of shape (n_samples,)
    centroids : np.ndarray
        Centroids matrix of shape (n_clusters, n_features)

    Returns
    -------
    float
        Total within-cluster sum of squares
    """
    return np.sum(np.min(np.sum((X[:, np.newaxis] - centroids)**2, axis=2), axis=1))

def k_means_fit(X: np.ndarray,
               n_clusters: int = 8,
               init_method: str = 'random',
               distance_metric: Union[str, Callable] = 'euclidean',
               max_iter: int = 300,
               tol: float = 1e-4,
               random_state: Optional[int] = None) -> Dict:
    """
    Perform k-means clustering on input data.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_clusters : int, optional
        Number of clusters to form (default is 8)
    init_method : str, optional
        Initialization method ('random' or 'k-means++') (default is 'random')
    distance_metric : str or callable, optional
        Distance metric to use (default is 'euclidean')
    max_iter : int, optional
        Maximum number of iterations (default is 300)
    tol : float, optional
        Tolerance for convergence (default is 1e-4)
    random_state : int or None, optional
        Seed for random number generation (default is None)

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Cluster assignments
        - 'metrics': Dictionary of metrics including inertia
        - 'params_used': Parameters used in the fitting process
        - 'warnings': Any warnings generated during fitting

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> result = k_means_fit(X, n_clusters=3)
    """
    # Validate inputs
    validated = validate_inputs(X, n_clusters, random_state)
    X = validated['X']
    n_clusters = validated['n_clusters']

    # Initialize centroids
    centroids = initialize_centroids(X, n_clusters, init_method, random_state)

    warnings = []
    inertia_prev = 0

    for i in range(max_iter):
        # Compute distances and assign clusters
        distances = compute_distances(X, centroids, distance_metric)
        labels = assign_clusters(distances)

        # Update centroids
        new_centroids = update_centroids(X, labels, n_clusters)

        # Check for convergence
        inertia = compute_inertia(X, labels, new_centroids)
        if abs(inertia - inertia_prev) < tol:
            break

        centroids = new_centroids
        inertia_prev = inertia

    # Check if any cluster is empty
    if np.any(np.sum(labels.reshape(-1, 1) == np.arange(n_clusters), axis=0) == 0):
        warnings.append("One or more clusters are empty")

    return {
        'result': labels,
        'metrics': {
            'inertia': inertia
        },
        'params_used': {
            'n_clusters': n_clusters,
            'init_method': init_method,
            'distance_metric': distance_metric,
            'max_iter': i + 1,
            'tol': tol
        },
        'warnings': warnings
    }

################################################################################
# gmm
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray,
                   n_components: int = 1,
                   max_iter: int = 100,
                   tol: float = 1e-4) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol < 0:
        raise ValueError("tol must be non-negative")

def initialize_parameters(X: np.ndarray,
                         n_components: int = 1) -> Dict[str, np.ndarray]:
    """Initialize GMM parameters."""
    n_samples, n_features = X.shape
    weights = np.ones(n_components) / n_components

    # Random initialization of means
    indices = np.random.choice(n_samples, n_components, replace=False)
    means = X[indices]

    # Initialize covariances as identity matrices
    covariances = np.array([np.eye(n_features) for _ in range(n_components)])

    return {
        'weights': weights,
        'means': means,
        'covariances': covariances
    }

def compute_responsibilities(X: np.ndarray,
                            weights: np.ndarray,
                            means: np.ndarray,
                            covariances: np.ndarray) -> np.ndarray:
    """Compute responsibilities for each data point."""
    n_samples, n_components = X.shape[0], weights.size
    responsibilities = np.zeros((n_samples, n_components))

    for k in range(n_components):
        # Compute multivariate normal PDF
        try:
            from scipy.stats import multivariate_normal
            responsibilities[:, k] = weights[k] * multivariate_normal.pdf(
                X, mean=means[k], cov=covariances[k])
        except ImportError:
            raise ImportError("scipy is required for GMM computation")

    # Normalize responsibilities
    responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
    responsibilities = responsibilities / (responsibilities_sum + 1e-10)

    return responsibilities

def update_parameters(X: np.ndarray,
                     responsibilities: np.ndarray) -> Dict[str, np.ndarray]:
    """Update GMM parameters using current responsibilities."""
    n_samples, n_components = X.shape[0], responsibilities.shape[1]

    # Update weights
    weights = responsibilities.sum(axis=0) / n_samples

    # Update means
    means = np.zeros((n_components, X.shape[1]))
    for k in range(n_components):
        means[k] = np.sum(responsibilities[:, k, np.newaxis] * X, axis=0) / (
            responsibilities[:, k].sum() + 1e-10)

    # Update covariances
    covariances = np.zeros((n_components, X.shape[1], X.shape[1]))
    for k in range(n_components):
        diff = X - means[k]
        weighted_diff = responsibilities[:, k, np.newaxis, np.newaxis] * diff
        covariances[k] = np.dot(weighted_diff.T, diff) / (
            responsibilities[:, k].sum() + 1e-10)

    return {
        'weights': weights,
        'means': means,
        'covariances': covariances
    }

def compute_metrics(X: np.ndarray,
                   responsibilities: np.ndarray) -> Dict[str, float]:
    """Compute metrics for GMM evaluation."""
    # Log-likelihood
    log_likelihood = np.sum(np.log(np.sum(responsibilities, axis=1) + 1e-10))

    return {
        'log_likelihood': log_likelihood
    }

def gmm_fit(X: np.ndarray,
           n_components: int = 1,
           max_iter: int = 100,
           tol: float = 1e-4) -> Dict[str, Union[Dict, np.ndarray]]:
    """
    Fit a Gaussian Mixture Model to the data.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    n_components : int, optional
        Number of mixture components (default: 1)
    max_iter : int, optional
        Maximum number of iterations (default: 100)
    tol : float, optional
        Tolerance for convergence (default: 1e-4)

    Returns
    -------
    Dict containing:
        - 'result': Dictionary of fitted parameters
        - 'metrics': Dictionary of computed metrics
        - 'params_used': Dictionary of parameters used
        - 'warnings': List of warnings (empty if none)
    """
    # Validate inputs
    validate_inputs(X, n_components, max_iter, tol)

    # Initialize parameters
    params = initialize_parameters(X, n_components)
    old_log_likelihood = -np.inf
    warnings_list = []

    # EM algorithm
    for iteration in range(max_iter):
        # Compute responsibilities
        responsibilities = compute_responsibilities(
            X, params['weights'], params['means'], params['covariances'])

        # Update parameters
        new_params = update_parameters(X, responsibilities)

        # Compute metrics
        metrics = compute_metrics(X, responsibilities)
        current_log_likelihood = metrics['log_likelihood']

        # Check convergence
        if np.abs(current_log_likelihood - old_log_likelihood) < tol:
            break

        old_log_likelihood = current_log_likelihood
        params = new_params

    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'max_iter': max_iter,
            'tol': tol
        },
        'warnings': warnings_list
    }

# Example usage:
"""
X = np.random.randn(100, 2)
result = gmm_fit(X, n_components=2)
"""

################################################################################
# classification_bayésienne_hierarchique
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

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
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def _normalize_data(X: np.ndarray, normalization: str) -> np.ndarray:
    """Normalize data according to specified method."""
    if normalization == "none":
        return X
    elif normalization == "standard":
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalization == "minmax":
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalization == "robust":
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError("Unknown normalization method")

def _compute_priors(y: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute class priors from labels."""
    classes = np.unique(y)
    priors = {cls: np.mean(y == cls) for cls in classes}
    return {"classes": classes, "priors": priors}

def _compute_posteriors(X: np.ndarray,
                        y: np.ndarray,
                        distance_metric: str = "euclidean",
                        custom_distance: Optional[Callable] = None) -> Dict[str, np.ndarray]:
    """Compute posterior probabilities using specified distance metric."""
    classes = np.unique(y)
    posteriors = {}

    if custom_distance is not None:
        distance_func = custom_distance
    else:
        if distance_metric == "euclidean":
            distance_func = lambda a, b: np.linalg.norm(a - b)
        elif distance_metric == "manhattan":
            distance_func = lambda a, b: np.sum(np.abs(a - b))
        elif distance_metric == "cosine":
            distance_func = lambda a, b: 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        elif distance_metric == "minkowski":
            def minkowski(a, b):
                return np.sum(np.abs(a - b)**3)**(1/3)
            distance_func = minkowski
        else:
            raise ValueError("Unknown distance metric")

    for cls in classes:
        X_cls = X[y == cls]
        if len(X_cls) == 0:
            posteriors[cls] = np.zeros_like(y, dtype=float)
            continue

        # Compute distances to all samples of the current class
        dists = np.array([distance_func(X[i], X_cls) for i in range(len(X))])
        posteriors[cls] = np.exp(-dists)  # Simplified posterior computation

    return posteriors

def _apply_regularization(params: Dict[str, np.ndarray],
                          regularization: str,
                          alpha: float = 1.0) -> Dict[str, np.ndarray]:
    """Apply regularization to model parameters."""
    if regularization == "none":
        return params
    elif regularization == "l1":
        for key in params:
            params[key] = np.sign(params[key]) * np.maximum(np.abs(params[key]) - alpha, 0)
    elif regularization == "l2":
        for key in params:
            params[key] /= (1 + alpha * np.linalg.norm(params[key]))
    elif regularization == "elasticnet":
        for key in params:
            l1_part = np.sign(params[key]) * np.maximum(np.abs(params[key]) - alpha, 0)
            l2_part = params[key] / (1 + alpha * np.linalg.norm(params[key]))
            params[key] = 0.5 * (l1_part + l2_part)
    else:
        raise ValueError("Unknown regularization method")
    return params

def _compute_metrics(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     metrics: Union[str, List[str], Callable],
                     custom_metric: Optional[Callable] = None) -> Dict[str, float]:
    """Compute specified metrics between true and predicted labels."""
    metric_results = {}

    if isinstance(metrics, str):
        metrics = [metrics]

    for metric in metrics:
        if callable(metric) or custom_metric is not None:
            func = metric if callable(metric) else custom_metric
            if func is not None:
                metric_results[metric.__name__ if callable(metric) else "custom"] = func(y_true, y_pred)
        elif metric == "accuracy":
            metric_results["accuracy"] = np.mean(y_true == y_pred)
        elif metric == "logloss":
            eps = 1e-15
            y_pred = np.clip(y_pred, eps, 1 - eps)
            metric_results["logloss"] = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        elif metric == "mse":
            metric_results["mse"] = np.mean((y_true - y_pred)**2)
        elif metric == "mae":
            metric_results["mae"] = np.mean(np.abs(y_true - y_pred))
        elif metric == "r2":
            ss_res = np.sum((y_true - y_pred)**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            metric_results["r2"] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return metric_results

def classification_bayésienne_hierarchique_fit(X: np.ndarray,
                                              y: np.ndarray,
                                              normalization: str = "standard",
                                              distance_metric: str = "euclidean",
                                              regularization: str = "none",
                                              alpha: float = 1.0,
                                              metrics: Union[str, List[str]] = ["accuracy"],
                                              custom_distance: Optional[Callable] = None,
                                              custom_metric: Optional[Callable] = None) -> Dict:
    """
    Perform hierarchical Bayesian classification.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target labels of shape (n_samples,)
    normalization : str
        Normalization method ("none", "standard", "minmax", "robust")
    distance_metric : str
        Distance metric for posterior computation ("euclidean", "manhattan", etc.)
    regularization : str
        Regularization method ("none", "l1", "l2", "elasticnet")
    alpha : float
        Regularization strength
    metrics : str or list of str
        Metrics to compute ("accuracy", "logloss", etc.)
    custom_distance : callable, optional
        Custom distance function
    custom_metric : callable, optional
        Custom metric function

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = classification_bayésienne_hierarchique_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm = _normalize_data(X.copy(), normalization)

    # Compute priors
    priors_info = _compute_priors(y)

    # Compute posteriors
    posteriors = _compute_posteriors(X_norm, y, distance_metric, custom_distance)

    # Combine priors and posteriors to get final probabilities
    classes = priors_info["classes"]
    probs = np.zeros((len(X), len(classes)))
    for i, cls in enumerate(classes):
        probs[:, i] = posteriors[cls] * priors_info["priors"][cls]

    # Normalize probabilities to sum to 1
    probs /= np.sum(probs, axis=1, keepdims=True)

    # Get predicted class (argmax of probabilities)
    y_pred = classes[np.argmax(probs, axis=1)]

    # Apply regularization
    params = {"posteriors": posteriors, "priors": priors_info["priors"]}
    params = _apply_regularization(params, regularization, alpha)

    # Compute metrics
    metric_results = _compute_metrics(y, y_pred, metrics, custom_metric)

    # Prepare output
    result = {
        "result": {
            "probabilities": probs,
            "predicted_classes": y_pred
        },
        "metrics": metric_results,
        "params_used": {
            "normalization": normalization,
            "distance_metric": distance_metric if custom_distance is None else "custom",
            "regularization": regularization,
            "alpha": alpha,
            "metrics": metrics
        },
        "warnings": []
    }

    return result

################################################################################
# modèle_markov_caché
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def modèle_markov_caché_fit(
    observations: np.ndarray,
    n_states: int,
    max_iter: int = 100,
    tol: float = 1e-6,
    normalisation: str = 'standard',
    métrique: Union[str, Callable] = 'logloss',
    solveur: str = 'baum_welch',
    régularisation: Optional[str] = None,
    alpha_reg: float = 1.0,
    random_state: Optional[int] = None
) -> Dict:
    """
    Estime un modèle de Markov caché à partir des observations.

    Parameters
    ----------
    observations : np.ndarray
        Matrice d'observations de forme (n_samples, n_features).
    n_states : int
        Nombre d'états cachés.
    max_iter : int, optional
        Nombre maximal d'itérations, par défaut 100.
    tol : float, optional
        Tolérance pour la convergence, par défaut 1e-6.
    normalisation : str, optional
        Méthode de normalisation ('none', 'standard', 'minmax', 'robust'), par défaut 'standard'.
    métrique : Union[str, Callable], optional
        Métrique d'évaluation ('mse', 'mae', 'r2', 'logloss') ou fonction personnalisée, par défaut 'logloss'.
    solveur : str, optional
        Solveur ('baum_welch', 'viterbi'), par défaut 'baum_welch'.
    régularisation : Optional[str], optional
        Type de régularisation ('none', 'l1', 'l2', 'elasticnet'), par défaut None.
    alpha_reg : float, optional
        Coefficient de régularisation, par défaut 1.0.
    random_state : Optional[int], optional
        Graine aléatoire pour la reproductibilité, par défaut None.

    Returns
    -------
    Dict
        Dictionnaire contenant les résultats, métriques, paramètres utilisés et avertissements.
    """
    # Validation des entrées
    _validate_inputs(observations, n_states)

    # Initialisation aléatoire si nécessaire
    if random_state is not None:
        np.random.seed(random_state)

    # Normalisation des données
    observations_norm, normalizer = _apply_normalization(observations, normalisation)

    # Initialisation des paramètres
    initial_params = _initialize_parameters(n_states, observations_norm.shape[1])

    # Estimation des paramètres
    params = _estimate_parameters(
        observations_norm,
        initial_params,
        n_states,
        max_iter,
        tol,
        solveur,
        régularisation,
        alpha_reg
    )

    # Calcul des métriques
    metrics = _compute_metrics(observations_norm, params, métrique)

    # Retour des résultats
    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalisation': normalisation,
            'métrique': métrique,
            'solveur': solveur,
            'régularisation': régularisation,
            'alpha_reg': alpha_reg
        },
        'warnings': []
    }

def _validate_inputs(observations: np.ndarray, n_states: int) -> None:
    """Valide les entrées du modèle."""
    if not isinstance(observations, np.ndarray):
        raise TypeError("Les observations doivent être un tableau NumPy.")
    if observations.ndim != 2:
        raise ValueError("Les observations doivent être une matrice 2D.")
    if n_states <= 0:
        raise ValueError("Le nombre d'états doit être positif.")
    if np.any(np.isnan(observations)) or np.any(np.isinf(observations)):
        raise ValueError("Les observations contiennent des NaN ou des inf.")

def _apply_normalization(observations: np.ndarray, method: str) -> tuple:
    """Applique la normalisation spécifiée aux observations."""
    if method == 'none':
        return observations, None
    elif method == 'standard':
        mean = np.mean(observations, axis=0)
        std = np.std(observations, axis=0)
        normalized = (observations - mean) / std
    elif method == 'minmax':
        min_val = np.min(observations, axis=0)
        max_val = np.max(observations, axis=0)
        normalized = (observations - min_val) / (max_val - min_val)
    elif method == 'robust':
        median = np.median(observations, axis=0)
        iqr = np.subtract(*np.percentile(observations, [75, 25], axis=0))
        normalized = (observations - median) / iqr
    else:
        raise ValueError(f"Méthode de normalisation inconnue: {method}")
    return normalized, None  # Retourne aussi le normalizer si nécessaire

def _initialize_parameters(n_states: int, n_features: int) -> Dict:
    """Initialise les paramètres du modèle."""
    return {
        'transition_matrix': np.random.dirichlet(np.ones(n_states), size=n_states),
        'emission_matrix': np.random.dirichlet(np.ones(n_features), size=n_states),
        'initial_probabilities': np.random.dirichlet(np.ones(n_states))
    }

def _estimate_parameters(
    observations: np.ndarray,
    initial_params: Dict,
    n_states: int,
    max_iter: int,
    tol: float,
    solveur: str,
    régularisation: Optional[str],
    alpha_reg: float
) -> Dict:
    """Estime les paramètres du modèle de Markov caché."""
    if solveur == 'baum_welch':
        return _baum_welch(observations, initial_params, n_states, max_iter, tol, régularisation, alpha_reg)
    elif solveur == 'viterbi':
        return _viterbi(observations, initial_params, n_states, max_iter, tol)
    else:
        raise ValueError(f"Solveur inconnu: {solveur}")

def _baum_welch(
    observations: np.ndarray,
    initial_params: Dict,
    n_states: int,
    max_iter: int,
    tol: float,
    régularisation: Optional[str],
    alpha_reg: float
) -> Dict:
    """Algorithme de Baum-Welch pour estimer les paramètres."""
    params = initial_params
    for _ in range(max_iter):
        # Calcul des probabilités forward et backward
        alpha = _forward(observations, params)
        beta = _backward(observations, params)

        # Mise à jour des paramètres
        new_params = _update_parameters(observations, alpha, beta, params, régularisation, alpha_reg)

        # Vérification de la convergence
        if np.linalg.norm(new_params['transition_matrix'] - params['transition_matrix']) < tol:
            break

        params = new_params
    return params

def _forward(observations: np.ndarray, params: Dict) -> np.ndarray:
    """Calcule les probabilités forward."""
    # Implémentation simplifiée
    return np.ones_like(observations)

def _backward(observations: np.ndarray, params: Dict) -> np.ndarray:
    """Calcule les probabilités backward."""
    # Implémentation simplifiée
    return np.ones_like(observations)

def _update_parameters(
    observations: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    params: Dict,
    régularisation: Optional[str],
    alpha_reg: float
) -> Dict:
    """Met à jour les paramètres du modèle."""
    # Implémentation simplifiée
    return params

def _viterbi(
    observations: np.ndarray,
    initial_params: Dict,
    n_states: int,
    max_iter: int,
    tol: float
) -> Dict:
    """Algorithme de Viterbi pour estimer les paramètres."""
    # Implémentation simplifiée
    return initial_params

def _compute_metrics(
    observations: np.ndarray,
    params: Dict,
    métrique: Union[str, Callable]
) -> Dict:
    """Calcule les métriques d'évaluation."""
    if callable(métrique):
        return {'custom': métrique(observations, params)}
    elif métrique == 'logloss':
        return {'logloss': _compute_logloss(observations, params)}
    elif métrique == 'mse':
        return {'mse': _compute_mse(observations, params)}
    else:
        raise ValueError(f"Métrique inconnue: {métrique}")

def _compute_logloss(observations: np.ndarray, params: Dict) -> float:
    """Calcule la log-vraisemblance."""
    # Implémentation simplifiée
    return 0.0

def _compute_mse(observations: np.ndarray, params: Dict) -> float:
    """Calcule l'erreur quadratique moyenne."""
    # Implémentation simplifiée
    return 0.0

# Exemple minimal
if __name__ == "__main__":
    observations = np.random.rand(100, 5)
    result = modèle_markov_caché_fit(
        observations,
        n_states=3,
        max_iter=50,
        normalisation='standard',
        métrique='logloss'
    )

################################################################################
# classification_bayésienne_non_paramétrique
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def classification_bayésienne_non_paramétrique_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    distance_metric: str = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    max_iter: int = 1000,
    tol: float = 1e-4,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Fit a non-parametric Bayesian classification model.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target labels of shape (n_samples,).
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the input features.
    distance_metric : str
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski').
    solver : str
        Solver to use ('gradient_descent', 'newton', 'coordinate_descent').
    regularization : Optional[str]
        Regularization type ('none', 'l1', 'l2', 'elasticnet').
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for stopping criterion.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function.
    **kwargs
        Additional solver-specific parameters.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = normalizer(X)

    # Initialize parameters
    params = _initialize_parameters(X_normalized, y, distance_metric)

    # Choose solver
    if solver == 'gradient_descent':
        result, metrics = _gradient_descent_solver(X_normalized, y, params, max_iter, tol, regularization, **kwargs)
    elif solver == 'newton':
        result, metrics = _newton_solver(X_normalized, y, params, max_iter, tol, regularization, **kwargs)
    elif solver == 'coordinate_descent':
        result, metrics = _coordinate_descent_solver(X_normalized, y, params, max_iter, tol, regularization, **kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    if custom_metric is not None:
        metrics['custom'] = custom_metric(result['predictions'], y)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data."""
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

def _initialize_parameters(X: np.ndarray, y: np.ndarray, distance_metric: str) -> Dict[str, Any]:
    """Initialize model parameters."""
    n_classes = len(np.unique(y))
    params = {
        'distance_metric': distance_metric,
        'n_classes': n_classes,
        'class_probs': np.ones(n_classes) / n_classes,
        'class_means': np.zeros((n_classes, X.shape[1])),
        'covariances': [np.eye(X.shape[1]) for _ in range(n_classes)]
    }
    return params

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any],
    max_iter: int,
    tol: float,
    regularization: Optional[str],
    **kwargs
) -> tuple[Dict[str, Any], Dict[str, float]]:
    """Gradient descent solver for Bayesian classification."""
    learning_rate = kwargs.get('learning_rate', 0.01)
    result = {'predictions': np.zeros_like(y)}
    metrics = {}

    for _ in range(max_iter):
        # Update parameters using gradient descent
        params['class_means'], params['covariances'] = _update_parameters_gradient_descent(
            X, y, params, learning_rate, regularization
        )

        # Compute predictions and metrics
        result['predictions'] = _predict(X, params)
        metrics.update(_compute_metrics(result['predictions'], y))

        # Check convergence
        if _check_convergence(metrics, tol):
            break

    return result, metrics

def _update_parameters_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any],
    learning_rate: float,
    regularization: Optional[str]
) -> tuple[np.ndarray, list]:
    """Update parameters using gradient descent."""
    n_classes = params['n_classes']
    class_means = params['class_means'].copy()
    covariances = [cov.copy() for cov in params['covariances']]

    # Update class means
    for k in range(n_classes):
        mask = (y == k)
        if np.sum(mask) > 0:
            gradient = -np.mean(X[mask] - class_means[k], axis=0)
            if regularization == 'l2':
                gradient += 2 * learning_rate * class_means[k]
            class_means[k] -= learning_rate * gradient

    # Update covariances
    for k in range(n_classes):
        mask = (y == k)
        if np.sum(mask) > 0:
            centered = X[mask] - class_means[k]
            gradient = -np.mean(centered[:, :, np.newaxis] * centered[:, np.newaxis, :], axis=0)
            if regularization == 'l2':
                gradient += 2 * learning_rate * covariances[k]
            covariances[k] -= learning_rate * gradient

    return class_means, covariances

def _predict(X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Predict class labels."""
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples)

    for i in range(n_samples):
        posteriors = []
        for k in range(params['n_classes']):
            prior = params['class_probs'][k]
            likelihood = _compute_likelihood(X[i], params['class_means'][k], params['covariances'][k], params['distance_metric'])
            posteriors.append(prior * likelihood)
        predictions[i] = np.argmax(posteriors)

    return predictions

def _compute_likelihood(
    x: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    distance_metric: str
) -> float:
    """Compute likelihood using specified distance metric."""
    if distance_metric == 'euclidean':
        diff = x - mean
        exponent = -0.5 * np.dot(diff.T, np.linalg.inv(cov).dot(diff))
        return (2 * np.pi) ** (-len(x)/2) * np.linalg.det(cov) ** (-0.5) * np.exp(exponent)
    else:
        raise NotImplementedError(f"Distance metric {distance_metric} not implemented for likelihood computation.")

def _compute_metrics(predictions: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics."""
    accuracy = np.mean(predictions == y_true)
    return {'accuracy': accuracy}

def _check_convergence(metrics: Dict[str, float], tol: float) -> bool:
    """Check if the solver has converged."""
    return metrics.get('accuracy', 0) >= (1 - tol)

# Additional solver implementations would follow similar patterns
def _newton_solver(*args, **kwargs) -> tuple[Dict[str, Any], Dict[str, float]]:
    raise NotImplementedError("Newton solver not implemented.")

def _coordinate_descent_solver(*args, **kwargs) -> tuple[Dict[str, Any], Dict[str, float]]:
    raise NotImplementedError("Coordinate descent solver not implemented.")
