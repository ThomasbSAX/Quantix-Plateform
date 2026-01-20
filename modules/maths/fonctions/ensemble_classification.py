"""
Quantix – Module ensemble_classification
Généré automatiquement
Date: 2026-01-07
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
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive")
    if not (0 < max_samples <= 1):
        raise ValueError("max_samples must be in (0, 1]")
    if not (0 < max_features <= 1):
        raise ValueError("max_features must be in (0, 1]")

def _bootstrap_sample(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a bootstrap sample from the data."""
    n_samples = int(np.round(max_samples * len(X)))
    indices = np.random.choice(len(X), size=n_samples, replace=True)
    return X[indices], y[indices]

def _fit_estimator(
    estimator: Callable,
    X: np.ndarray,
    y: np.ndarray
) -> Any:
    """Fit a single estimator."""
    return estimator.fit(X, y)

def _predict_estimator(
    estimator: Any,
    X: np.ndarray
) -> np.ndarray:
    """Make predictions using a single estimator."""
    return estimator.predict(X)

def bagging_fit(
    base_estimator: Callable,
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    max_samples: float = 1.0,
    max_features: float = 1.0,
    bootstrap: bool = True,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Bagging ensemble classifier.

    Parameters:
    -----------
    base_estimator : callable
        The base estimator to fit on random subsets of the data.
    X : numpy.ndarray
        Training data of shape (n_samples, n_features).
    y : numpy.ndarray
        Target values of shape (n_samples,).
    n_estimators : int, default=100
        The number of base estimators in the ensemble.
    max_samples : float, default=1.0
        The fraction of samples to draw for each base estimator.
    max_features : float, default=1.0
        The fraction of features to draw for each base estimator.
    bootstrap : bool, default=True
        Whether to use bootstrap sampling.
    random_state : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    dict
        A dictionary containing the fitted ensemble, metrics, and other information.
    """
    _validate_inputs(X, y, n_estimators, max_samples, max_features, bootstrap)

    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_features = X.shape
    estimators = []

    for _ in range(n_estimators):
        if bootstrap:
            X_sample, y_sample = _bootstrap_sample(X, y, max_samples)
        else:
            X_sample, y_sample = X, y

        if max_features < 1.0:
            n_features_sample = int(np.round(max_features * n_features))
            feature_indices = np.random.choice(n_features, size=n_features_sample, replace=False)
            X_sample = X_sample[:, feature_indices]

        estimator = _fit_estimator(base_estimator, X_sample, y_sample)
        estimators.append(estimator)

    def predict(X_pred: np.ndarray) -> np.ndarray:
        """Predict using the bagging ensemble."""
        predictions = np.array([_predict_estimator(est, X_pred) for est in estimators])
        return np.mean(predictions, axis=0)

    result = {
        "estimators": estimators,
        "predict": predict
    }

    return {
        "result": result,
        "metrics": {},
        "params_used": {
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "max_features": max_features,
            "bootstrap": bootstrap
        },
        "warnings": []
    }

# Example usage:
"""
from sklearn.tree import DecisionTreeClassifier

X_train = np.random.rand(100, 5)
y_train = np.random.randint(0, 2, size=100)

bagging_model = bagging_fit(
    base_estimator=DecisionTreeClassifier(),
    X=X_train,
    y=y_train,
    n_estimators=10,
    max_samples=0.8,
    max_features=0.5
)

predictions = bagging_model["result"]["predict"](X_train)
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
    loss: str = 'exponential',
    criterion: str = 'mse',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    random_state: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Fit a boosting ensemble model to the training data.

    Parameters:
    -----------
    X : np.ndarray
        Training data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    n_estimators : int, optional
        Number of boosting stages to be run. Default is 100.
    learning_rate : float, optional
        Shrinkage factor for the contribution of each tree. Default is 0.1.
    loss : str, optional
        Loss function to be optimized. Options are 'exponential', 'deviance'. Default is 'exponential'.
    criterion : str, optional
        Function to measure the quality of a split. Options are 'mse', 'mae'. Default is 'mse'.
    normalizer : Callable, optional
        Function to normalize the data. Default is None.
    metric : str or Callable, optional
        Metric to evaluate the quality of the split. Default is 'mse'.
    random_state : int, optional
        Seed for random number generation. Default is None.
    verbose : bool, optional
        Whether to print progress messages. Default is False.

    Returns:
    --------
    dict
        A dictionary containing the fitted model, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Initialize random state
    rng = np.random.RandomState(random_state)

    # Normalize data if a normalizer is provided
    X_norm = _apply_normalization(X, normalizer)

    # Initialize weights
    sample_weights = np.ones(y.shape[0]) / y.shape[0]

    # Initialize predictions
    y_pred = np.zeros(y.shape[0])

    # Store estimators
    estimators = []

    for i in range(n_estimators):
        # Fit a weak learner
        estimator = _fit_weak_learner(
            X_norm, y - learning_rate * y_pred,
            sample_weights, criterion, rng
        )

        # Update predictions
        y_pred += learning_rate * estimator.predict(X_norm)

        # Compute loss gradient and hessian
        if loss == 'exponential':
            grad = _compute_exponential_gradient(y, y_pred)
            hess = _compute_exponential_hessian(y, y_pred)
        elif loss == 'deviance':
            grad = _compute_deviance_gradient(y, y_pred)
            hess = _compute_deviance_hessian(y, y_pred)

        # Update sample weights
        sample_weights = _update_sample_weights(grad, hess)

        # Store estimator
        estimators.append(estimator)

        if verbose:
            print(f'Boosting stage {i + 1}/{n_estimators}')

    # Compute metrics
    metrics = _compute_metrics(y, y_pred, metric)

    return {
        'result': {'estimators': estimators},
        'metrics': metrics,
        'params_used': {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'loss': loss,
            'criterion': criterion
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
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to the data."""
    if normalizer is not None:
        return normalizer(X)
    return X

def _fit_weak_learner(
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    criterion: str,
    rng: np.random.RandomState
) -> Any:
    """Fit a weak learner."""
    # This is a placeholder for the actual weak learner fitting logic
    # In practice, this would be replaced with an actual implementation
    return WeakLearner()

def _compute_exponential_gradient(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute the gradient for exponential loss."""
    return -np.exp(-y * y_pred) * y

def _compute_exponential_hessian(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute the hessian for exponential loss."""
    return np.exp(-y * y_pred)

def _compute_deviance_gradient(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute the gradient for deviance loss."""
    return (y_pred - y) / (1 + np.exp(-(2 * y * y_pred)))

def _compute_deviance_hessian(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute the hessian for deviance loss."""
    return 1 / (4 * np.cosh(y_pred - y) ** 2)

def _update_sample_weights(grad: np.ndarray, hess: np.ndarray) -> np.ndarray:
    """Update the sample weights."""
    return (grad ** 2) / (hess + 1e-10)

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """Compute the metrics."""
    if callable(metric):
        return {'custom_metric': metric(y_true, y_pred)}
    elif metric == 'mse':
        return {'mse': np.mean((y_true - y_pred) ** 2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(y_true - y_pred))}
    else:
        raise ValueError(f"Unknown metric: {metric}")

class WeakLearner:
    """Placeholder for a weak learner."""
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the weak learner."""
        return np.zeros(X.shape[0])

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
    max_features: Union[str, float] = "auto",
    bootstrap: bool = True,
    criterion: str = "gini",
    random_state: Optional[int] = None,
    n_jobs: int = 1,
    verbose: int = 0
) -> Dict[str, Any]:
    """
    Fit a random forest classifier.

    Parameters:
    -----------
    X : np.ndarray
        Training data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    n_estimators : int, optional
        Number of trees in the forest (default=100).
    max_depth : int, optional
        Maximum depth of the tree (default=None).
    min_samples_split : int, optional
        Minimum number of samples required to split a node (default=2).
    min_samples_leaf : int, optional
        Minimum number of samples required at each leaf node (default=1).
    max_features : str or float, optional
        Number of features to consider when looking for the best split.
        "auto" means sqrt(n_features) (default="auto").
    bootstrap : bool, optional
        Whether bootstrap samples are used when building trees (default=True).
    criterion : str, optional
        Function to measure the quality of a split.
        Supported criteria are "gini" for Gini impurity and "entropy" for information gain (default="gini").
    random_state : int, optional
        Controls both the randomness of the bootstrapping of the samples and the splitting of the nodes (default=None).
    n_jobs : int, optional
        Number of jobs to run in parallel (default=1).
    verbose : int, optional
        Controls the verbosity when fitting and predicting (default=0).

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the fitted model and related information.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set random state if provided
    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    # Initialize the ensemble
    trees = []
    feature_importances_ = np.zeros(X.shape[1])

    for _ in range(n_estimators):
        # Bootstrap sample if enabled
        if bootstrap:
            indices = rng.choice(X.shape[0], size=X.shape[0], replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
        else:
            X_sample, y_sample = X, y

        # Grow a tree
        tree = _grow_tree(
            X_sample, y_sample,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            criterion=criterion,
            random_state=rng
        )

        # Update feature importances
        if hasattr(tree, 'feature_importances_'):
            feature_importances_ += tree.feature_importances_

        trees.append(tree)

    # Normalize feature importances
    if n_estimators > 0:
        feature_importances_ /= n_estimators

    # Prepare the output
    result = {
        "trees": trees,
        "feature_importances_": feature_importances_
    }

    metrics = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "bootstrap": bootstrap,
        "criterion": criterion
    }

    params_used = {
        "random_state": random_state,
        "n_jobs": n_jobs
    }

    warnings = []

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
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
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X must not contain NaN or Inf values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y must not contain NaN or Inf values.")

def _grow_tree(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: Union[str, float] = "auto",
    criterion: str = "gini",
    random_state: np.random.RandomState = None
) -> Dict[str, Any]:
    """
    Grow a single decision tree.

    Parameters:
    -----------
    X : np.ndarray
        Training data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    max_depth : int, optional
        Maximum depth of the tree (default=None).
    min_samples_split : int, optional
        Minimum number of samples required to split a node (default=2).
    min_samples_leaf : int, optional
        Minimum number of samples required at each leaf node (default=1).
    max_features : str or float, optional
        Number of features to consider when looking for the best split.
        "auto" means sqrt(n_features) (default="auto").
    criterion : str, optional
        Function to measure the quality of a split.
        Supported criteria are "gini" for Gini impurity and "entropy" for information gain (default="gini").
    random_state : np.random.RandomState, optional
        Random state for reproducibility (default=None).

    Returns:
    --------
    Dict[str, Any]
        A dictionary representing the decision tree.
    """
    # Determine the number of features to consider
    if max_features == "auto":
        n_features = int(np.sqrt(X.shape[1]))
    elif isinstance(max_features, float):
        n_features = max(1, int(X.shape[1] * max_features))
    else:
        n_features = max_features

    # Initialize the tree
    tree = {
        "node": _build_node(
            X, y,
            max_depth=max_depth if max_depth is not None else 0,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=n_features,
            criterion=criterion,
            random_state=random_state
        )
    }

    return tree

def _build_node(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
    max_features: int,
    criterion: str,
    random_state: np.random.RandomState
) -> Dict[str, Any]:
    """
    Build a single node of the decision tree.

    Parameters:
    -----------
    X : np.ndarray
        Training data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    max_depth : int
        Maximum depth of the tree.
    min_samples_split : int
        Minimum number of samples required to split a node.
    min_samples_leaf : int
        Minimum number of samples required at each leaf node.
    max_features : int
        Number of features to consider when looking for the best split.
    criterion : str
        Function to measure the quality of a split.
    random_state : np.random.RandomState
        Random state for reproducibility.

    Returns:
    --------
    Dict[str, Any]
        A dictionary representing the node.
    """
    # Check if the node is a leaf
    if (max_depth == 0 or len(y) < min_samples_split):
        return _create_leaf_node(y)

    # Find the best split
    best_split = _find_best_split(
        X, y,
        max_features=max_features,
        criterion=criterion,
        random_state=random_state
    )

    # If no split is found, create a leaf node
    if best_split is None:
        return _create_leaf_node(y)

    # Split the data
    left_indices = X[:, best_split["feature"]] <= best_split["threshold"]
    right_indices = ~left_indices

    # Recursively build the left and right subtrees
    node = {
        "feature": best_split["feature"],
        "threshold": best_split["threshold"],
        "left": _build_node(
            X[left_indices], y[left_indices],
            max_depth=max_depth - 1,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            criterion=criterion,
            random_state=random_state
        ),
        "right": _build_node(
            X[right_indices], y[right_indices],
            max_depth=max_depth - 1,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            criterion=criterion,
            random_state=random_state
        )
    }

    return node

def _create_leaf_node(y: np.ndarray) -> Dict[str, Any]:
    """
    Create a leaf node.

    Parameters:
    -----------
    y : np.ndarray
        Target values of shape (n_samples,).

    Returns:
    --------
    Dict[str, Any]
        A dictionary representing the leaf node.
    """
    return {
        "value": np.bincount(y.astype(int)).argmax()
    }

def _find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    max_features: int,
    criterion: str,
    random_state: np.random.RandomState
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
        Number of features to consider when looking for the best split.
    criterion : str
        Function to measure the quality of a split.
    random_state : np.random.RandomState
        Random state for reproducibility.

    Returns:
    --------
    Optional[Dict[str, Any]]
        A dictionary representing the best split, or None if no split is found.
    """
    # Randomly select features
    feature_indices = random_state.choice(X.shape[1], size=max_features, replace=False)

    best_split = None
    best_score = -np.inf

    for feature in feature_indices:
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_indices = X[:, feature] <= threshold
            right_indices = ~left_indices

            if len(y[left_indices]) < 2 or len(y[right_indices]) < 2:
                continue

            if criterion == "gini":
                score = _gini_impurity(y[left_indices], y[right_indices])
            elif criterion == "entropy":
                score = _information_gain(y[left_indices], y[right_indices])
            else:
                raise ValueError(f"Unknown criterion: {criterion}")

            if score > best_score:
                best_score = score
                best_split = {
                    "feature": feature,
                    "threshold": threshold
                }

    return best_split

def _gini_impurity(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """
    Calculate the Gini impurity.

    Parameters:
    -----------
    y_left : np.ndarray
        Target values for the left split.
    y_right : np.ndarray
        Target values for the right split.

    Returns:
    --------
    float
        The Gini impurity.
    """
    def gini(y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right

    return (n_left / n_total) * gini(y_left) + (n_right / n_total) * gini(y_right)

def _information_gain(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """
    Calculate the information gain.

    Parameters:
    -----------
    y_left : np.ndarray
        Target values for the left split.
    y_right : np.ndarray
        Target values for the right split.

    Returns:
    --------
    float
        The information gain.
    """
    def entropy(y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right

    h_parent = entropy(np.concatenate([y_left, y_right]))
    h_children = (n_left / n_total) * entropy(y_left) + (n_right / n_total) * entropy(y_right)

    return h_parent - h_children

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
    loss: str = 'deviance',
    criterion: str = 'mse',
    subsample: float = 1.0,
    random_state: Optional[int] = None,
    normalization: str = 'none',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'gradient_descent',
    regularization: str = 'none',
    tol: float = 1e-4,
    n_iter_no_change: int = None,
    validation_fraction: float = 0.1,
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
        Shrinkage factor for the contributions of each tree.
    max_depth : int, optional
        Maximum depth of the individual regression estimators.
    min_samples_split : int, optional
        Minimum number of samples required to split an internal node.
    loss : str, optional
        Loss function to be optimized ('deviance' for classification, 'ls' for regression).
    criterion : str, optional
        Function to measure the quality of a split ('mse' for regression).
    subsample : float, optional
        Fraction of samples to be used for fitting the individual trees.
    random_state : int, optional
        Controls both the randomness of the bootstrapping of the samples and the sampling of the features.
    normalization : str, optional
        Type of normalization to apply ('none', 'standard', 'minmax', 'robust').
    metric : Union[str, Callable], optional
        Metric to evaluate the quality of the prediction ('mse', 'mae', 'r2', custom callable).
    solver : str, optional
        Solver to use for optimization ('gradient_descent', 'newton', 'coordinate_descent').
    regularization : str, optional
        Type of regularization to apply ('none', 'l1', 'l2', 'elasticnet').
    tol : float, optional
        Tolerance for the early stopping.
    n_iter_no_change : int, optional
        Number of iterations with no improvement after which training will be stopped.
    validation_fraction : float, optional
        The proportion of training data to set aside as validation set for early stopping.
    verbose : bool, optional
        Whether to print progress messages.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the fitted model, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_normalized = _apply_normalization(X, normalization)

    # Initialize model parameters
    params_used = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'loss': loss,
        'criterion': criterion,
        'subsample': subsample,
        'random_state': random_state,
        'normalization': normalization,
        'metric': metric,
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'n_iter_no_change': n_iter_no_change,
        'validation_fraction': validation_fraction
    }

    # Initialize the model
    model = _initialize_model(params_used)

    # Fit the model
    model = _fit_gradient_boosting(
        X_normalized, y, model, params_used,
        metric, solver, regularization,
        tol, n_iter_no_change, validation_fraction, verbose
    )

    # Calculate metrics
    metrics = _calculate_metrics(y, model.predict(X_normalized), metric)

    # Prepare the result dictionary
    result = {
        'result': model,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': []
    }

    return result

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

def _apply_normalization(X: np.ndarray, normalization: str) -> np.ndarray:
    """Apply the specified normalization to the data."""
    if normalization == 'standard':
        X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalization == 'minmax':
        X_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalization == 'robust':
        X_normalized = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        X_normalized = X
    return X_normalized

def _initialize_model(params: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize the gradient boosting model."""
    return {
        'estimators': [],
        'learning_rate': params['learning_rate'],
        'loss': params['loss']
    }

def _fit_gradient_boosting(
    X: np.ndarray,
    y: np.ndarray,
    model: Dict[str, Any],
    params: Dict[str, Any],
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    solver: str,
    regularization: str,
    tol: float,
    n_iter_no_change: Optional[int],
    validation_fraction: float,
    verbose: bool
) -> Dict[str, Any]:
    """Fit the gradient boosting model to the data."""
    n_samples = X.shape[0]
    validation_size = int(n_samples * validation_fraction)
    if validation_size > 0:
        np.random.seed(params['random_state'])
        indices = np.random.permutation(n_samples)
        train_indices, val_indices = indices[validation_size:], indices[:validation_size]
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
    else:
        X_train, y_train = X, y
        X_val, y_val = None, None

    # Initialize predictions
    y_pred = np.mean(y_train) if params['loss'] == 'deviance' else 0.0

    # Early stopping variables
    best_metric = float('inf')
    no_change_count = 0

    for i in range(params['n_estimators']):
        # Compute residuals
        residuals = _compute_residuals(y_train, y_pred, params['loss'])

        # Fit a weak learner
        estimator = _fit_weak_learner(
            X_train, residuals,
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            criterion=params['criterion'],
            subsample=params['subsample'],
            random_state=params['random_state']
        )

        # Update predictions
        y_pred += params['learning_rate'] * estimator.predict(X_train)

        # Add the estimator to the model
        model['estimators'].append(estimator)

        # Calculate metrics on validation set if available
        if X_val is not None:
            val_pred = np.mean(y_val) if params['loss'] == 'deviance' else 0.0
            for est in model['estimators']:
                val_pred += params['learning_rate'] * est.predict(X_val)
            current_metric = _calculate_metrics(y_val, val_pred, metric)[metric]

            if verbose:
                print(f"Iteration {i + 1}, Metric: {current_metric}")

            if abs(best_metric - current_metric) < tol:
                no_change_count += 1
            else:
                best_metric = current_metric
                no_change_count = 0

            if n_iter_no_change is not None and no_change_count >= n_iter_no_change:
                if verbose:
                    print(f"Early stopping at iteration {i + 1}")
                break

    return model

def _compute_residuals(y_true: np.ndarray, y_pred: np.ndarray, loss: str) -> np.ndarray:
    """Compute the residuals based on the specified loss function."""
    if loss == 'deviance':
        return y_true - (y_pred + 1e-10) / (1 + np.exp(-(y_pred + 1e-10)) + 1e-10)
    elif loss == 'ls':
        return y_true - y_pred
    else:
        raise ValueError(f"Unsupported loss function: {loss}")

def _fit_weak_learner(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int,
    min_samples_split: int,
    criterion: str,
    subsample: float,
    random_state: Optional[int]
) -> Any:
    """Fit a weak learner (decision tree) to the data."""
    if subsample < 1.0:
        np.random.seed(random_state)
        n_samples = int(X.shape[0] * subsample)
        indices = np.random.choice(X.shape[0], n_samples, replace=False)
        X_subsampled = X[indices]
        y_subsampled = y[indices]
    else:
        X_subsampled, y_subsampled = X, y

    # Here you would implement the actual decision tree fitting logic
    # For simplicity, we return a mock object with a predict method
    class WeakLearner:
        def __init__(self, max_depth, min_samples_split, criterion):
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.criterion = criterion

        def predict(self, X: np.ndarray) -> np.ndarray:
            # Mock prediction for demonstration purposes
            return np.zeros(X.shape[0])

    return WeakLearner(max_depth, min_samples_split, criterion)

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Calculate the specified metrics."""
    metrics_dict = {}

    if isinstance(metric, str):
        if metric == 'mse':
            metrics_dict['mse'] = np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            metrics_dict['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics_dict['r2'] = 1 - (ss_res / ss_tot)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    else:
        metrics_dict['custom'] = metric(y_true, y_pred)

    return metrics_dict

# Example usage:
"""
X_train = np.random.rand(100, 5)
y_train = np.random.randint(0, 2, size=100)

result = gradient_boosting_fit(
    X_train, y_train,
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=2,
    loss='deviance',
    criterion='mse',
    subsample=1.0,
    random_state=42,
    normalization='standard',
    metric='mse',
    solver='gradient_descent',
    regularization='none',
    tol=1e-4,
    n_iter_no_change=5,
    validation_fraction=0.1,
    verbose=True
)
"""

################################################################################
# adaboost
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> None:
    """
    Validate input data for AdaBoost.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    sample_weight : Optional[np.ndarray], default=None
        Sample weights array of shape (n_samples,).

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if sample_weight is not None:
        if X.shape[0] != sample_weight.shape[0]:
            raise ValueError("X and sample_weight must have the same number of samples.")
        if np.any(sample_weight <= 0):
            raise ValueError("sample_weight must be positive.")

def compute_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> float:
    """
    Compute the weighted error of predictions.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.
    sample_weight : Optional[np.ndarray], default=None
        Sample weights.

    Returns
    ------
    float
        Weighted error.
    """
    if sample_weight is None:
        sample_weight = np.ones_like(y_true)
    return np.sum(sample_weight * (y_pred != y_true)) / np.sum(sample_weight)

def compute_weights(
    error: float,
    sample_weight: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute the weights for the next iteration.

    Parameters
    ----------
    error : float
        Weighted error of predictions.
    sample_weight : Optional[np.ndarray], default=None
        Sample weights.

    Returns
    ------
    np.ndarray
        Updated sample weights.
    """
    if error == 0:
        return sample_weight if sample_weight is not None else np.ones_like(y_true)
    beta = error / (1.0 - error)
    if sample_weight is None:
        return np.where(y_pred != y_true, beta * np.ones_like(y_true), np.ones_like(y_true))
    return sample_weight * (beta ** (y_pred != y_true))

def adaboost_fit(
    X: np.ndarray,
    y: np.ndarray,
    base_estimator: Callable,
    n_estimators: int = 50,
    learning_rate: float = 1.0,
    sample_weight: Optional[np.ndarray] = None,
    metric: Callable = compute_error
) -> Dict[str, Any]:
    """
    Fit an AdaBoost classifier.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    base_estimator : Callable
        Base estimator to fit on random subsets of the data.
    n_estimators : int, default=50
        The number of estimators in the ensemble.
    learning_rate : float, default=1.0
        Learning rate shrinks the contribution of each estimator.
    sample_weight : Optional[np.ndarray], default=None
        Sample weights.
    metric : Callable, default=compute_error
        Metric to evaluate the performance of the ensemble.

    Returns
    ------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    validate_inputs(X, y, sample_weight)

    if sample_weight is None:
        sample_weight = np.ones_like(y) / len(y)
    else:
        sample_weight = sample_weight / np.sum(sample_weight)

    estimators = []
    estimator_weights = []

    for _ in range(n_estimators):
        # Fit base estimator
        estimator = base_estimator(X, y, sample_weight=sample_weight)
        estimators.append(estimator)

        # Compute predictions
        y_pred = estimator.predict(X)

        # Compute error and weight for the current estimator
        error = metric(y, y_pred, sample_weight)
        estimator_weight = learning_rate * np.log((1.0 - error) / max(error, 1e-10))
        estimator_weights.append(estimator_weight)

        # Update sample weights
        if error == 0:
            break
        beta = (1.0 - error) / max(error, 1e-10)
        sample_weight *= np.where(y_pred == y, beta, 1.0)

    # Compute final predictions
    def predict(X: np.ndarray) -> np.ndarray:
        pred = np.zeros_like(y)
        for estimator, weight in zip(estimators, estimator_weights):
            pred += weight * estimator.predict(X)
        return np.sign(pred)

    # Compute metrics
    y_pred = predict(X)
    error = metric(y, y_pred)

    return {
        "result": {"estimators": estimators, "estimator_weights": estimator_weights},
        "metrics": {"error": error},
        "params_used": {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate
        },
        "warnings": []
    }

# Example usage:
"""
from sklearn.tree import DecisionTreeClassifier

X = np.random.rand(100, 5)
y = np.random.randint(0, 2, size=100)

def base_estimator(X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> DecisionTreeClassifier:
    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(X, y, sample_weight=sample_weight)
    return clf

result = adaboost_fit(X, y, base_estimator)
"""

################################################################################
# xgboost
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
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

def _normalize_data(X: np.ndarray, normalization: str) -> np.ndarray:
    """Normalize data according to specified method."""
    if normalization == "none":
        return X
    elif normalization == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif normalization == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif normalization == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _compute_metric(y_true: np.ndarray, y_pred: np.ndarray,
                    metric: Union[str, Callable]) -> float:
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
        return 1 - (ss_res / (ss_tot + 1e-8))
    elif metric == "logloss":
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _xgboost_train(X: np.ndarray, y: np.ndarray,
                   n_estimators: int = 100,
                   learning_rate: float = 0.3,
                   max_depth: int = 6,
                   min_child_weight: float = 1.0,
                   gamma: float = 0.0,
                   subsample: float = 1.0,
                   colsample_bytree: float = 1.0,
                   objective: str = "binary:logistic",
                   eval_metric: Union[str, Callable] = "logloss",
                   normalization: str = "none") -> Dict:
    """Train XGBoost model with specified parameters."""
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _normalize_data(X, normalization)

    # Initialize model parameters
    params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "gamma": gamma,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "objective": objective
    }

    # Placeholder for actual XGBoost training logic
    # In a real implementation, this would call the XGBoost library
    y_pred = np.zeros_like(y)

    # Compute evaluation metric
    if callable(eval_metric):
        metric_value = eval_metric(y, y_pred)
    else:
        metric_value = _compute_metric(y, y_pred, eval_metric)

    return {
        "result": y_pred,
        "metrics": {eval_metric: metric_value},
        "params_used": params,
        "warnings": []
    }

def xgboost_fit(X: np.ndarray, y: np.ndarray,
                n_estimators: int = 100,
                learning_rate: float = 0.3,
                max_depth: int = 6,
                min_child_weight: float = 1.0,
                gamma: float = 0.0,
                subsample: float = 1.0,
                colsample_bytree: float = 1.0,
                objective: str = "binary:logistic",
                eval_metric: Union[str, Callable] = "logloss",
                normalization: str = "none") -> Dict:
    """
    Train an XGBoost model with specified parameters.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    n_estimators : int, optional
        Number of boosting rounds (default: 100)
    learning_rate : float, optional
        Step size shrinkage (default: 0.3)
    max_depth : int, optional
        Maximum tree depth (default: 6)
    min_child_weight : float, optional
        Minimum sum of instance weight needed in a child (default: 1.0)
    gamma : float, optional
        Minimum loss reduction required to make a split (default: 0.0)
    subsample : float, optional
        Subsample ratio of training instances (default: 1.0)
    colsample_bytree : float, optional
        Subsample ratio of features (default: 1.0)
    objective : str, optional
        Objective function (default: "binary:logistic")
    eval_metric : str or callable, optional
        Evaluation metric (default: "logloss")
    normalization : str, optional
        Data normalization method (default: "none")

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = xgboost_fit(X, y)
    """
    return _xgboost_train(
        X=X,
        y=y,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective=objective,
        eval_metric=eval_metric,
        normalization=normalization
    )

################################################################################
# lightgbm
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def lightgbm_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'logloss',
    objective: str = 'binary',
    learning_rate: float = 0.1,
    num_boost_round: int = 100,
    early_stopping_rounds: Optional[int] = None,
    verbose: int = 0
) -> Dict[str, Any]:
    """
    Fit a LightGBM model to the given data.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the features. Default is identity.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Metric to evaluate the model. Can be a string ('mse', 'mae', 'r2', 'logloss') or a custom callable.
    objective : str
        Objective function ('binary', 'multiclass', 'regression').
    learning_rate : float
        Learning rate for the boosting process.
    num_boost_round : int
        Number of boosting rounds.
    early_stopping_rounds : Optional[int]
        Number of rounds to wait before stopping if no improvement.
    verbose : int
        Verbosity level.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features
    X_normalized = normalizer(X)

    # Initialize model parameters
    params_used = {
        'normalization': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
        'metric': metric,
        'objective': objective,
        'learning_rate': learning_rate,
        'num_boost_round': num_boost_round,
        'early_stopping_rounds': early_stopping_rounds
    }

    # Initialize metrics
    metrics = {}

    # Placeholder for the actual LightGBM implementation
    # In a real scenario, this would call the LightGBM library or implement the algorithm
    result = _lightgbm_train(
        X_normalized, y,
        objective=objective,
        learning_rate=learning_rate,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds
    )

    # Calculate metrics
    if isinstance(metric, str):
        metrics['metric'] = _calculate_metric(y, result.predictions, metric)
    else:
        metrics['metric'] = metric(y, result.predictions)

    # Return results
    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate the input data.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.

    Raises:
    -------
    ValueError
        If inputs are invalid.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains NaN or infinite values.")
    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("y contains NaN or infinite values.")

def _lightgbm_train(
    X: np.ndarray,
    y: np.ndarray,
    objective: str,
    learning_rate: float,
    num_boost_round: int,
    early_stopping_rounds: Optional[int]
) -> Dict[str, Any]:
    """
    Train a LightGBM model.

    Parameters:
    -----------
    X : np.ndarray
        Normalized feature matrix.
    y : np.ndarray
        Target vector.
    objective : str
        Objective function.
    learning_rate : float
        Learning rate.
    num_boost_round : int
        Number of boosting rounds.
    early_stopping_rounds : Optional[int]
        Early stopping rounds.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the trained model and predictions.
    """
    # Placeholder for actual training logic
    return {
        'model': None,  # In a real scenario, this would be the trained model
        'predictions': np.random.rand(X.shape[0])  # Placeholder predictions
    }

def _calculate_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str
) -> float:
    """
    Calculate the specified metric.

    Parameters:
    -----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values.
    metric : str
        Metric to calculate ('mse', 'mae', 'r2', 'logloss').

    Returns:
    --------
    float
        Calculated metric value.

    Raises:
    -------
    ValueError
        If the specified metric is not supported.
    """
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
        raise ValueError(f"Unsupported metric: {metric}")

################################################################################
# catboost
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> None:
    """Validate input data."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if sample_weight is not None:
        if X.shape[0] != sample_weight.shape[0]:
            raise ValueError("X and sample_weight must have the same number of samples")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative values")

def _calculate_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    sample_weight: Optional[np.ndarray] = None
) -> float:
    """Calculate the specified metric."""
    if callable(metric):
        return metric(y_true, y_pred, sample_weight)
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

def _catboost_train(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    depth: int = 6,
    l2_leaf_reg: float = 3.0,
    metric: Union[str, Callable] = "logloss",
    sample_weight: Optional[np.ndarray] = None,
    random_state: Optional[int] = None
) -> Dict:
    """Train a CatBoost model."""
    _validate_inputs(X, y, sample_weight)

    # Initialize random number generator
    rng = np.random.RandomState(random_state)

    # Initialize model parameters
    params_used = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "depth": depth,
        "l2_leaf_reg": l2_leaf_reg,
        "metric": metric,
        "random_state": random_state
    }

    # Initialize predictions with mean value
    y_pred = np.mean(y) if sample_weight is None else np.average(y, weights=sample_weight)
    y_pred = np.full_like(y, y_pred)

    # Initialize ensemble
    ensemble = []

    for _ in range(n_estimators):
        # Calculate residuals
        residuals = y - y_pred

        if sample_weight is not None:
            residuals *= sample_weight
            sample_weights = sample_weight.copy()
        else:
            sample_weights = np.ones_like(y)

        # Train a weak learner (decision tree)
        tree_pred = _train_weak_learner(X, residuals, depth, l2_leaf_reg, sample_weights, rng)

        # Update predictions
        y_pred += learning_rate * tree_pred

        # Store the weak learner
        ensemble.append(tree_pred)

    # Calculate final metrics
    metrics = {
        "metric": _calculate_metric(y, y_pred, metric, sample_weight)
    }

    return {
        "result": y_pred,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": []
    }

def _train_weak_learner(
    X: np.ndarray,
    y: np.ndarray,
    depth: int,
    l2_leaf_reg: float,
    sample_weight: np.ndarray,
    rng: np.random.RandomState
) -> np.ndarray:
    """Train a single weak learner (decision tree)."""
    # This is a simplified version of a decision tree
    # In a real implementation, you would use a proper decision tree algorithm

    # Initialize leaf values
    leaf_values = np.zeros(X.shape[0])

    # Simple random forest approach for demonstration
    n_samples, n_features = X.shape

    # Randomly select features and split points
    selected_features = rng.choice(n_features, size=depth * 2, replace=False)
    split_points = np.percentile(X[:, selected_features], rng.uniform(0, 100, size=depth * 2))

    # Create splits
    for i in range(depth):
        feature = selected_features[i * 2]
        split_point = split_points[i * 2]

        left_mask = X[:, feature] < split_point
        right_mask = ~left_mask

        # Calculate weighted mean for left and right nodes
        if np.any(left_mask):
            leaf_values[left_mask] = np.average(y[left_mask], weights=sample_weight[left_mask])
        if np.any(right_mask):
            leaf_values[right_mask] = np.average(y[right_mask], weights=sample_weight[right_mask])

    return leaf_values

def catboost_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    depth: int = 6,
    l2_leaf_reg: float = 3.0,
    metric: Union[str, Callable] = "logloss",
    sample_weight: Optional[np.ndarray] = None,
    random_state: Optional[int] = None
) -> Dict:
    """
    Train a CatBoost model.

    Parameters:
    -----------
    X : np.ndarray
        Training data of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    n_estimators : int, optional
        Number of boosting iterations (default: 100)
    learning_rate : float, optional
        Learning rate (default: 0.1)
    depth : int, optional
        Depth of the decision trees (default: 6)
    l2_leaf_reg : float, optional
        L2 regularization coefficient (default: 3.0)
    metric : str or callable, optional
        Metric to evaluate the quality of a split (default: "logloss")
    sample_weight : np.ndarray, optional
        Sample weights (default: None)
    random_state : int, optional
        Random seed for reproducibility (default: None)

    Returns:
    --------
    dict
        Dictionary containing the results, metrics, parameters used, and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = catboost_fit(X, y)
    """
    return _catboost_train(
        X, y, n_estimators, learning_rate, depth,
        l2_leaf_reg, metric, sample_weight, random_state
    )

################################################################################
# stacking
################################################################################

import numpy as np
from typing import Callable, Dict, List, Optional, Union

def stacking_fit(
    base_models: List[Callable],
    meta_model: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    normalize: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict:
    """
    Stacking ensemble method for classification/regression.

    Parameters:
    -----------
    base_models : List[Callable]
        List of callable base models (e.g., classifiers/regressors)
    meta_model : Callable
        Callable meta model (e.g., classifier/regressor)
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training targets
    X_val : Optional[np.ndarray]
        Validation features (optional)
    y_val : Optional[np.ndarray]
        Validation targets (optional)
    normalize : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : Union[str, Callable]
        Evaluation metric ('mse', 'mae', 'r2', 'logloss') or custom callable
    solver : str
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    regularization : Optional[str]
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    random_state : Optional[int]
        Random seed for reproducibility

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> base_models = [LogisticRegression() for _ in range(5)]
    >>> meta_model = LogisticRegression()
    >>> result = stacking_fit(base_models, meta_model, X_train, y_train)
    """
    # Input validation
    _validate_inputs(X_train, y_train, X_val, y_val)

    # Normalize data
    X_train_norm, X_val_norm = _normalize_data(X_train, X_val, method=normalize)

    # Generate meta-features
    meta_features = _generate_meta_features(base_models, X_train_norm, y_train)

    # Train base models
    _train_base_models(base_models, X_train_norm, y_train)

    # Train meta model
    meta_model = _train_meta_model(
        meta_model,
        meta_features,
        y_train,
        metric=metric,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter
    )

    # Evaluate if validation data provided
    metrics = {}
    warnings = []
    if X_val is not None and y_val is not None:
        val_preds = _predict_stacking(base_models, meta_model, X_val_norm)
        metrics = _compute_metrics(y_val, val_preds, metric)

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
        'warnings': warnings
    }

def _validate_inputs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None
) -> None:
    """Validate input dimensions and types."""
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train must have the same number of samples")
    if X_val is not None and (X_val.shape[0] != y_val.shape[0]):
        raise ValueError("X_val and y_val must have the same number of samples")
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        raise ValueError("X_train contains NaN or infinite values")
    if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
        raise ValueError("y_train contains NaN or infinite values")

def _normalize_data(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray],
    method: str = 'standard'
) -> tuple:
    """Normalize data using specified method."""
    if method == 'none':
        return X_train, X_val

    # Calculate normalization parameters
    if method == 'standard':
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
    elif method == 'minmax':
        min_val = np.min(X_train, axis=0)
        max_val = np.max(X_train, axis=0)
    elif method == 'robust':
        median = np.median(X_train, axis=0)
        iqr = np.subtract(*np.percentile(X_train, [75, 25], axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Normalize training data
    if method == 'standard':
        X_train_norm = (X_train - mean) / std
    elif method == 'minmax':
        X_train_norm = (X_train - min_val) / (max_val - min_val)
    elif method == 'robust':
        X_train_norm = (X_train - median) / iqr

    # Normalize validation data if provided
    X_val_norm = None
    if X_val is not None:
        if method == 'standard':
            X_val_norm = (X_val - mean) / std
        elif method == 'minmax':
            X_val_norm = (X_val - min_val) / (max_val - min_val)
        elif method == 'robust':
            X_val_norm = (X_val - median) / iqr

    return X_train_norm, X_val_norm

def _generate_meta_features(
    base_models: List[Callable],
    X_train: np.ndarray,
    y_train: np.ndarray
) -> np.ndarray:
    """Generate meta-features from base models."""
    meta_features = []
    for model in base_models:
        # Fit the base model
        model.fit(X_train, y_train)
        # Generate predictions (probabilities for classifiers)
        if hasattr(model, 'predict_proba'):
            preds = model.predict_proba(X_train)
        else:
            preds = model.predict(X_train).reshape(-1, 1)
        meta_features.append(preds)

    return np.hstack(meta_features)

def _train_base_models(
    base_models: List[Callable],
    X_train: np.ndarray,
    y_train: np.ndarray
) -> None:
    """Train all base models."""
    for model in base_models:
        model.fit(X_train, y_train)

def _train_meta_model(
    meta_model: Callable,
    X_meta: np.ndarray,
    y_train: np.ndarray,
    metric: Union[str, Callable] = 'mse',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> Callable:
    """Train the meta model with specified parameters."""
    # Set solver and regularization if supported
    if hasattr(meta_model, 'set_params'):
        params = {'solver': solver}
        if regularization is not None:
            params[f'penalty'] = regularization
        meta_model.set_params(**params)

    # Fit the meta model
    meta_model.fit(X_meta, y_train)
    return meta_model

def _predict_stacking(
    base_models: List[Callable],
    meta_model: Callable,
    X_test: np.ndarray
) -> np.ndarray:
    """Make predictions using stacking ensemble."""
    # Generate meta-features for test data
    meta_features = []
    for model in base_models:
        if hasattr(model, 'predict_proba'):
            preds = model.predict_proba(X_test)
        else:
            preds = model.predict(X_test).reshape(-1, 1)
        meta_features.append(preds)

    X_meta = np.hstack(meta_features)

    # Make final predictions
    return meta_model.predict(X_meta)

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable] = 'mse'
) -> Dict:
    """Compute evaluation metrics."""
    metrics = {}

    if metric == 'mse':
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)
    elif metric == 'logloss':
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        metrics['logloss'] = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    elif callable(metric):
        metrics['custom'] = metric(y_true, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

################################################################################
# voting_classifier
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def voting_classifier_fit(
    estimators: list,
    X_train: np.ndarray,
    y_train: np.ndarray,
    voting_type: str = 'hard',
    weights: Optional[list] = None,
    n_jobs: int = 1
) -> Dict[str, Any]:
    """
    Fit a voting classifier ensemble.

    Parameters:
    -----------
    estimators : list
        List of (name, estimator) tuples.
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Target values.
    voting_type : str, optional (default='hard')
        Type of voting: 'hard' or 'soft'.
    weights : list, optional
        List of weights for each estimator.
    n_jobs : int, optional (default=1)
        Number of parallel jobs.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the fitted model and metadata.
    """
    # Validate inputs
    _validate_inputs(estimators, X_train, y_train, voting_type, weights)

    # Fit estimators
    fitted_estimators = _fit_estimators(estimators, X_train, y_train, n_jobs)

    # Prepare output
    result = {
        'estimators': fitted_estimators,
        'voting_type': voting_type,
        'weights': weights
    }

    return result

def _validate_inputs(
    estimators: list,
    X_train: np.ndarray,
    y_train: np.ndarray,
    voting_type: str,
    weights: Optional[list]
) -> None:
    """
    Validate inputs for voting classifier.
    """
    if not isinstance(estimators, list) or len(estimators) == 0:
        raise ValueError("Estimators must be a non-empty list")

    if X_train.ndim != 2:
        raise ValueError("X_train must be a 2D array")

    if y_train.ndim != 1:
        raise ValueError("y_train must be a 1D array")

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same length")

    if voting_type not in ['hard', 'soft']:
        raise ValueError("voting_type must be either 'hard' or 'soft'")

    if weights is not None:
        if len(weights) != len(estimators):
            raise ValueError("Weights must have the same length as estimators")
        if not all(isinstance(w, (int, float)) for w in weights):
            raise ValueError("Weights must be numeric")

def _fit_estimators(
    estimators: list,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_jobs: int
) -> list:
    """
    Fit all estimators in parallel.
    """
    fitted = []
    for name, estimator in estimators:
        estimator.fit(X_train, y_train)
        fitted.append((name, estimator))
    return fitted

def voting_classifier_predict(
    model: Dict[str, Any],
    X_test: np.ndarray
) -> np.ndarray:
    """
    Predict using a voting classifier ensemble.

    Parameters:
    -----------
    model : Dict[str, Any]
        Fitted voting classifier model.
    X_test : np.ndarray
        Test data.

    Returns:
    --------
    np.ndarray
        Predicted class labels.
    """
    # Validate inputs
    if not isinstance(model, dict):
        raise ValueError("Model must be a dictionary")

    estimators = model.get('estimators', [])
    voting_type = model.get('voting_type', 'hard')
    weights = model.get('weights', None)

    # Get predictions from all estimators
    predictions = []
    for name, estimator in estimators:
        if voting_type == 'hard':
            pred = estimator.predict(X_test)
        else:  # soft
            pred = estimator.predict_proba(X_test)
        predictions.append(pred)

    # Combine predictions
    if voting_type == 'hard':
        combined = _combine_hard_votes(predictions, weights)
    else:
        combined = _combine_soft_votes(predictions, weights)

    return combined

def _combine_hard_votes(
    predictions: list,
    weights: Optional[list]
) -> np.ndarray:
    """
    Combine hard votes from estimators.
    """
    if weights is None:
        weights = [1] * len(predictions)

    weighted_votes = []
    for i, pred in enumerate(predictions):
        if len(pred.shape) == 1:
            # Convert to one-hot encoding
            classes = np.unique(pred)
            pred_onehot = np.zeros((len(pred), len(classes)))
            for j, cls in enumerate(classes):
                pred_onehot[pred == cls, j] = 1
        else:
            pred_onehot = pred

        weighted_votes.append(pred_onehot * weights[i])

    combined = np.sum(weighted_votes, axis=0)
    return np.argmax(combined, axis=1)

def _combine_soft_votes(
    predictions: list,
    weights: Optional[list]
) -> np.ndarray:
    """
    Combine soft votes from estimators.
    """
    if weights is None:
        weights = [1] * len(predictions)

    weighted_probs = []
    for i, pred in enumerate(predictions):
        weighted_probs.append(pred * weights[i])

    combined = np.sum(weighted_probs, axis=0)
    return np.argmax(combined, axis=1)

# Example usage:
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Create estimators
estimator1 = ('rf', RandomForestClassifier(n_estimators=10))
estimator2 = ('lr', LogisticRegression())

# Fit model
model = voting_classifier_fit(
    estimators=[estimator1, estimator2],
    X_train=np.random.rand(100, 5),
    y_train=np.random.randint(0, 2, size=100)
)

# Predict
predictions = voting_classifier_predict(model, np.random.rand(10, 5))
"""

################################################################################
# feature_importance
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def feature_importance_fit(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    n_estimators: int = 100,
    normalizer: Optional[Callable] = None,
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute feature importance using ensemble classification methods.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    model : Callable
        Ensemble classification model with fit and predict methods.
    n_estimators : int, optional
        Number of estimators in the ensemble (default: 100).
    normalizer : Callable, optional
        Feature normalization function (default: None).
    metric : str or Callable, optional
        Metric to evaluate feature importance (default: 'mse').
    distance : str, optional
        Distance metric for feature importance calculation (default: 'euclidean').
    solver : str, optional
        Solver method for optimization (default: 'closed_form').
    regularization : str, optional
        Regularization type (none, l1, l2, elasticnet) (default: None).
    tol : float, optional
        Tolerance for convergence (default: 1e-4).
    max_iter : int, optional
        Maximum number of iterations (default: 1000).
    random_state : int, optional
        Random seed for reproducibility (default: None).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Fit the ensemble model
    model.fit(X_normalized, y)

    # Compute feature importance
    feature_importance = _compute_feature_importance(
        model, X_normalized, y,
        metric=metric,
        distance=distance,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter
    )

    # Calculate metrics
    metrics = _calculate_metrics(model, X_normalized, y, metric)

    # Prepare output
    result = {
        'result': feature_importance,
        'metrics': metrics,
        'params_used': {
            'n_estimators': n_estimators,
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

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply feature normalization."""
    if normalizer is None:
        return X
    return normalizer(X)

def _compute_feature_importance(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    distance: str,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Compute feature importance using the specified method."""
    # Placeholder for actual implementation
    # This would involve permutation importance, SHAP values, etc.
    return np.zeros(X.shape[1])

def _calculate_metrics(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    y_pred = model.predict(X)
    if callable(metric):
        return {metric.__name__: metric(y, y_pred)}
    elif metric == 'mse':
        return {'mse': np.mean((y - y_pred) ** 2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(y - y_pred))}
    elif metric == 'r2':
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return {'r2': 1 - (ss_res / ss_tot)}
    else:
        raise ValueError(f"Unknown metric: {metric}")

################################################################################
# oob_score
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def oob_score_fit(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    metric: Union[str, Callable] = 'accuracy',
    normalize: str = 'none',
    random_state: Optional[int] = None,
    sample_weight: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute the out-of-bag (OOB) score for an ensemble classifier.

    Parameters:
    -----------
    estimator : Any
        The ensemble classifier (e.g., RandomForestClassifier).
    X : np.ndarray
        Training data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    n_estimators : int, optional
        Number of estimators in the ensemble (default=100).
    metric : str or callable, optional
        Metric to evaluate the OOB score. Options: 'accuracy', 'logloss',
        or a custom callable (default='accuracy').
    normalize : str, optional
        Normalization method for features: 'none', 'standard', 'minmax',
        or 'robust' (default='none').
    random_state : int, optional
        Random seed for reproducibility (default=None).
    sample_weight : np.ndarray, optional
        Sample weights of shape (n_samples,) (default=None).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - 'result': OOB score.
        - 'metrics': Additional metrics if applicable.
        - 'params_used': Parameters used in the computation.
        - 'warnings': Any warnings encountered.

    Example:
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, 100)
    >>> oob_score_fit(RandomForestClassifier(), X, y)
    """
    # Validate inputs
    _validate_inputs(X, y, sample_weight)

    # Normalize features if required
    X_normalized = _normalize_features(X, normalize)

    # Set random state for reproducibility
    if random_state is not None:
        np.random.seed(random_state)

    # Initialize warnings and metrics
    warnings = []
    metrics = {}

    # Check if the estimator has oob_score attribute
    if not hasattr(estimator, 'oob_score_'):
        warnings.append("Estimator does not support OOB score computation.")
        return {
            'result': None,
            'metrics': metrics,
            'params_used': {
                'n_estimators': n_estimators,
                'metric': metric,
                'normalize': normalize
            },
            'warnings': warnings
        }

    # Fit the estimator and compute OOB score
    estimator.fit(X_normalized, y, sample_weight=sample_weight)

    # Compute OOB score based on the chosen metric
    oob_score = _compute_oob_metric(estimator, y, metric)

    return {
        'result': oob_score,
        'metrics': metrics,
        'params_used': {
            'n_estimators': n_estimators,
            'metric': metric,
            'normalize': normalize
        },
        'warnings': warnings
    }

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray]
) -> None:
    """
    Validate input data dimensions and types.

    Parameters:
    -----------
    X : np.ndarray
        Training data.
    y : np.ndarray
        Target values.
    sample_weight : Optional[np.ndarray]
        Sample weights.

    Raises:
    -------
    ValueError
        If inputs are invalid.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples.")
    if sample_weight is not None:
        if len(sample_weight) != len(y):
            raise ValueError("sample_weight must have the same length as y.")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight must contain non-negative values.")

def _normalize_features(
    X: np.ndarray,
    method: str
) -> np.ndarray:
    """
    Normalize features based on the specified method.

    Parameters:
    -----------
    X : np.ndarray
        Training data.
    method : str
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.

    Returns:
    --------
    np.ndarray
        Normalized features.
    """
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

def _compute_oob_metric(
    estimator: Any,
    y_true: np.ndarray,
    metric: Union[str, Callable]
) -> float:
    """
    Compute the OOB score based on the specified metric.

    Parameters:
    -----------
    estimator : Any
        The fitted ensemble classifier.
    y_true : np.ndarray
        True target values.
    metric : str or callable
        Metric to evaluate the OOB score.

    Returns:
    --------
    float
        The computed OOB score.
    """
    if callable(metric):
        return metric(estimator, y_true)
    elif metric == 'accuracy':
        return estimator.oob_score_
    elif metric == 'logloss':
        # Example for log loss, adjust as needed
        y_proba = estimator.predict_proba(y_true)
        return -np.mean(np.sum(y_true * np.log(y_proba + 1e-8), axis=1))
    else:
        raise ValueError(f"Unknown metric: {metric}")

################################################################################
# learning_rate
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def learning_rate_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = "mse",
    distance: str = "euclidean",
    solver: str = "gradient_descent",
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
    regularization: Optional[str] = None,
    l1_ratio: float = 0.5,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None,
    verbose: bool = False
) -> Dict:
    """
    Compute the optimal learning rate for ensemble classification models.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    normalizer : Optional[Callable]
        Function to normalize the input features.
    metric : Union[str, Callable]
        Metric to optimize. Can be "mse", "mae", "r2", "logloss" or a custom callable.
    distance : str
        Distance metric for ensemble methods. Can be "euclidean", "manhattan", "cosine", "minkowski".
    solver : str
        Solver to use. Can be "gradient_descent", "newton", or "coordinate_descent".
    learning_rate : float
        Initial learning rate.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for stopping criterion.
    regularization : Optional[str]
        Regularization type. Can be "l1", "l2", or "elasticnet".
    l1_ratio : float
        Elastic net mixing parameter.
    custom_metric : Optional[Callable]
        Custom metric function if not using built-in metrics.
    custom_distance : Optional[Callable]
        Custom distance function if not using built-in distances.
    verbose : bool
        Whether to print progress information.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if a normalizer is provided
    X_normalized = _apply_normalization(X, normalizer)

    # Initialize parameters
    params = {
        "learning_rate": learning_rate,
        "max_iter": max_iter,
        "tol": tol,
        "solver": solver,
        "distance": distance,
        "regularization": regularization,
        "l1_ratio": l1_ratio
    }

    # Choose metric function
    metric_func = _get_metric_function(metric, custom_metric)

    # Choose distance function
    distance_func = _get_distance_function(distance, custom_distance)

    # Initialize model parameters
    weights = np.zeros(X_normalized.shape[1])

    # Training loop
    for i in range(max_iter):
        gradients = _compute_gradients(X_normalized, y, weights, distance_func)

        if solver == "gradient_descent":
            weights -= learning_rate * gradients
        elif solver == "newton":
            hessian = _compute_hessian(X_normalized, weights)
            weights -= np.linalg.solve(hessian, gradients)
        elif solver == "coordinate_descent":
            weights = _coordinate_descent_step(X_normalized, y, weights)

        # Apply regularization
        if regularization == "l1":
            weights -= learning_rate * np.sign(weights)
        elif regularization == "l2":
            weights -= learning_rate * 2 * weights
        elif regularization == "elasticnet":
            weights -= learning_rate * (l1_ratio * np.sign(weights) + (1 - l1_ratio) * 2 * weights)

        # Check convergence
        if _check_convergence(weights, tol):
            break

    # Compute final metrics
    predictions = _predict(X_normalized, weights)
    metrics = {
        "metric_value": metric_func(y, predictions),
        "final_loss": _compute_loss(y, predictions)
    }

    # Prepare results
    result = {
        "result": weights,
        "metrics": metrics,
        "params_used": params,
        "warnings": []
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
    """Apply normalization to input features."""
    if normalizer is None:
        return X
    return normalizer(X)

def _get_metric_function(metric: Union[str, Callable], custom_metric: Optional[Callable]) -> Callable:
    """Get the appropriate metric function."""
    if custom_metric is not None:
        return custom_metric
    metrics = {
        "mse": _mean_squared_error,
        "mae": _mean_absolute_error,
        "r2": _r_squared,
        "logloss": _log_loss
    }
    return metrics.get(metric, _mean_squared_error)

def _get_distance_function(distance: str, custom_distance: Optional[Callable]) -> Callable:
    """Get the appropriate distance function."""
    if custom_distance is not None:
        return custom_distance
    distances = {
        "euclidean": _euclidean_distance,
        "manhattan": _manhattan_distance,
        "cosine": _cosine_distance,
        "minkowski": _minkowski_distance
    }
    return distances.get(distance, _euclidean_distance)

def _compute_gradients(X: np.ndarray, y: np.ndarray, weights: np.ndarray, distance_func: Callable) -> np.ndarray:
    """Compute gradients for the given model."""
    predictions = _predict(X, weights)
    residuals = y - predictions
    return -2 * np.dot(X.T, residuals) / X.shape[0]

def _compute_hessian(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute the Hessian matrix."""
    return 2 * np.dot(X.T, X) / X.shape[0]

def _coordinate_descent_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Perform a coordinate descent step."""
    for i in range(X.shape[1]):
        X_i = X[:, i]
        residuals = y - np.dot(X, weights) + weights[i] * X_i
        numerator = np.dot(X_i, residuals)
        denominator = np.dot(X_i, X_i) + 1e-6
        weights[i] = numerator / denominator
    return weights

def _check_convergence(weights: np.ndarray, tol: float) -> bool:
    """Check if the model has converged."""
    return np.linalg.norm(weights) < tol

def _predict(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Make predictions using the current weights."""
    return np.dot(X, weights)

def _compute_loss(y: np.ndarray, predictions: np.ndarray) -> float:
    """Compute the loss between true and predicted values."""
    return np.mean((y - predictions) ** 2)

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
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _euclidean_distance(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute Euclidean distance."""
    if Y is None:
        Y = X
    return np.sqrt(np.sum((X[:, np.newaxis] - Y) ** 2, axis=2))

def _manhattan_distance(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute Manhattan distance."""
    if Y is None:
        Y = X
    return np.sum(np.abs(X[:, np.newaxis] - Y), axis=2)

def _cosine_distance(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute cosine distance."""
    if Y is None:
        Y = X
    dot_products = np.dot(X, Y.T)
    norms = np.sqrt(np.sum(X ** 2, axis=1))[:, np.newaxis] * np.sqrt(np.sum(Y ** 2, axis=1))
    return 1 - dot_products / norms

def _minkowski_distance(X: np.ndarray, Y: Optional[np.ndarray] = None, p: int = 3) -> np.ndarray:
    """Compute Minkowski distance."""
    if Y is None:
        Y = X
    return np.sum(np.abs(X[:, np.newaxis] - Y) ** p, axis=2) ** (1 / p)

################################################################################
# n_estimators
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for n_estimators."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """Normalize input data using specified method."""
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

def compute_metric(y_true: np.ndarray, y_pred: np.ndarray,
                   metric: Union[str, Callable]) -> float:
    """Compute specified metric between true and predicted values."""
    if callable(metric):
        return metric(y_true, y_pred)
    elif metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    elif metric == 'logloss':
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def n_estimators_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators: int = 100,
    base_estimator: Optional[Callable] = None,
    metric: Union[str, Callable] = 'mse',
    normalization: str = 'standard',
    random_state: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Fit an ensemble of classifiers and determine optimal number of estimators.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    n_estimators : int, optional
        Maximum number of estimators to consider (default: 100)
    base_estimator : Callable, optional
        Base estimator to use for the ensemble (default: None)
    metric : str or Callable, optional
        Metric to evaluate performance (default: 'mse')
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard')
    random_state : int, optional
        Random seed for reproducibility (default: None)

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_norm = normalize_data(X, method=normalization)

    # Initialize results dictionary
    result = {
        'result': None,
        'metrics': [],
        'params_used': {
            'n_estimators': n_estimators,
            'base_estimator': base_estimator.__name__ if base_estimator else None,
            'metric': metric if not callable(metric) else 'custom',
            'normalization': normalization,
            'random_state': random_state
        },
        'warnings': []
    }

    # If no base estimator provided, use a simple decision stump
    if base_estimator is None:
        def default_base_estimator(X, y):
            # Simple decision stump implementation
            best_feature = np.argmax(np.var(X, axis=0))
            threshold = np.median(X[:, best_feature])
            pred = (X[:, best_feature] > threshold).astype(int)
            return pred

        base_estimator = default_base_estimator

    # Initialize ensemble
    ensemble_predictions = np.zeros((n_estimators, len(y)))

    # Train each estimator and track performance
    for i in range(n_estimators):
        if random_state is not None:
            np.random.seed(random_state + i)

        # Train base estimator
        pred = base_estimator(X_norm, y)

        # Store predictions
        ensemble_predictions[i] = pred

        # Compute and store metric
        current_metric = compute_metric(y, pred, metric)
        result['metrics'].append(current_metric)

    # Determine optimal number of estimators
    best_n = np.argmin(result['metrics']) + 1 if metric in ['mse', 'mae'] else np.argmax(result['metrics']) + 1
    result['result'] = {
        'optimal_n_estimators': best_n,
        'final_metric_value': result['metrics'][best_n - 1]
    }

    return result

################################################################################
# max_depth
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int = 5,
    min_samples_split: int = 2,
    metric: Union[str, Callable] = 'gini',
    distance: str = 'euclidean'
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if max_depth <= 0:
        raise ValueError("max_depth must be positive")
    if min_samples_split <= 1:
        raise ValueError("min_samples_split must be at least 2")
    if isinstance(metric, str) and metric not in ['gini', 'entropy']:
        raise ValueError("metric must be 'gini' or 'entropy'")
    if isinstance(distance, str) and distance not in ['euclidean', 'manhattan', 'cosine']:
        raise ValueError("distance must be 'euclidean', 'manhattan' or 'cosine'")

def _compute_split(
    X: np.ndarray,
    y: np.ndarray,
    feature_idx: int,
    threshold: float,
    metric: Union[str, Callable]
) -> float:
    """Compute the split quality for a given feature and threshold."""
    left_mask = X[:, feature_idx] <= threshold
    right_mask = ~left_mask

    n_left, n_right = np.sum(left_mask), np.sum(right_mask)
    if n_left == 0 or n_right == 0:
        return float('inf')

    y_left, y_right = y[left_mask], y[right_mask]
    n_classes = len(np.unique(y))

    if callable(metric):
        return metric(y_left, y_right)
    elif metric == 'gini':
        def gini_impurity(y_sub):
            _, counts = np.unique(y_sub, return_counts=True)
            probs = counts / len(y_sub)
            return 1 - np.sum(probs ** 2)

        impurity_left = gini_impurity(y_left)
        impurity_right = gini_impurity(y_right)
        return (n_left * impurity_left + n_right * impurity_right) / len(y)
    elif metric == 'entropy':
        def entropy(y_sub):
            _, counts = np.unique(y_sub, return_counts=True)
            probs = counts / len(y_sub)
            return -np.sum([p * np.log2(p) for p in probs if p > 0])

        entropy_left = entropy(y_left)
        entropy_right = entropy(y_right)
        return (n_left * entropy_left + n_right * entropy_right) / len(y)

def _find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable]
) -> Dict[str, Any]:
    """Find the best split for a node."""
    best_split = {'feature_idx': None, 'threshold': None, 'value': float('inf')}

    for feature_idx in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_idx])
        for threshold in thresholds:
            value = _compute_split(X, y, feature_idx, threshold, metric)
            if value < best_split['value']:
                best_split.update({
                    'feature_idx': feature_idx,
                    'threshold': threshold,
                    'value': value
                })

    return best_split

def _build_tree(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int,
    min_samples_split: int,
    current_depth: int = 0,
    metric: Union[str, Callable] = 'gini',
    distance: str = 'euclidean'
) -> Dict[str, Any]:
    """Recursively build a decision tree."""
    n_samples = len(y)
    if (current_depth >= max_depth or
        n_samples < min_samples_split or
        len(np.unique(y)) == 1):
        return {'leaf': True, 'value': np.bincount(y).argmax()}

    best_split = _find_best_split(X, y, metric)
    if best_split['feature_idx'] is None:
        return {'leaf': True, 'value': np.bincount(y).argmax()}

    left_mask = X[:, best_split['feature_idx']] <= best_split['threshold']
    right_mask = ~left_mask

    left_tree = _build_tree(
        X[left_mask],
        y[left_mask],
        max_depth,
        min_samples_split,
        current_depth + 1,
        metric,
        distance
    )

    right_tree = _build_tree(
        X[right_mask],
        y[right_mask],
        max_depth,
        min_samples_split,
        current_depth + 1,
        metric,
        distance
    )

    return {
        'leaf': False,
        'feature_idx': best_split['feature_idx'],
        'threshold': best_split['threshold'],
        'left': left_tree,
        'right': right_tree
    }

def max_depth_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_depth: int = 5,
    min_samples_split: int = 2,
    metric: Union[str, Callable] = 'gini',
    distance: str = 'euclidean'
) -> Dict[str, Any]:
    """
    Fit a decision tree classifier with maximum depth constraint.

    Parameters:
    -----------
    X : np.ndarray
        Training data of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    max_depth : int, optional
        Maximum depth of the tree (default=5)
    min_samples_split : int, optional
        Minimum number of samples required to split a node (default=2)
    metric : str or callable, optional
        Metric to measure split quality ('gini', 'entropy' or custom callable)
    distance : str, optional
        Distance metric for feature computation ('euclidean', 'manhattan', 'cosine')

    Returns:
    --------
    dict
        Dictionary containing the fitted tree and related information

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = max_depth_fit(X, y, max_depth=3, metric='entropy')
    """
    _validate_inputs(X, y, max_depth, min_samples_split, metric, distance)

    tree = _build_tree(X, y, max_depth, min_samples_split, 0, metric, distance)

    return {
        'result': tree,
        'metrics': {'max_depth_used': max_depth},
        'params_used': {
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'metric': metric if not callable(metric) else 'custom',
            'distance': distance
        },
        'warnings': []
    }

################################################################################
# min_samples_split
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def min_samples_split_fit(
    X: np.ndarray,
    y: np.ndarray,
    criterion: str = "gini",
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: Optional[Union[int, float]] = None,
    random_state: Optional[int] = None,
    metric: Callable[[np.ndarray, np.ndarray], float] = None
) -> Dict[str, Any]:
    """
    Fit a decision tree classifier with configurable parameters.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    criterion : str, optional
        The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain.
    max_depth : int, optional
        The maximum depth of the tree. If None, nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    min_samples_split : int, optional
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, optional
        The minimum number of samples required to be at a leaf node.
    max_features : int or float, optional
        The number of features to consider when looking for the best split. If None, then max_features=X.shape[1].
    random_state : int, optional
        Controls the randomness of the estimator.
    metric : Callable[[np.ndarray, np.ndarray], float], optional
        Custom metric function to evaluate the quality of a split.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the fitted model parameters and results.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set random state for reproducibility
    rng = np.random.RandomState(random_state)

    # Initialize parameters
    params_used = {
        "criterion": criterion,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "random_state": random_state
    }

    # Fit the model
    tree = _fit_decision_tree(
        X, y,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=rng,
        metric=metric
    )

    # Calculate metrics
    metrics = _calculate_metrics(X, y, tree)

    return {
        "result": tree,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate the input features and target values.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).

    Raises:
    -------
    ValueError
        If inputs are invalid.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")

def _fit_decision_tree(
    X: np.ndarray,
    y: np.ndarray,
    criterion: str,
    max_depth: Optional[int],
    min_samples_split: int,
    min_samples_leaf: int,
    max_features: Optional[Union[int, float]],
    random_state: np.random.RandomState,
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, Any]:
    """
    Fit a decision tree recursively.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    criterion : str
        The function to measure the quality of a split.
    max_depth : int, optional
        The maximum depth of the tree.
    min_samples_split : int
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int
        The minimum number of samples required to be at a leaf node.
    max_features : int or float, optional
        The number of features to consider when looking for the best split.
    random_state : np.random.RandomState
        Random state for reproducibility.
    metric : Callable[[np.ndarray, np.ndarray], float], optional
        Custom metric function to evaluate the quality of a split.

    Returns:
    --------
    Dict[str, Any]
        The fitted decision tree.
    """
    # Base case: stop splitting if conditions are met
    if (max_depth is not None and max_depth <= 0) or len(y) < min_samples_split:
        return _create_leaf_node(y)

    # Determine the best split
    best_split = _find_best_split(
        X, y,
        criterion=criterion,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        metric=metric
    )

    # If no split improves the criterion, create a leaf node
    if best_split is None:
        return _create_leaf_node(y)

    # Split the data
    left_indices, right_indices = best_split["indices"]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]

    # Recursively fit the left and right subtrees
    left_subtree = _fit_decision_tree(
        X_left, y_left,
        criterion=criterion,
        max_depth=max_depth - 1 if max_depth is not None else None,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        metric=metric
    )

    right_subtree = _fit_decision_tree(
        X_right, y_right,
        criterion=criterion,
        max_depth=max_depth - 1 if max_depth is not None else None,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        metric=metric
    )

    # Create the current node
    return {
        "feature": best_split["feature"],
        "threshold": best_split["threshold"],
        "left": left_subtree,
        "right": right_subtree
    }

def _find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    criterion: str,
    min_samples_leaf: int,
    max_features: Optional[Union[int, float]],
    random_state: np.random.RandomState,
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Optional[Dict[str, Any]]:
    """
    Find the best split for a node.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    criterion : str
        The function to measure the quality of a split.
    min_samples_leaf : int
        The minimum number of samples required to be at a leaf node.
    max_features : int or float, optional
        The number of features to consider when looking for the best split.
    random_state : np.random.RandomState
        Random state for reproducibility.
    metric : Callable[[np.ndarray, np.ndarray], float], optional
        Custom metric function to evaluate the quality of a split.

    Returns:
    --------
    Dict[str, Any] or None
        The best split information or None if no split improves the criterion.
    """
    n_samples, n_features = X.shape
    if max_features is None:
        max_features = n_features
    elif isinstance(max_features, float):
        max_features = int(max_features * n_features)

    # Randomly select features to consider
    feature_indices = random_state.permutation(n_features)[:max_features]

    best_split = None
    best_criterion_value = -np.inf

    for feature_idx in feature_indices:
        thresholds = np.unique(X[:, feature_idx])
        for threshold in thresholds:
            left_indices = X[:, feature_idx] <= threshold
            right_indices = ~left_indices

            # Ensure minimum samples in each leaf
            if (np.sum(left_indices) < min_samples_leaf) or (np.sum(right_indices) < min_samples_leaf):
                continue

            # Calculate the criterion value
            if metric is not None:
                left_metric = metric(y[left_indices], y[left_indices])
                right_metric = metric(y[right_indices], y[right_indices])
                criterion_value = left_metric + right_metric
            else:
                if criterion == "gini":
                    criterion_value = _calculate_gini_impurity(y[left_indices], y[right_indices])
                elif criterion == "entropy":
                    criterion_value = _calculate_entropy(y[left_indices], y[right_indices])
                else:
                    raise ValueError(f"Unknown criterion: {criterion}")

            # Update the best split if current split is better
            if criterion_value > best_criterion_value:
                best_criterion_value = criterion_value
                best_split = {
                    "feature": feature_idx,
                    "threshold": threshold,
                    "indices": (left_indices, right_indices)
                }

    return best_split

def _calculate_gini_impurity(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """
    Calculate the Gini impurity for a split.

    Parameters:
    -----------
    y_left : np.ndarray
        Target values of the left child.
    y_right : np.ndarray
        Target values of the right child.

    Returns:
    --------
    float
        The Gini impurity.
    """
    def gini(y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right

    return (n_left / n_total) * gini(y_left) + (n_right / n_total) * gini(y_right)

def _calculate_entropy(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """
    Calculate the entropy for a split.

    Parameters:
    -----------
    y_left : np.ndarray
        Target values of the left child.
    y_right : np.ndarray
        Target values of the right child.

    Returns:
    --------
    float
        The entropy.
    """
    def entropy(y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right

    return (n_left / n_total) * entropy(y_left) + (n_right / n_total) * entropy(y_right)

def _create_leaf_node(y: np.ndarray) -> Dict[str, Any]:
    """
    Create a leaf node.

    Parameters:
    -----------
    y : np.ndarray
        Target values of shape (n_samples,).

    Returns:
    --------
    Dict[str, Any]
        The leaf node.
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    most_common_class = unique_classes[np.argmax(counts)]

    return {
        "is_leaf": True,
        "class": most_common_class
    }

def _calculate_metrics(X: np.ndarray, y: np.ndarray, tree: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate metrics for the fitted decision tree.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    tree : Dict[str, Any]
        The fitted decision tree.

    Returns:
    --------
    Dict[str, float]
        A dictionary of metrics.
    """
    y_pred = _predict(tree, X)
    accuracy = np.mean(y_pred == y)

    return {
        "accuracy": accuracy
    }

def _predict(tree: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """
    Predict class labels for samples in X using the fitted decision tree.

    Parameters:
    -----------
    tree : Dict[str, Any]
        The fitted decision tree.
    X : np.ndarray
        Input features of shape (n_samples, n_features).

    Returns:
    --------
    np.ndarray
        Predicted class labels of shape (n_samples,).
    """
    y_pred = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        node = tree
        while not node.get("is_leaf", False):
            if X[i, node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        y_pred[i] = node["class"]

    return y_pred

################################################################################
# min_samples_leaf
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    min_samples_leaf: int = 1,
    random_state: Optional[int] = None
) -> None:
    """
    Validate input data and parameters.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    min_samples_leaf : int, optional
        Minimum number of samples required to be at a leaf node (default=1)
    random_state : int, optional
        Random seed for reproducibility

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
    if min_samples_leaf <= 0:
        raise ValueError("min_samples_leaf must be positive")
    if random_state is not None and (not isinstance(random_state, int) or random_state < 0):
        raise ValueError("random_state must be a non-negative integer")

def _compute_split_criterion(
    X_left: np.ndarray,
    y_left: np.ndarray,
    X_right: np.ndarray,
    y_right: np.ndarray,
    criterion: str = 'gini',
    metric_func: Optional[Callable] = None
) -> float:
    """
    Compute the splitting criterion for a potential split.

    Parameters
    ----------
    X_left, y_left : np.ndarray
        Left child data
    X_right, y_right : np.ndarray
        Right child data
    criterion : str, optional
        Splitting criterion ('gini', 'entropy') (default='gini')
    metric_func : callable, optional
        Custom metric function

    Returns
    ------
    float
        Computed criterion value
    """
    if metric_func is not None:
        return metric_func(X_left, y_left, X_right, y_right)

    if criterion == 'gini':
        def gini(y):
            _, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            return 1 - np.sum(probs ** 2)

        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right

        return (n_left / n_total) * gini(y_left) + (n_right / n_total) * gini(y_right)

    elif criterion == 'entropy':
        def entropy(y):
            _, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            return -np.sum([p * np.log2(p) for p in probs if p > 0])

        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right

        return (n_left / n_total) * entropy(y_left) + (n_right / n_total) * entropy(y_right)

    else:
        raise ValueError(f"Unknown criterion: {criterion}")

def _find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    feature_idx: int,
    threshold: float,
    min_samples_leaf: int
) -> Dict[str, Any]:
    """
    Find the best split for a given feature and threshold.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    feature_idx : int
        Index of the feature to split on
    threshold : float
        Threshold value for splitting
    min_samples_leaf : int
        Minimum number of samples required to be at a leaf node

    Returns
    ------
    dict
        Dictionary containing split information and criterion value
    """
    left_mask = X[:, feature_idx] <= threshold
    right_mask = ~left_mask

    n_left = np.sum(left_mask)
    n_right = np.sum(right_mask)

    if n_left < min_samples_leaf or n_right < min_samples_leaf:
        return {'criterion': float('inf')}

    X_left = X[left_mask]
    y_left = y[left_mask]
    X_right = X[right_mask]
    y_right = y[right_mask]

    criterion_value = _compute_split_criterion(X_left, y_left, X_right, y_right)

    return {
        'left_mask': left_mask,
        'right_mask': right_mask,
        'criterion': criterion_value
    }

def min_samples_leaf_fit(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int = 5,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    criterion: str = 'gini',
    metric_func: Optional[Callable] = None,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Fit a decision tree with min_samples_leaf constraint.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    max_depth : int, optional
        Maximum depth of the tree (default=5)
    min_samples_split : int, optional
        Minimum number of samples required to split a node (default=2)
    min_samples_leaf : int, optional
        Minimum number of samples required to be at a leaf node (default=1)
    criterion : str, optional
        Splitting criterion ('gini', 'entropy') (default='gini')
    metric_func : callable, optional
        Custom metric function for splitting
    random_state : int, optional
        Random seed for reproducibility (default=None)

    Returns
    ------
    dict
        Dictionary containing:
        - 'tree': the fitted tree structure
        - 'metrics': evaluation metrics
        - 'params_used': parameters used for fitting
        - 'warnings': any warnings generated during fitting

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = min_samples_leaf_fit(X, y, min_samples_leaf=5)
    """
    _validate_inputs(X, y, min_samples_leaf, random_state)

    if random_state is not None:
        np.random.seed(random_state)

    # Initialize tree structure
    tree = {
        'left': None,
        'right': None,
        'feature_idx': None,
        'threshold': None,
        'value': np.bincount(y).argmax() if len(y) > 0 else None,
        'is_leaf': True
    }

    def _build_tree(node, depth=0):
        if (depth >= max_depth or
            len(y) < min_samples_split or
            np.all(y == y[0])):
            return

        best_criterion = float('inf')
        best_split = None

        # Try all possible thresholds for each feature
        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                split_info = _find_best_split(
                    X, y, feature_idx, threshold, min_samples_leaf
                )
                if split_info['criterion'] < best_criterion:
                    best_criterion = split_info['criterion']
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'left_mask': split_info['left_mask'],
                        'right_mask': split_info['right_mask']
                    }

        if best_split is None or best_criterion == float('inf'):
            return

        node['is_leaf'] = False
        node['feature_idx'] = best_split['feature_idx']
        node['threshold'] = best_split['threshold']

        # Recursively build left and right subtrees
        node['left'] = {
            'left': None,
            'right': None,
            'feature_idx': None,
            'threshold': None,
            'value': np.bincount(y[best_split['left_mask']]).argmax(),
            'is_leaf': True
        }
        _build_tree(node['left'], depth + 1)

        node['right'] = {
            'left': None,
            'right': None,
            'feature_idx': None,
            'threshold': None,
            'value': np.bincount(y[best_split['right_mask']]).argmax(),
            'is_leaf': True
        }
        _build_tree(node['right'], depth + 1)

    _build_tree(tree)

    # Calculate metrics (accuracy in this case)
    def predict_sample(sample, node):
        if node['is_leaf']:
            return node['value']

        if sample[node['feature_idx']] <= node['threshold']:
            return predict_sample(sample, node['left'])
        else:
            return predict_sample(sample, node['right'])

    predictions = np.array([predict_sample(sample, tree) for sample in X])
    accuracy = np.mean(predictions == y)

    return {
        'tree': tree,
        'metrics': {'accuracy': accuracy},
        'params_used': {
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'criterion': criterion
        },
        'warnings': []
    }

################################################################################
# subsample
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    sample_size: int,
    random_state: Optional[int] = None
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if sample_size <= 0 or sample_size > X.shape[0]:
        raise ValueError("sample_size must be between 1 and number of samples")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("Input arrays must contain only finite values")

def apply_normalization(
    X: np.ndarray,
    y: np.ndarray,
    normalization: str = "standard"
) -> tuple[np.ndarray, np.ndarray]:
    """Apply specified normalization to input data."""
    if normalization == "none":
        return X, y
    elif normalization == "standard":
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1.0
        X_normalized = (X - X_mean) / X_std
        return X_normalized, y
    elif normalization == "minmax":
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1.0
        X_normalized = (X - X_min) / X_range
        return X_normalized, y
    elif normalization == "robust":
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_iqr[X_iqr == 0] = 1.0
        X_normalized = (X - X_median) / X_iqr
        return X_normalized, y
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "mse",
    custom_metric: Optional[Callable] = None
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
        return 1 - (ss_res / ss_tot)
    elif metric == "logloss":
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def subsample_fit(
    X: np.ndarray,
    y: np.ndarray,
    sample_size: int = 100,
    normalization: str = "standard",
    metric: str = "mse",
    custom_metric: Optional[Callable] = None,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform subsampling on the input data and compute metrics.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    sample_size : int, optional
        Number of samples to include in each subsample (default: 100)
    normalization : str, optional
        Normalization method to apply (default: "standard")
    metric : str, optional
        Metric to compute (default: "mse")
    custom_metric : Callable, optional
        Custom metric function to use instead of built-in metrics
    random_state : int, optional
        Random seed for reproducibility (default: None)

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings

    Example:
    --------
    >>> X = np.random.rand(1000, 5)
    >>> y = np.random.randint(0, 2, size=1000)
    >>> result = subsample_fit(X, y, sample_size=200, normalization="standard")
    """
    # Validate inputs
    validate_inputs(X, y, sample_size, random_state)

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Apply normalization
    X_normalized, y_normalized = apply_normalization(X, y, normalization)

    # Generate random subsample indices
    n_samples = X.shape[0]
    subsample_indices = np.random.choice(n_samples, size=sample_size, replace=False)

    # Get subsample data
    X_sub = X_normalized[subsample_indices]
    y_sub = y_normalized[subsample_indices]

    # Compute predictions (simple mean for demonstration)
    y_pred = np.mean(y_sub)

    # Compute metric
    try:
        score = compute_metric(y_sub, y_pred * np.ones_like(y_sub), metric, custom_metric)
    except Exception as e:
        return {
            "result": None,
            "metrics": None,
            "params_used": {
                "sample_size": sample_size,
                "normalization": normalization,
                "metric": metric
            },
            "warnings": [f"Error computing metric: {str(e)}"]
        }

    return {
        "result": {
            "subsample_indices": subsample_indices,
            "X_sub": X_sub,
            "y_sub": y_sub
        },
        "metrics": {
            "score": score,
            "metric_name": metric if custom_metric is None else "custom"
        },
        "params_used": {
            "sample_size": sample_size,
            "normalization": normalization,
            "metric": metric
        },
        "warnings": []
    }

################################################################################
# colsample_bytree
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def colsample_bytree_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    subsample_ratio: float = 1.0,
    colsample_bytree: float = 1.0,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "gini",
    max_depth: int = 3,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Fit a decision tree with column subsampling (colsample_bytree) for ensemble classification.

    Parameters:
    -----------
    X : np.ndarray
        Training data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    subsample_ratio : float, default=1.0
        Subsample ratio of the training instances.
    colsample_bytree : float, default=1.0
        Subsample ratio of the features for each tree.
    metric : str or callable, default="gini"
        The metric to evaluate the quality of a split. Can be "gini", "entropy", or a custom callable.
    max_depth : int, default=3
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    random_state : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the fitted model and related information.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set random state
    rng = np.random.RandomState(random_state)

    # Subsample the data if needed
    if subsample_ratio < 1.0:
        n_samples = int(X.shape[0] * subsample_ratio)
        indices = rng.choice(X.shape[0], n_samples, replace=False)
        X_subsampled = X[indices]
        y_subsampled = y[indices]
    else:
        X_subsampled, y_subsampled = X, y

    # Subsample the features if needed
    if colsample_bytree < 1.0:
        n_features = int(X.shape[1] * colsample_bytree)
        feature_indices = rng.choice(X.shape[1], n_features, replace=False)
        X_subsampled = X_subsampled[:, feature_indices]

    # Fit the decision tree
    tree = _fit_decision_tree(
        X_subsampled,
        y_subsampled,
        metric=metric,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )

    # Calculate metrics
    y_pred = _predict(tree, X_subsampled)
    metrics = _calculate_metrics(y_subsampled, y_pred)

    # Prepare the output
    result = {
        "result": tree,
        "metrics": metrics,
        "params_used": {
            "subsample_ratio": subsample_ratio,
            "colsample_bytree": colsample_bytree,
            "metric": metric,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state
        },
        "warnings": []
    }

    return result

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

def _fit_decision_tree(
    X: np.ndarray,
    y: np.ndarray,
    *,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "gini",
    max_depth: int = 3,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1
) -> Dict[str, Any]:
    """Fit a decision tree recursively."""
    # Base case: stop if max depth is reached or not enough samples
    if (max_depth == 0) or (X.shape[0] < min_samples_split):
        leaf_value = _calculate_leaf_value(y)
        return {"is_leaf": True, "value": leaf_value}

    # Find the best split
    best_split = _find_best_split(X, y, metric=metric, min_samples_leaf=min_samples_leaf)

    # If no split improves the metric, create a leaf node
    if best_split is None:
        leaf_value = _calculate_leaf_value(y)
        return {"is_leaf": True, "value": leaf_value}

    # Split the data
    left_indices = X[:, best_split["feature"]] <= best_split["threshold"]
    right_indices = ~left_indices

    # Recursively fit the left and right subtrees
    left_subtree = _fit_decision_tree(
        X[left_indices],
        y[left_indices],
        metric=metric,
        max_depth=max_depth - 1,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )
    right_subtree = _fit_decision_tree(
        X[right_indices],
        y[right_indices],
        metric=metric,
        max_depth=max_depth - 1,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )

    # Return the node with the split information and subtrees
    return {
        "is_leaf": False,
        "feature": best_split["feature"],
        "threshold": best_split["threshold"],
        "left": left_subtree,
        "right": right_subtree
    }

def _find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "gini",
    min_samples_leaf: int = 1
) -> Optional[Dict[str, Any]]:
    """Find the best split for a node."""
    best_split = None
    best_metric_value = -np.inf

    # Iterate over each feature
    for feature in range(X.shape[1]):
        # Get unique values of the feature
        thresholds = np.unique(X[:, feature])

        # Iterate over each unique value as a potential threshold
        for threshold in thresholds:
            left_indices = X[:, feature] <= threshold
            right_indices = ~left_indices

            # Ensure minimum samples in each leaf
            if (np.sum(left_indices) < min_samples_leaf) or (np.sum(right_indices) < min_samples_leaf):
                continue

            # Calculate the metric value for this split
            left_metric = _calculate_metric(y[left_indices], y[left_indices], metric=metric)
            right_metric = _calculate_metric(y[right_indices], y[right_indices], metric=metric)
            weighted_metric = (np.sum(left_indices) * left_metric + np.sum(right_indices) * right_metric) / len(y)

            # Update the best split if this one is better
            if weighted_metric > best_metric_value:
                best_metric_value = weighted_metric
                best_split = {
                    "feature": feature,
                    "threshold": threshold,
                    "left_metric": left_metric,
                    "right_metric": right_metric
                }

    return best_split

def _calculate_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "gini"
) -> float:
    """Calculate the metric value for a split."""
    if callable(metric):
        return metric(y_true, y_pred)
    elif metric == "gini":
        return _calculate_gini(y_true)
    elif metric == "entropy":
        return _calculate_entropy(y_true)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _calculate_gini(y: np.ndarray) -> float:
    """Calculate the Gini impurity."""
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)

def _calculate_entropy(y: np.ndarray) -> float:
    """Calculate the entropy."""
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def _calculate_leaf_value(y: np.ndarray) -> Any:
    """Calculate the value of a leaf node."""
    unique, counts = np.unique(y, return_counts=True)
    return unique[np.argmax(counts)]

def _predict(tree: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Predict the target values for a given input."""
    y_pred = np.zeros(X.shape[0], dtype=tree["value"].dtype)

    for i in range(X.shape[0]):
        node = tree
        while not node["is_leaf"]:
            if X[i, node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        y_pred[i] = node["value"]

    return y_pred

def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate various metrics for the predictions."""
    accuracy = np.mean(y_true == y_pred)
    return {"accuracy": accuracy}

################################################################################
# early_stopping
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def early_stopping_fit(
    model: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: Union[str, Callable] = 'mse',
    patience: int = 5,
    min_delta: float = 0.0,
    max_iter: int = 1000,
    normalize: Optional[str] = None,
    custom_metric: Optional[Callable] = None,
    **model_kwargs
) -> Dict[str, Any]:
    """
    Early stopping for model training based on validation performance.

    Parameters:
    -----------
    model : Callable
        The model to train. Must have fit and predict methods.
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training targets.
    X_val : np.ndarray
        Validation features.
    y_val : np.ndarray
        Validation targets.
    metric : str or Callable, optional
        Metric to monitor for early stopping. Default is 'mse'.
    patience : int, optional
        Number of epochs to wait before stopping when no improvement. Default is 5.
    min_delta : float, optional
        Minimum change to qualify as an improvement. Default is 0.0.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    normalize : str, optional
        Normalization method: 'standard', 'minmax', or None. Default is None.
    custom_metric : Callable, optional
        Custom metric function if not using built-in metrics.
    **model_kwargs : dict
        Additional keyword arguments for the model.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X_train, y_train, X_val, y_val)

    # Normalize data if specified
    X_train, X_val = _apply_normalization(X_train, X_val, normalize)

    # Initialize early stopping variables
    best_metric = float('inf')
    no_improvement_count = 0
    best_model = None

    # Initialize metric function
    if isinstance(metric, str):
        metric_func = _get_metric_function(metric)
    else:
        metric_func = metric

    # Training loop with early stopping
    for i in range(max_iter):
        model.fit(X_train, y_train, **model_kwargs)
        y_pred = model.predict(X_val)

        current_metric = metric_func(y_val, y_pred)
        is_improvement = current_metric < best_metric - min_delta

        if is_improvement:
            best_metric = current_metric
            no_improvement_count = 0
            best_model = model
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            break

    # Calculate final metrics
    y_pred_final = best_model.predict(X_val)
    final_metric_value = metric_func(y_val, y_pred_final)

    # Prepare results
    result = {
        'result': best_model,
        'metrics': {'final_metric': final_metric_value},
        'params_used': {
            'metric': metric,
            'patience': patience,
            'min_delta': min_delta,
            'max_iter': i + 1,
            'normalize': normalize
        },
        'warnings': []
    }

    return result

def _validate_inputs(X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train must have the same number of samples")
    if X_val.shape[0] != y_val.shape[0]:
        raise ValueError("X_val and y_val must have the same number of samples")
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("X_train and X_val must have the same number of features")

    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        raise ValueError("X_train contains NaN or infinite values")
    if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
        raise ValueError("y_train contains NaN or infinite values")
    if np.any(np.isnan(X_val)) or np.any(np.isinf(X_val)):
        raise ValueError("X_val contains NaN or infinite values")
    if np.any(np.isnan(y_val)) or np.any(np.isinf(y_val)):
        raise ValueError("y_val contains NaN or infinite values")

def _apply_normalization(X_train: np.ndarray, X_val: np.ndarray,
                        method: Optional[str]) -> tuple:
    """Apply specified normalization to the data."""
    if method is None:
        return X_train, X_val

    if method == 'standard':
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
    elif method == 'minmax':
        min_val = np.min(X_train, axis=0)
        max_val = np.max(X_train, axis=0)
        X_train = (X_train - min_val) / (max_val - min_val + 1e-8)
        X_val = (X_val - min_val) / (max_val - min_val + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_train, X_val

def _get_metric_function(metric_name: str) -> Callable:
    """Get the metric function based on name."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared
    }

    if metric_name not in metrics:
        raise ValueError(f"Unknown metric: {metric_name}")

    return metrics[metric_name]

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
