"""
Quantix – Module classification_non_lineaire
Généré automatiquement
Date: 2026-01-06
"""

################################################################################
# SVM
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

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

def compute_kernel(X: np.ndarray, y: Optional[np.ndarray] = None,
                   kernel_func: Callable = lambda x, y: np.dot(x, y),
                   **kernel_params) -> np.ndarray:
    """Compute kernel matrix."""
    if y is None:
        y = X
    return np.array([[kernel_func(x, y_i) for y_i in y] for x in X])

def svm_dual(X: np.ndarray, y: np.ndarray, C: float = 1.0,
             kernel_func: Callable = lambda x, y: np.dot(x, y),
             tol: float = 1e-3, max_iter: int = 1000) -> np.ndarray:
    """Solve SVM dual problem using coordinate descent."""
    n_samples = X.shape[0]
    K = compute_kernel(X, kernel_func=kernel_func)
    alpha = np.zeros(n_samples)

    for _ in range(max_iter):
        alpha_prev = alpha.copy()
        for i in range(n_samples):
            alpha_i_old = alpha[i]
            sum_alpha_y = np.sum(alpha * y)
            alpha[i] = min(C, max(0, alpha[i] + y[i] * (K[i, i] - np.sum(K[i] * alpha * y))))

            if abs(alpha[i] - alpha_i_old) < tol:
                continue

        if np.linalg.norm(alpha - alpha_prev) < tol:
            break

    return alpha

def compute_decision_function(X: np.ndarray, X_train: np.ndarray,
                             y_train: np.ndarray, alpha: np.ndarray,
                             kernel_func: Callable) -> np.ndarray:
    """Compute decision function values."""
    K = compute_kernel(X, X_train, kernel_func=kernel_func)
    return np.dot(alpha * y_train, K)

def SVM_fit(X: np.ndarray, y: np.ndarray,
            normalization: str = 'standard',
            C: float = 1.0,
            kernel_func: Callable = lambda x, y: np.dot(x, y),
            tol: float = 1e-3,
            max_iter: int = 1000) -> Dict:
    """
    Fit SVM model to data.

    Parameters:
    -----------
    X : np.ndarray
        Training data of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    C : float
        Regularization parameter
    kernel_func : callable
        Kernel function (default: linear)
    tol : float
        Tolerance for stopping criterion
    max_iter : int
        Maximum number of iterations

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, 100)
    >>> result = SVM_fit(X, y)
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_norm = normalize_data(X, method=normalization)

    # Solve SVM dual problem
    alpha = svm_dual(X_norm, y, C=C, kernel_func=kernel_func,
                     tol=tol, max_iter=max_iter)

    # Compute decision function
    decision_values = compute_decision_function(X_norm, X_norm, y, alpha, kernel_func)

    # Calculate accuracy
    predictions = np.sign(decision_values)
    accuracy = np.mean(predictions == y)

    return {
        'result': {
            'alpha': alpha,
            'support_vectors': X_norm[alpha > 1e-5],
            'decision_values': decision_values
        },
        'metrics': {
            'accuracy': accuracy
        },
        'params_used': {
            'normalization': normalization,
            'C': C,
            'kernel_func': kernel_func.__name__ if hasattr(kernel_func, '__name__') else 'custom',
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

################################################################################
# RandomForest
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

def compute_metric(y_true: np.ndarray, y_pred: np.ndarray,
                   metric: Union[str, Callable]) -> float:
    """Compute specified metric between true and predicted values."""
    if callable(metric):
        return metric(y_true, y_pred)
    elif metric == 'accuracy':
        return np.mean(y_true == y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def bootstrap_sample(X: np.ndarray, y: np.ndarray,
                     sample_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a bootstrap sample from the data."""
    indices = np.random.choice(X.shape[0], size=sample_size, replace=True)
    return X[indices], y[indices]

def build_tree(X: np.ndarray, y: np.ndarray,
               max_depth: int = None,
               min_samples_split: int = 2) -> Dict[str, Any]:
    """Build a single decision tree."""
    # Placeholder for actual tree building logic
    return {'type': 'tree', 'depth': 0}

def random_forest_fit(X: np.ndarray, y: np.ndarray,
                      n_estimators: int = 100,
                      max_depth: Optional[int] = None,
                      min_samples_split: int = 2,
                      normalization: str = 'standard',
                      metric: Union[str, Callable] = 'accuracy') -> Dict[str, Any]:
    """
    Fit a Random Forest classifier.

    Parameters:
    - X: Feature matrix of shape (n_samples, n_features)
    - y: Target vector of shape (n_samples,)
    - n_estimators: Number of trees in the forest
    - max_depth: Maximum depth of each tree
    - min_samples_split: Minimum number of samples required to split a node
    - normalization: Normalization method ('none', 'standard', 'minmax', 'robust')
    - metric: Evaluation metric (string or callable)

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_normalized = normalize_data(X, normalization)

    # Initialize results dictionary
    results = {
        'result': None,
        'metrics': {},
        'params_used': {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'normalization': normalization
        },
        'warnings': []
    }

    # Build forest (placeholder for actual implementation)
    trees = [build_tree(*bootstrap_sample(X_normalized, y, X.shape[0]))
             for _ in range(n_estimators)]

    # Store results
    results['result'] = trees

    # Compute metrics (placeholder for actual prediction and evaluation)
    y_pred = np.zeros_like(y)  # Placeholder
    results['metrics']['accuracy'] = compute_metric(y, y_pred, metric)

    return results

# Example usage:
"""
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, size=100)

results = random_forest_fit(X, y,
                           n_estimators=50,
                           max_depth=10,
                           normalization='standard')
"""

################################################################################
# GradientBoosting
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
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

def compute_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute gradient for classification."""
    return (y_true - y_pred)

def compute_hessian(y_pred: np.ndarray) -> np.ndarray:
    """Compute hessian for classification."""
    return y_pred * (1 - y_pred)

def initialize_estimator(n_classes: int) -> np.ndarray:
    """Initialize estimator with class probabilities."""
    return np.full(n_classes, 1.0 / n_classes)

def update_estimator(F: np.ndarray, gradient: np.ndarray,
                    hessian: np.ndarray, learning_rate: float) -> None:
    """Update estimator with gradient and hessian."""
    F -= learning_rate * (gradient / (hessian + 1e-8))

def GradientBoosting_fit(X: np.ndarray, y: np.ndarray,
                        n_estimators: int = 100,
                        learning_rate: float = 0.1,
                        max_depth: int = 3,
                        min_samples_split: int = 2,
                        loss: str = 'log_loss',
                        normalization: str = 'standard',
                        metric: Union[str, Callable] = 'logloss') -> Dict[str, Any]:
    """
    Gradient Boosting for classification.

    Parameters:
    -----------
    X : np.ndarray
        Training data of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    n_estimators : int, optional
        Number of boosting stages to perform (default=100)
    learning_rate : float, optional
        Shrinkage factor for each tree (default=0.1)
    max_depth : int, optional
        Maximum depth of each tree (default=3)
    min_samples_split : int, optional
        Minimum number of samples required to split a node (default=2)
    loss : str, optional
        Loss function ('log_loss' or 'exponential') (default='log_loss')
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default='standard')
    metric : str or callable, optional
        Metric to evaluate model ('mse', 'mae', 'r2', 'logloss') (default='logloss')

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_norm = normalize_data(X, normalization)
    n_samples, n_features = X_norm.shape
    n_classes = len(np.unique(y))

    # Initialize estimator
    F = initialize_estimator(n_classes)

    # Training loop
    for _ in range(n_estimators):
        # Compute predictions
        y_pred = 1.0 / (1.0 + np.exp(-F))

        # Compute gradient and hessian
        grad = compute_gradient(y, y_pred)
        hess = compute_hessian(y_pred)

        # Fit tree to negative gradient
        # (In a real implementation, this would be replaced with actual tree fitting)
        tree = fit_tree(X_norm, -grad / (hess + 1e-8), max_depth, min_samples_split)

        # Update estimator
        update_estimator(F, tree.predict(X_norm), hess, learning_rate)

    # Compute final predictions and metrics
    y_pred = 1.0 / (1.0 + np.exp(-F))
    final_metric = compute_metric(y, y_pred, metric)

    return {
        'result': {'estimator': F},
        'metrics': {metric: final_metric},
        'params_used': {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'loss': loss,
            'normalization': normalization
        },
        'warnings': []
    }

def fit_tree(X: np.ndarray, grad: np.ndarray,
            max_depth: int, min_samples_split: int) -> Any:
    """
    Placeholder for tree fitting function.
    In a real implementation, this would be replaced with actual decision tree fitting.
    """
    class DummyTree:
        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.zeros(X.shape[0])
    return DummyTree()

################################################################################
# XGBoost
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

def _xgboost_objective(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Default XGBoost objective function for binary classification."""
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
    grad = y_pred - y_true
    hess = np.ones_like(y_true)
    return grad, hess

def _xgboost_evaluate(preds: np.ndarray, dtrain: Dict) -> str:
    """Default XGBoost evaluation function."""
    labels = dtrain.get('label')
    metric = _compute_metric(labels, preds, 'logloss')
    return f'metric', float(metric)

def XGBoost_fit(X: np.ndarray, y: np.ndarray,
                n_estimators: int = 100,
                max_depth: int = 3,
                learning_rate: float = 0.1,
                normalization: str = 'none',
                metric: Union[str, Callable] = 'logloss',
                objective: Optional[Callable] = None,
                eval_metric: Union[str, Callable] = 'logloss',
                early_stopping_rounds: Optional[int] = None,
                verbose: bool = False) -> Dict:
    """
    Fit XGBoost model to the given data.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    n_estimators : int, optional
        Number of boosting rounds (default: 100)
    max_depth : int, optional
        Maximum tree depth (default: 3)
    learning_rate : float, optional
        Learning rate (default: 0.1)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'none')
    metric : str or callable, optional
        Metric for evaluation (default: 'logloss')
    objective : callable, optional
        Custom objective function
    eval_metric : str or callable, optional
        Metric for evaluation (default: 'logloss')
    early_stopping_rounds : int, optional
        Number of rounds for early stopping (default: None)
    verbose : bool, optional
        Whether to print progress (default: False)

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = XGBoost_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm = _normalize_data(X, normalization)
    dtrain = {'data': X_norm, 'label': y}

    # Set default objective if not provided
    if objective is None:
        objective = _xgboost_objective

    # Set default evaluation function
    if callable(eval_metric):
        eval_func = lambda preds, dtrain: ('custom', float(eval_metric(preds, dtrain['label'])))
    else:
        eval_func = _xgboost_evaluate

    # Initialize model parameters
    params = {
        'objective': objective,
        'eval_metric': eval_func,
        'max_depth': max_depth,
        'eta': learning_rate,
        'silent': not verbose
    }

    # Placeholder for actual XGBoost implementation
    # In a real implementation, this would call the XGBoost library
    model = None  # This would be the trained model

    # Train model (placeholder for actual training)
    for i in range(n_estimators):
        if verbose:
            print(f"Boosting round {i+1}/{n_estimators}")

        # In a real implementation, this would train one boosting round
        pass

    # Compute final predictions and metrics
    y_pred = np.zeros_like(y)  # Placeholder for actual predictions
    final_metric = _compute_metric(y, y_pred, metric)

    # Prepare results
    result = {
        'result': model,
        'metrics': {'final_metric': final_metric},
        'params_used': {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'normalization': normalization
        },
        'warnings': []
    }

    return result

################################################################################
# LightGBM
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

def _normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
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

def _compute_metric(y_true: np.ndarray, y_pred: np.ndarray,
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

def _lightgbm_objective(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Default LightGBM objective function for binary classification."""
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
    grad = y_pred - y_true
    hess = np.ones_like(y_pred)
    return grad, hess

def _lightgbm_evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Default LightGBM evaluation metric (binary logloss)."""
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
    return 'logloss', -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)), False

def LightGBM_fit(X: np.ndarray, y: np.ndarray,
                 normalizer: Callable = _normalize_data,
                 metric: Union[str, Callable] = 'logloss',
                 objective: Optional[Callable] = None,
                 eval_metric: Optional[Union[str, Callable]] = None,
                 **kwargs) -> Dict[str, Any]:
    """
    Fit a LightGBM model for non-linear classification.

    Parameters:
    -----------
    X : np.ndarray
        Input features (n_samples, n_features)
    y : np.ndarray
        Target values (n_samples,)
    normalizer : Callable, optional
        Function to normalize input data
    metric : str or Callable, optional
        Metric for evaluation (default: 'logloss')
    objective : Callable, optional
        Custom objective function
    eval_metric : str or Callable, optional
        Metric for evaluation (default: same as metric)
    **kwargs : dict
        Additional parameters passed to the LightGBM implementation

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = LightGBM_fit(X, y, normalizer=_normalize_data, metric='logloss')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = normalizer(X, method='standard')

    # Set default objective and eval_metric if not provided
    if objective is None:
        objective = _lightgbm_objective
    if eval_metric is None:
        eval_metric = metric

    # Here you would typically call the actual LightGBM implementation
    # For this example, we'll simulate a simple result

    # Simulate training (in a real implementation, this would be the actual LightGBM training)
    class SimulatedModel:
        def __init__(self):
            self.feature_importances_ = np.random.rand(X.shape[1])
            self.classes_ = np.unique(y)

        def predict(self, X):
            return np.random.rand(X.shape[0])

    model = SimulatedModel()

    # Compute metrics on training data
    y_pred = model.predict(X_normalized)
    train_metric = _compute_metric(y, y_pred, metric)

    # Prepare results
    result = {
        'result': model,
        'metrics': {'train': train_metric},
        'params_used': {
            'normalization_method': 'standard',
            'metric': metric,
            'objective_function': objective.__name__ if callable(objective) else str(objective),
            'eval_metric': eval_metric.__name__ if callable(eval_metric) else str(eval_metric),
            **kwargs
        },
        'warnings': []
    }

    return result

################################################################################
# CatBoost
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    normalize: str = 'none',
    custom_normalize: Optional[Callable] = None
) -> Dict[str, np.ndarray]:
    """
    Validate and preprocess input data.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalize : str or None
        Normalization method: 'none', 'standard', 'minmax', 'robust'
    custom_normalize : callable or None
        Custom normalization function

    Returns:
    --------
    dict
        Dictionary containing validated and processed data
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")

    if normalize == 'none' and custom_normalize is None:
        X_processed = X
    elif normalize == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_processed = (X - mean) / std
    elif normalize == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_processed = (X - min_val) / (max_val - min_val + 1e-8)
    elif normalize == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_processed = (X - median) / (iqr + 1e-8)
    elif custom_normalize is not None:
        X_processed = custom_normalize(X)
    else:
        raise ValueError("Invalid normalization method")

    return {
        'X': X_processed,
        'y': y,
        'params_used': {'normalize': normalize}
    }

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = 'logloss',
    custom_metric: Optional[Callable] = None
) -> float:
    """
    Compute classification metric.

    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted probabilities or classes
    metric : str
        Metric to compute: 'logloss', 'accuracy'
    custom_metric : callable or None
        Custom metric function

    Returns:
    --------
    float
        Computed metric value
    """
    if custom_metric is not None:
        return custom_metric(y_true, y_pred)

    if metric == 'logloss':
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    elif metric == 'accuracy':
        return np.mean(y_true == (y_pred > 0.5))
    else:
        raise ValueError("Invalid metric specified")

def _gradient_boosting(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    depth: int = 6
) -> np.ndarray:
    """
    Gradient boosting implementation.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    n_estimators : int
        Number of boosting iterations
    learning_rate : float
        Learning rate
    depth : int
        Maximum tree depth

    Returns:
    --------
    np.ndarray
        Predicted probabilities
    """
    # Initialize predictions with mean value
    y_pred = np.mean(y)

    for _ in range(n_estimators):
        # Compute pseudo-residuals
        residuals = y - _predict_tree(X, y_pred, depth)

        # Fit weak learner to residuals
        tree = _fit_tree(X, residuals, depth)

        # Update predictions
        y_pred += learning_rate * _predict_tree(X, tree)

    return np.clip(y_pred, 1e-15, 1 - 1e-15)

def _fit_tree(
    X: np.ndarray,
    y: np.ndarray,
    depth: int
) -> Dict:
    """
    Fit a decision tree to data.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    depth : int
        Maximum tree depth

    Returns:
    --------
    dict
        Tree structure
    """
    # Simplified tree fitting implementation
    return {'type': 'tree', 'depth': depth}

def _predict_tree(
    X: np.ndarray,
    tree: Dict
) -> np.ndarray:
    """
    Predict using a decision tree.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    tree : dict
        Tree structure

    Returns:
    --------
    np.ndarray
        Predictions
    """
    # Simplified tree prediction implementation
    return np.zeros(X.shape[0])

def CatBoost_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalize: str = 'none',
    metric: str = 'logloss',
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    depth: int = 6,
    custom_normalize: Optional[Callable] = None,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Fit CatBoost model to data.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,) with values in {0, 1}
    normalize : str or None
        Normalization method: 'none', 'standard', 'minmax', 'robust'
    metric : str
        Metric to optimize: 'logloss', 'accuracy'
    n_estimators : int
        Number of boosting iterations
    learning_rate : float
        Learning rate
    depth : int
        Maximum tree depth
    custom_normalize : callable or None
        Custom normalization function
    custom_metric : callable or None
        Custom metric function

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = CatBoost_fit(X, y)
    """
    # Validate and preprocess inputs
    processed_data = _validate_inputs(X, y, normalize, custom_normalize)
    X_processed = processed_data['X']
    y_processed = processed_data['y']

    # Train model
    y_pred = _gradient_boosting(
        X_processed,
        y_processed,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        depth=depth
    )

    # Compute metrics
    main_metric = _compute_metric(y_processed, y_pred, metric, custom_metric)

    return {
        'result': {'predictions': y_pred},
        'metrics': {metric: main_metric},
        'params_used': {
            **processed_data['params_used'],
            'metric': metric,
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'depth': depth
        },
        'warnings': []
    }

################################################################################
# KNN
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate input data for KNN.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).

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
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")

def normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Normalize the feature matrix.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    method : str, optional
        Normalization method. Options: 'none', 'standard', 'minmax', 'robust'.

    Returns
    ------
    np.ndarray
        Normalized feature matrix.
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

def compute_distance(X_train: np.ndarray, X_test: np.ndarray,
                     metric: str = 'euclidean',
                     custom_metric: Optional[Callable] = None) -> np.ndarray:
    """
    Compute distance between training and test data.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix of shape (n_samples, n_features).
    X_test : np.ndarray
        Test feature matrix of shape (m_samples, n_features).
    metric : str, optional
        Distance metric. Options: 'euclidean', 'manhattan', 'cosine', 'minkowski'.
    custom_metric : Callable, optional
        Custom distance metric function.

    Returns
    ------
    np.ndarray
        Distance matrix of shape (m_samples, n_samples).
    """
    if custom_metric is not None:
        return np.array([custom_metric(x, X_train) for x in X_test])
    if metric == 'euclidean':
        return np.sqrt(np.sum((X_train[:, np.newaxis] - X_test) ** 2, axis=2))
    elif metric == 'manhattan':
        return np.sum(np.abs(X_train[:, np.newaxis] - X_test), axis=2)
    elif metric == 'cosine':
        dot_products = np.dot(X_test, X_train.T)
        norms = np.linalg.norm(X_test, axis=1)[:, np.newaxis] * np.linalg.norm(X_train, axis=1)
        return 1 - dot_products / (norms + 1e-8)
    elif metric == 'minkowski':
        p = 3
        return np.sum(np.abs(X_train[:, np.newaxis] - X_test) ** p, axis=2) ** (1/p)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def predict_knn(distances: np.ndarray, y_train: np.ndarray,
                k: int = 5,
                weighted: bool = False) -> np.ndarray:
    """
    Predict classes using KNN.

    Parameters
    ----------
    distances : np.ndarray
        Distance matrix of shape (m_samples, n_samples).
    y_train : np.ndarray
        Training target vector of shape (n_samples,).
    k : int, optional
        Number of neighbors.
    weighted : bool, optional
        Whether to use weighted voting.

    Returns
    ------
    np.ndarray
        Predicted classes of shape (m_samples,).
    """
    n_samples = distances.shape[0]
    predictions = np.zeros(n_samples)
    for i in range(n_samples):
        closest_indices = np.argsort(distances[i])[:k]
        if weighted:
            weights = 1 / (distances[i][closest_indices] + 1e-8)
            weights /= np.sum(weights)
        else:
            weights = np.ones(k) / k
        classes, counts = np.unique(y_train[closest_indices], return_counts=True)
        weighted_counts = weights * (counts / np.sum(counts))
        predictions[i] = classes[np.argmax(weighted_counts)]
    return predictions

def KNN_fit(X_train: np.ndarray, y_train: np.ndarray,
            X_test: Optional[np.ndarray] = None,
            k: int = 5,
            metric: str = 'euclidean',
            custom_metric: Optional[Callable] = None,
            normalize_method: str = 'standard',
            weighted_voting: bool = False) -> Dict:
    """
    Fit KNN model and predict on test data.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix of shape (n_samples, n_features).
    y_train : np.ndarray
        Training target vector of shape (n_samples,).
    X_test : np.ndarray, optional
        Test feature matrix of shape (m_samples, n_features).
    k : int, optional
        Number of neighbors.
    metric : str, optional
        Distance metric. Options: 'euclidean', 'manhattan', 'cosine', 'minkowski'.
    custom_metric : Callable, optional
        Custom distance metric function.
    normalize_method : str, optional
        Normalization method. Options: 'none', 'standard', 'minmax', 'robust'.
    weighted_voting : bool, optional
        Whether to use weighted voting.

    Returns
    ------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    validate_inputs(X_train, y_train)
    if X_test is not None:
        validate_inputs(X_test, np.zeros(X_test.shape[0]))

    # Normalize data
    X_train_norm = normalize_data(X_train, method=normalize_method)
    if X_test is not None:
        X_test_norm = normalize_data(X_test, method=normalize_method)
    else:
        X_test_norm = None

    # Compute distances
    distances = compute_distance(X_train_norm, X_test_norm if X_test is not None else X_train_norm,
                                metric=metric, custom_metric=custom_metric)

    # Predict
    y_pred = predict_knn(distances, y_train, k=k, weighted=weighted_voting)

    # Prepare output
    result = {
        'result': y_pred,
        'metrics': {},
        'params_used': {
            'k': k,
            'metric': metric if custom_metric is None else 'custom',
            'normalize_method': normalize_method,
            'weighted_voting': weighted_voting
        },
        'warnings': []
    }

    return result

# Example usage:
"""
X_train = np.random.rand(100, 5)
y_train = np.random.randint(0, 2, size=100)
X_test = np.random.rand(10, 5)

result = KNN_fit(X_train, y_train, X_test, k=3, metric='euclidean', normalize_method='standard')
"""

################################################################################
# NeuralNetworks
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def NeuralNetworks_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: str = "logloss",
    solver: str = "gradient_descent",
    activation: Callable[[np.ndarray], np.ndarray] = lambda x: 1 / (1 + np.exp(-x)),
    optimizer: Callable[[np.ndarray, float], np.ndarray] = lambda w, lr: w - lr * gradient,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
    batch_size: Optional[int] = None,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Fit a neural network model for non-linear classification.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target labels of shape (n_samples,) or (n_samples, n_outputs).
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the input features.
    metric : str
        Metric to evaluate model performance. Options: "logloss", "accuracy".
    solver : str
        Solver to use for optimization. Options: "gradient_descent", "newton".
    activation : Callable[[np.ndarray], np.ndarray]
        Activation function for the neural network.
    optimizer : Callable[[np.ndarray, float], np.ndarray]
        Optimization function to update weights.
    learning_rate : float
        Learning rate for gradient descent.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for stopping criterion.
    batch_size : Optional[int]
        Batch size for stochastic gradient descent. If None, uses full batch.
    random_state : Optional[int]
        Random seed for reproducibility.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]
        Dictionary containing:
        - "result": Fitted model weights.
        - "metrics": Computed metrics.
        - "params_used": Parameters used during fitting.
        - "warnings": Any warnings encountered.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = NeuralNetworks_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_features = X.shape
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Initialize weights
    weights = _initialize_weights(n_features, y.shape[1])

    # Normalize features
    X_normalized = normalizer(X)

    # Training loop
    for i in range(max_iter):
        if batch_size is not None:
            indices = np.random.choice(n_samples, batch_size, replace=False)
            X_batch = X_normalized[indices]
            y_batch = y[indices]
        else:
            X_batch = X_normalized
            y_batch = y

        # Forward pass
        predictions = _forward_pass(X_batch, weights, activation)

        # Compute loss and gradient
        loss, gradient = _compute_loss_and_gradient(predictions, y_batch, metric)

        # Update weights
        weights = optimizer(weights, learning_rate * gradient)

        if verbose and i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss}")

        if loss < tol:
            break

    # Compute final metrics
    predictions = _forward_pass(X_normalized, weights, activation)
    metrics = _compute_metrics(predictions, y, metric)

    return {
        "result": weights,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if hasattr(normalizer, '__name__') else "custom",
            "metric": metric,
            "solver": solver,
            "activation": activation.__name__ if hasattr(activation, '__name__') else "custom",
            "optimizer": optimizer.__name__ if hasattr(optimizer, '__name__') else "custom",
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "tol": tol,
            "batch_size": batch_size
        },
        "warnings": []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim not in (1, 2):
        raise ValueError("X must be 2D and y must be 1D or 2D")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains NaN or infinite values")
    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("y contains NaN or infinite values")

def _initialize_weights(n_features: int, n_outputs: int) -> np.ndarray:
    """Initialize weights for the neural network."""
    return np.random.randn(n_features, n_outputs) * 0.01

def _forward_pass(X: np.ndarray, weights: np.ndarray, activation: Callable) -> np.ndarray:
    """Perform forward pass through the neural network."""
    return activation(X @ weights)

def _compute_loss_and_gradient(
    predictions: np.ndarray,
    y: np.ndarray,
    metric: str
) -> tuple[float, np.ndarray]:
    """Compute loss and gradient based on the specified metric."""
    if metric == "logloss":
        m = y.shape[0]
        loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        gradient = (predictions - y) / m
    elif metric == "accuracy":
        loss = 1 - np.mean((predictions > 0.5) == y)
        gradient = (predictions - y) / y.shape[0]
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return loss, gradient

def _compute_metrics(
    predictions: np.ndarray,
    y: np.ndarray,
    metric: str
) -> Dict[str, float]:
    """Compute metrics for the model."""
    if metric == "logloss":
        m = y.shape[0]
        logloss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        accuracy = np.mean((predictions > 0.5) == y)
    elif metric == "accuracy":
        accuracy = np.mean((predictions > 0.5) == y)
        m = y.shape[0]
        logloss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return {"logloss": logloss, "accuracy": accuracy}

################################################################################
# KernelMethods
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
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
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

def _compute_kernel(X: np.ndarray, kernel_func: Callable, **kernel_params) -> np.ndarray:
    """Compute kernel matrix."""
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = kernel_func(X[i], X[j], **kernel_params)
    return K

def _solve_dual_problem(K: np.ndarray, y: np.ndarray, C: float = 1.0) -> np.ndarray:
    """Solve the dual optimization problem."""
    n_samples = K.shape[0]
    P = np.outer(y, y) * K
    q = -np.ones(n_samples)
    G = -np.eye(n_samples)
    h = np.zeros(n_samples)
    A = y.reshape(1, -1)
    b = 0.0

    from cvxopt import matrix, solvers
    solvers.options['show_progress'] = False

    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A, tc='d')
    b = matrix(b)

    solution = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(solution['x']).flatten()
    return alphas

def _compute_decision_function(X: np.ndarray, X_train: np.ndarray,
                              y_train: np.ndarray, alphas: np.ndarray,
                              kernel_func: Callable, bias: float = 0.0,
                              **kernel_params) -> np.ndarray:
    """Compute decision function for new data."""
    n_samples = X.shape[0]
    K = np.zeros(n_samples)
    for i in range(n_samples):
        K[i] = np.sum(alphas * y_train * kernel_func(X[i], X_train, **kernel_params))
    return K + bias

def KernelMethods_fit(X: np.ndarray, y: np.ndarray,
                     kernel_func: Callable = lambda x1, x2: np.dot(x1, x2),
                     normalization: str = 'standard',
                     C: float = 1.0,
                     metric_func: Optional[Callable] = None,
                     **kernel_params) -> Dict:
    """
    Fit kernel methods for non-linear classification.

    Parameters:
    -----------
    X : np.ndarray
        Training data of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    kernel_func : Callable
        Kernel function to use. Default is linear kernel.
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    C : float
        Regularization parameter
    metric_func : Optional[Callable]
        Custom metric function. If None, uses accuracy.
    **kernel_params
        Additional parameters for the kernel function

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> kernel_func = lambda x1, x2: np.exp(-np.linalg.norm(x1 - x2)**2)
    >>> result = KernelMethods_fit(X_train, y_train, kernel_func=kernel_func,
                                 normalization='standard', C=1.0)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _normalize_data(X, normalization)

    # Compute kernel matrix
    K = _compute_kernel(X_normalized, kernel_func, **kernel_params)

    # Solve dual problem
    alphas = _solve_dual_problem(K, y, C)

    # Compute support vectors and bias
    sv_indices = alphas > 1e-5
    X_sv = X_normalized[sv_indices]
    y_sv = y[sv_indices]
    alphas_sv = alphas[sv_indices]

    # Compute bias
    bias = np.mean(y_sv - _compute_decision_function(X_sv, X_sv,
                                                    y_sv, alphas_sv,
                                                    kernel_func, **kernel_params))

    # Compute metrics
    if metric_func is None:
        def accuracy(y_true, y_pred):
            return np.mean(y_true == y_pred)
        metric_func = accuracy

    # Create prediction function
    def predict(X_test):
        X_test_normalized = _normalize_data(X_test, normalization)
        return np.sign(_compute_decision_function(X_test_normalized,
                                                 X_normalized, y,
                                                 alphas, kernel_func,
                                                 bias=bias, **kernel_params))

    # Compute training accuracy
    y_pred = predict(X)
    train_metric = metric_func(y, y_pred)

    return {
        'result': predict,
        'metrics': {'train_metric': train_metric},
        'params_used': {
            'normalization': normalization,
            'C': C,
            'kernel_func': kernel_func.__name__ if hasattr(kernel_func, '__name__') else 'custom',
            'kernel_params': kernel_params
        },
        'warnings': []
    }

################################################################################
# DecisionTrees
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def DecisionTrees_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: Optional[Union[int, float]] = None,
    criterion: str = 'gini',
    splitter: str = 'best',
    random_state: Optional[int] = None,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = 'euclidean',
    normalization: Optional[str] = None,
    metric: str = 'accuracy',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    max_iter: int = 100,
    tol: float = 1e-4
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, any]]]:
    """
    Fit a decision tree classifier to the training data.

    Parameters:
    -----------
    X : np.ndarray
        Training data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    max_depth : Optional[int], default=None
        The maximum depth of the tree.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    max_features : Optional[Union[int, float]], default=None
        The number of features to consider when looking for the best split.
    criterion : str, default='gini'
        The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain.
    splitter : str, default='best'
        The strategy used to choose the split at each node. Supported strategies are "best" to choose the best split and "random" to choose the best random split.
    random_state : Optional[int], default=None
        Controls both the randomness of the bootstrapping of the samples used when building trees and the sampling of the features to consider when looking for the best split at each node.
    distance_metric : Callable[[np.ndarray, np.ndarray], float] or str, default='euclidean'
        The distance metric to use for splitting. Can be a callable or one of the predefined metrics.
    normalization : Optional[str], default=None
        The normalization method to apply to the data. Supported methods are "standard", "minmax", and "robust".
    metric : str, default='accuracy'
        The metric to evaluate the model. Supported metrics are "accuracy", "precision", "recall", and "f1".
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], default=None
        A custom metric function to evaluate the model.
    max_iter : int, default=100
        The maximum number of iterations for the solver.
    tol : float, default=1e-4
        The tolerance for the stopping criterion.

    Returns:
    --------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, any]]]
        A dictionary containing the fitted model, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_normalized = _apply_normalization(X, normalization)

    # Initialize parameters
    params_used = {
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'criterion': criterion,
        'splitter': splitter,
        'random_state': random_state,
        'distance_metric': distance_metric,
        'normalization': normalization,
        'metric': metric,
        'custom_metric': custom_metric,
        'max_iter': max_iter,
        'tol': tol
    }

    # Fit the decision tree
    tree = _fit_decision_tree(
        X_normalized, y,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        splitter=splitter,
        random_state=random_state,
        distance_metric=distance_metric
    )

    # Compute metrics
    y_pred = _predict(tree, X_normalized)
    metrics = _compute_metrics(y, y_pred, metric=metric, custom_metric=custom_metric)

    # Return results
    return {
        'result': tree,
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
        raise ValueError("Input data contains NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input data contains infinite values.")

def _apply_normalization(X: np.ndarray, method: Optional[str]) -> np.ndarray:
    """Apply normalization to the data."""
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
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _fit_decision_tree(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: Optional[Union[int, float]] = None,
    criterion: str = 'gini',
    splitter: str = 'best',
    random_state: Optional[int] = None,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = 'euclidean'
) -> Dict:
    """Fit a decision tree to the data."""
    np.random.seed(random_state)
    n_samples, n_features = X.shape

    if max_features is None:
        max_features = n_features
    elif isinstance(max_features, float):
        max_features = int(max_features * n_features)

    tree = _build_tree(
        X, y,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        splitter=splitter,
        distance_metric=distance_metric
    )

    return tree

def _build_tree(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: Optional[Union[int, float]] = None,
    criterion: str = 'gini',
    splitter: str = 'best',
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = 'euclidean',
    depth: int = 0
) -> Dict:
    """Recursively build a decision tree."""
    n_samples, n_features = X.shape
    n_labels = len(np.unique(y))

    # Stopping criteria
    if (max_depth is not None and depth >= max_depth) or \
       n_samples < min_samples_split or \
       (n_labels == 1 and criterion != 'entropy'):
        return {
            'leaf': True,
            'class': np.bincount(y).argmax()
        }

    # Find the best split
    if splitter == 'best':
        best_split = _find_best_split(
            X, y,
            max_features=max_features,
            criterion=criterion,
            distance_metric=distance_metric
        )
    else:
        best_split = _find_random_split(
            X, y,
            max_features=max_features,
            criterion=criterion,
            distance_metric=distance_metric
        )

    if best_split is None:
        return {
            'leaf': True,
            'class': np.bincount(y).argmax()
        }

    feature_idx, threshold = best_split
    left_mask = X[:, feature_idx] <= threshold
    right_mask = ~left_mask

    # Recursively build left and right subtrees
    left_subtree = _build_tree(
        X[left_mask], y[left_mask],
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        splitter=splitter,
        distance_metric=distance_metric,
        depth=depth + 1
    )

    right_subtree = _build_tree(
        X[right_mask], y[right_mask],
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        splitter=splitter,
        distance_metric=distance_metric,
        depth=depth + 1
    )

    return {
        'leaf': False,
        'feature_idx': feature_idx,
        'threshold': threshold,
        'left': left_subtree,
        'right': right_subtree
    }

def _find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_features: Optional[Union[int, float]] = None,
    criterion: str = 'gini',
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = 'euclidean'
) -> Optional[tuple]:
    """Find the best split for a node."""
    n_samples, n_features = X.shape
    if max_features is None:
        max_features = n_features

    best_score = -np.inf
    best_split = None

    # Randomly select features if max_features is specified
    feature_indices = np.random.choice(n_features, size=max_features, replace=False)

    for feature_idx in feature_indices:
        thresholds = np.unique(X[:, feature_idx])
        for threshold in thresholds:
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask

            if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
                continue

            if criterion == 'gini':
                score = _gini_impurity(y[left_mask], y[right_mask])
            elif criterion == 'entropy':
                score = _information_gain(y[left_mask], y[right_mask])
            else:
                raise ValueError(f"Unknown criterion: {criterion}")

            if score > best_score:
                best_score = score
                best_split = (feature_idx, threshold)

    return best_split

def _find_random_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_features: Optional[Union[int, float]] = None,
    criterion: str = 'gini',
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = 'euclidean'
) -> Optional[tuple]:
    """Find a random split for a node."""
    n_samples, n_features = X.shape
    if max_features is None:
        max_features = n_features

    feature_indices = np.random.choice(n_samples, size=max_features, replace=False)
    random_idx = np.random.randint(0, n_samples)

    feature_idx = feature_indices[random_idx]
    threshold = X[random_idx, feature_idx]

    left_mask = X[:, feature_idx] <= threshold
    right_mask = ~left_mask

    if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
        return None

    if criterion == 'gini':
        score = _gini_impurity(y[left_mask], y[right_mask])
    elif criterion == 'entropy':
        score = _information_gain(y[left_mask], y[right_mask])
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    return (feature_idx, threshold) if score > 0 else None

def _gini_impurity(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Calculate the Gini impurity for a split."""
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right

    if n_left == 0 or n_right == 0:
        return 0.0

    gini_left = 1.0 - np.sum(np.bincount(y_left) / n_left) ** 2
    gini_right = 1.0 - np.sum(np.bincount(y_right) / n_right) ** 2

    return (n_left * gini_left + n_right * gini_right) / n_total

def _information_gain(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Calculate the information gain for a split."""
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right

    if n_left == 0 or n_right == 0:
        return 0.0

    entropy_left = _entropy(y_left)
    entropy_right = _entropy(y_right)

    return (n_left * entropy_left + n_right * entropy_right) / n_total

def _entropy(y: np.ndarray) -> float:
    """Calculate the entropy of a set."""
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-8))

def _predict(tree: Dict, X: np.ndarray) -> np.ndarray:
    """Predict the class labels for a set of samples."""
    y_pred = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        node = tree
        while not node['leaf']:
            if X[i, node['feature_idx']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        y_pred[i] = node['class']

    return y_pred

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: str = 'accuracy',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Compute the metrics for the model."""
    metrics = {}

    if metric == 'accuracy':
        metrics['accuracy'] = np.mean(y_true == y_pred)
    elif metric == 'precision':
        metrics['precision'] = _precision(y_true, y_pred)
    elif metric == 'recall':
        metrics['recall'] = _recall(y_true, y_pred)
    elif metric == 'f1':
        metrics['f1'] = _f1_score(y_true, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)

    return metrics

def _precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the precision."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp + 1e-8)

def _recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the recall."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn + 1e-8)

def _f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the F1 score."""
    precision = _precision(y_true, y_pred)
    recall = _recall(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall + 1e-8)

################################################################################
# EnsembleMethods
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def EnsembleMethods_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'logloss',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Fit an ensemble of non-linear classifiers.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the input features. Default is None.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Metric to evaluate the model. Can be 'mse', 'mae', 'r2', 'logloss', or a custom callable.
        Default is 'logloss'.
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Distance metric for the model. Can be 'euclidean', 'manhattan', 'cosine', 'minkowski', or a custom callable.
        Default is 'euclidean'.
    solver : str, optional
        Solver to use. Can be 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'.
        Default is 'closed_form'.
    regularization : Optional[str], optional
        Regularization type. Can be 'none', 'l1', 'l2', or 'elasticnet'. Default is None.
    tol : float, optional
        Tolerance for stopping criteria. Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function. Default is None.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom distance function. Default is None.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if a normalizer is provided
    X_normalized = _apply_normalization(X, normalizer)

    # Initialize parameters
    params = {
        'normalizer': normalizer,
        'metric': metric,
        'distance': distance,
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'max_iter': max_iter
    }

    # Choose the appropriate solver
    if solver == 'closed_form':
        result = _closed_form_solver(X_normalized, y)
    elif solver == 'gradient_descent':
        result = _gradient_descent_solver(X_normalized, y, tol=tol, max_iter=max_iter)
    elif solver == 'newton':
        result = _newton_solver(X_normalized, y, tol=tol, max_iter=max_iter)
    elif solver == 'coordinate_descent':
        result = _coordinate_descent_solver(X_normalized, y, tol=tol, max_iter=max_iter)
    else:
        raise ValueError("Invalid solver specified.")

    # Calculate metrics
    metrics = _calculate_metrics(result, y, metric=metric, custom_metric=custom_metric)

    # Return the results
    return {
        'result': result,
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate the input arrays."""
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
    if normalizer is None:
        return X
    return normalizer(X)

def _closed_form_solver(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed form solution for the ensemble methods."""
    # Placeholder for closed form solution
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def _gradient_descent_solver(X: np.ndarray, y: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
    """Gradient descent solver for the ensemble methods."""
    # Placeholder for gradient descent solution
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ weights - y) / n_samples
        new_weights = weights - learning_rate * gradient

        if np.linalg.norm(new_weights - weights) < tol:
            break
        weights = new_weights

    return weights

def _newton_solver(X: np.ndarray, y: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
    """Newton's method solver for the ensemble methods."""
    # Placeholder for Newton's method solution
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)

    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ weights - y) / n_samples
        hessian = 2 * X.T @ X / n_samples
        new_weights = weights - np.linalg.pinv(hessian) @ gradient

        if np.linalg.norm(new_weights - weights) < tol:
            break
        weights = new_weights

    return weights

def _coordinate_descent_solver(X: np.ndarray, y: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
    """Coordinate descent solver for the ensemble methods."""
    # Placeholder for coordinate descent solution
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        for i in range(n_features):
            X_i = X[:, i]
            residual = y - (X @ weights - X_i * weights[i])
            weights[i] += learning_rate * (X_i.T @ residual) / (X_i.T @ X_i)

        if np.linalg.norm(weights - weights) < tol:
            break

    return weights

def _calculate_metrics(
    result: np.ndarray,
    y_true: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Calculate the metrics for the model."""
    y_pred = result

    if custom_metric is not None:
        return {'custom_metric': custom_metric(y_true, y_pred)}

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
        metrics['logloss'] = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError("Invalid metric specified.")

    return metrics

################################################################################
# AdaBoost
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
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

def compute_error(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """Compute the error based on the specified metric."""
    if metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif metric == 'logloss':
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def compute_weighted_error(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    """Compute the weighted error."""
    return np.sum(weights * (y_true != y_pred)) / np.sum(weights)

def update_weights(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray, alpha: float) -> np.ndarray:
    """Update the sample weights."""
    error = compute_weighted_error(y_true, y_pred, weights)
    beta = error / (1.0 - error + 1e-10)
    new_weights = weights * np.exp(alpha * (y_true != y_pred))
    return new_weights / np.sum(new_weights)

def fit_weak_learner(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Callable:
    """Fit a weak learner (stub implementation)."""
    # In a real implementation, this would fit a decision stump or other weak learner
    def predict(x: np.ndarray) -> np.ndarray:
        return np.random.randint(0, 2, size=x.shape[0])
    return predict

def AdaBoost_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 50,
    learning_rate: float = 1.0,
    metric: str = 'mse',
    base_estimator: Optional[Callable] = None,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit an AdaBoost classifier.

    Parameters
    ----------
    X : np.ndarray
        Training data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    n_estimators : int, optional
        Number of boosting stages to perform. Default is 50.
    learning_rate : float, optional
        Learning rate shrinks the contribution of each classifier. Default is 1.0.
    metric : str, optional
        Metric to use for error calculation. Default is 'mse'.
    base_estimator : Callable, optional
        The base estimator to fit on random subsets of the data. Default is None.
    random_state : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        A dictionary containing the results, metrics, parameters used, and warnings.
    """
    validate_inputs(X, y)

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    weights = np.ones(n_samples) / n_samples
    estimators = []
    estimator_weights = []

    for _ in range(n_estimators):
        if base_estimator is None:
            estimator = fit_weak_learner(X, y, weights)
        else:
            estimator = base_estimator(X, y, weights)

        y_pred = np.array([estimator(x.reshape(1, -1)) for x in X]).flatten()
        error = compute_weighted_error(y, y_pred, weights)

        if error >= 0.5:
            break

        alpha = learning_rate * np.log((1 - error) / (error + 1e-10))
        estimators.append(estimator)
        estimator_weights.append(alpha)

        weights = update_weights(y, y_pred, weights, alpha)

    def predict(x: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        preds = np.array([estimator(x.reshape(1, -1)) for estimator in estimators])
        weighted_preds = np.dot(preds.T, estimator_weights)
        return np.sign(weighted_preds).astype(int)

    result = {
        'predict': predict,
        'estimators': estimators,
        'estimator_weights': estimator_weights
    }

    metrics = {
        'final_error': compute_weighted_error(y, predict(X), np.ones(n_samples) / n_samples)
    }

    params_used = {
        'n_estimators': len(estimators),
        'learning_rate': learning_rate,
        'metric': metric
    }

    warnings = []

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

################################################################################
# Bagging
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

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
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def _apply_normalization(X: np.ndarray, normalization: str) -> np.ndarray:
    """Apply specified normalization to the input data."""
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
    """Compute the specified metric between true and predicted values."""
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

def _base_estimator_fit(X_subset: np.ndarray, y_subset: np.ndarray,
                        base_model: Callable) -> Any:
    """Fit a base estimator on the given subset of data."""
    return base_model.fit(X_subset, y_subset)

def _base_estimator_predict(model: Any, X: np.ndarray) -> np.ndarray:
    """Make predictions using the base estimator."""
    return model.predict(X)

def Bagging_fit(X: np.ndarray, y: np.ndarray,
                base_model: Callable,
                n_estimators: int = 10,
                max_samples: float = 1.0,
                max_features: float = 1.0,
                normalization: str = "none",
                metric: Union[str, Callable] = "mse") -> Dict[str, Any]:
    """
    Bagging ensemble method for non-linear classification.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    base_model : Callable
        Base estimator model with fit and predict methods
    n_estimators : int, optional
        Number of base estimators in the ensemble (default: 10)
    max_samples : float, optional
        Fraction of samples to draw for each base estimator (default: 1.0)
    max_features : float, optional
        Fraction of features to draw for each base estimator (default: 1.0)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'none')
    metric : str or Callable, optional
        Metric to evaluate performance (default: 'mse')

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = Bagging_fit(X, y, DecisionTreeClassifier(), n_estimators=5)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Apply normalization
    X_norm = _apply_normalization(X, normalization)

    # Initialize ensemble
    estimators = []
    metrics = []

    # Bootstrap sampling and training
    n_samples = X.shape[0]
    n_features = X.shape[1]

    for _ in range(n_estimators):
        # Draw random samples and features
        sample_indices = np.random.choice(n_samples, size=int(max_samples * n_samples), replace=True)
        feature_indices = np.random.choice(n_features, size=int(max_features * n_features), replace=True)

        X_subset = X_norm[sample_indices][:, feature_indices]
        y_subset = y[sample_indices]

        # Train base estimator
        model = _base_estimator_fit(X_subset, y_subset, base_model)
        estimators.append(model)

        # Evaluate on full dataset
        y_pred = _base_estimator_predict(model, X_norm)
        metric_value = _compute_metric(y, y_pred, metric)
        metrics.append(metric_value)

    # Compute ensemble predictions (average for regression, majority vote for classification)
    y_pred_ensemble = np.mean([_base_estimator_predict(model, X_norm) for model in estimators], axis=0)

    # Compute final metric
    final_metric = _compute_metric(y, y_pred_ensemble, metric)

    return {
        "result": {"predictions": y_pred_ensemble},
        "metrics": {"ensemble_metric": final_metric, "individual_metrics": metrics},
        "params_used": {
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "max_features": max_features,
            "normalization": normalization,
            "metric": metric
        },
        "warnings": []
    }

################################################################################
# Boosting
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

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

def compute_distance(X1: np.ndarray, X2: np.ndarray,
                     distance: str = 'euclidean') -> np.ndarray:
    """Compute pairwise distances between two sets of samples."""
    if distance == 'euclidean':
        return np.sqrt(np.sum((X1[:, np.newaxis] - X2) ** 2, axis=2))
    elif distance == 'manhattan':
        return np.sum(np.abs(X1[:, np.newaxis] - X2), axis=2)
    elif distance == 'cosine':
        dot_products = np.dot(X1, X2.T)
        norms = np.sqrt(np.sum(X1**2, axis=1))[:, np.newaxis] * \
                np.sqrt(np.sum(X2**2, axis=1))[np.newaxis, :]
        return 1 - dot_products / (norms + 1e-8)
    elif distance == 'minkowski':
        p = 3
        return np.sum(np.abs(X1[:, np.newaxis] - X2) ** p, axis=2) ** (1/p)
    else:
        raise ValueError(f"Unknown distance: {distance}")

def boosting_fit(X: np.ndarray, y: np.ndarray,
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 loss: str = 'log_loss',
                 metric: Union[str, Callable] = 'logloss',
                 normalization: str = 'standard',
                 random_state: Optional[int] = None) -> Dict:
    """
    Fit a boosting model to the data.

    Parameters
    ----------
    X : np.ndarray
        Training data of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    n_estimators : int, optional
        Number of boosting iterations to perform (default: 100)
    learning_rate : float, optional
        Shrinkage factor for each estimator (default: 0.1)
    loss : str, optional
        Loss function to minimize ('log_loss', 'exponential') (default: 'log_loss')
    metric : str or callable, optional
        Metric to evaluate model performance (default: 'logloss')
    normalization : str, optional
        Data normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard')
    random_state : int, optional
        Random seed for reproducibility (default: None)

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': trained model parameters
        - 'metrics': evaluation metrics
        - 'params_used': parameters used during fitting
        - 'warnings': any warnings encountered

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = boosting_fit(X, y)
    """
    # Validate inputs
    validate_inputs(X, y)

    # Initialize random state
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize data
    X_norm = normalize_data(X, normalization)
    y_pred = np.zeros_like(y, dtype=float)

    # Initialize warnings
    warnings = []

    # Store parameters used
    params_used = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'loss': loss,
        'metric': metric if not callable(metric) else 'custom',
        'normalization': normalization,
        'random_state': random_state
    }

    # Boosting loop
    for i in range(n_estimators):
        # Compute residuals
        if loss == 'log_loss':
            residuals = y - y_pred
            weights = np.exp(y_pred) / (1 + np.exp(y_pred))
        elif loss == 'exponential':
            residuals = y - y_pred
            weights = np.exp(-y * y_pred)
        else:
            raise ValueError(f"Unknown loss function: {loss}")

        # Fit weak learner (stub implementation - replace with actual model)
        if i == 0:
            # First estimator: fit to original data
            weak_learner = np.random.randn(X_norm.shape[1])
        else:
            # Subsequent estimators: fit to weighted residuals
            weak_learner = np.random.randn(X_norm.shape[1])

        # Update predictions
        y_pred += learning_rate * weak_learner.dot(X_norm.T)

    # Compute final metrics
    metrics = {
        'train_metric': compute_metric(y, y_pred, metric)
    }

    # Prepare result
    result = {
        'estimators': [],  # In a real implementation, store all weak learners
        'final_prediction': y_pred,
        'classes': np.unique(y)
    }

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

################################################################################
# Stacking
################################################################################

import numpy as np
from typing import Callable, Dict, List, Optional, Union

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

def base_model_fit(X: np.ndarray, y: np.ndarray,
                   model_func: Callable) -> Callable:
    """Fit a base model and return prediction function."""
    model = model_func()
    model.fit(X, y)
    return lambda x: model.predict(x)

def stacking_fit(
    X: np.ndarray,
    y: np.ndarray,
    base_models: List[Callable],
    meta_model_func: Callable,
    normalizations: Union[str, List[str]] = 'standard',
    metrics: Union[str, List[str], List[Callable]] = 'mse',
    random_state: Optional[int] = None
) -> Dict:
    """
    Perform stacking ensemble learning for non-linear classification.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    base_models : List[Callable]
        List of callable base model constructors
    meta_model_func : Callable
        Meta model constructor
    normalizations : Union[str, List[str]]
        Normalization method(s) to apply ('none', 'standard', 'minmax', 'robust')
    metrics : Union[str, List[str], List[Callable]]
        Metric(s) to evaluate performance
    random_state : Optional[int]
        Random seed for reproducibility

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> base_models = [LogisticRegression, DecisionTreeClassifier]
    >>> meta_model = LogisticRegression
    >>> result = stacking_fit(X_train, y_train, base_models, meta_model)
    """
    # Validate inputs
    validate_inputs(X, y)

    # Initialize results dictionary
    result_dict = {
        'result': None,
        'metrics': {},
        'params_used': {
            'normalizations': normalizations,
            'metrics': metrics
        },
        'warnings': []
    }

    # Set random state if provided
    np.random.seed(random_state)

    # Convert single normalization/metric to list if needed
    if isinstance(normalizations, str):
        normalizations = [normalizations] * len(base_models)
    if isinstance(metrics, (str, Callable)):
        metrics = [metrics] * len(base_models)

    # Check consistency of normalization and metric lists
    if len(normalizations) != len(base_models):
        result_dict['warnings'].append("Number of normalizations doesn't match number of base models")
        normalizations = [normalizations[0]] * len(base_models)
    if len(metrics) != len(base_models):
        result_dict['warnings'].append("Number of metrics doesn't match number of base models")
        metrics = [metrics[0]] * len(base_models)

    # Fit base models and generate meta-features
    meta_features = np.zeros((X.shape[0], len(base_models)))
    base_predictors = []

    for i, (model_func, norm_method, metric) in enumerate(zip(base_models, normalizations, metrics)):
        # Normalize data
        X_norm = normalize_data(X.copy(), norm_method)

        # Fit base model and get predictor function
        predictor = base_model_fit(X_norm, y, model_func)
        base_predictors.append(predictor)

        # Generate meta-features
        meta_features[:, i] = predictor(X_norm)

    # Fit meta-model
    meta_model = meta_model_func()
    meta_model.fit(meta_features, y)
    final_predictor = lambda x: meta_model.predict(
        np.column_stack([predictor(x) for predictor in base_predictors])
    )

    # Store final predictor
    result_dict['result'] = final_predictor

    # Calculate metrics on training data (for demonstration)
    y_pred = final_predictor(X)
    if isinstance(metrics[0], str) or callable(metrics[0]):
        # Use first metric if all are the same
        result_dict['metrics']['final'] = compute_metric(y, y_pred, metrics[0])
    else:
        # Calculate all different metrics
        for i, metric in enumerate(metrics):
            result_dict['metrics'][f'metric_{i}'] = compute_metric(y, y_pred, metric)

    return result_dict

################################################################################
# VotingClassifiers
################################################################################

import numpy as np
from typing import Callable, Dict, List, Optional, Union

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

def compute_metric(y_true: np.ndarray, y_pred: np.ndarray,
                   metric: Union[str, Callable]) -> float:
    """Compute specified metric between true and predicted values."""
    if callable(metric):
        return metric(y_true, y_pred)
    elif metric == 'accuracy':
        return np.mean(y_true == y_pred)
    elif metric == 'logloss':
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def voting_classifiers_fit(
    X: np.ndarray,
    y: np.ndarray,
    classifiers: List[Callable],
    voting_method: str = 'hard',
    normalize: str = 'standard',
    metric: Union[str, Callable] = 'accuracy'
) -> Dict:
    """
    Fit an ensemble of classifiers using voting.

    Parameters:
    - X: Input features (n_samples, n_features)
    - y: Target labels (n_samples,)
    - classifiers: List of callable classifiers with fit/predict methods
    - voting_method: 'hard' or 'soft'
    - normalize: Normalization method ('none', 'standard', 'minmax', 'robust')
    - metric: Evaluation metric

    Returns:
    Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_normalized = normalize_data(X, method=normalize)

    # Initialize results dictionary
    results = {
        'result': None,
        'metrics': {},
        'params_used': {
            'voting_method': voting_method,
            'normalize': normalize,
            'metric': metric
        },
        'warnings': []
    }

    # Fit classifiers
    fitted_classifiers = []
    for clf in classifiers:
        try:
            clf.fit(X_normalized, y)
            fitted_classifiers.append(clf)
        except Exception as e:
            results['warnings'].append(f"Classifier failed to fit: {str(e)}")
            continue

    if not fitted_classifiers:
        raise RuntimeError("All classifiers failed to fit")

    # Make predictions
    if voting_method == 'hard':
        y_preds = np.array([clf.predict(X_normalized) for clf in fitted_classifiers])
        y_ensemble = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=y_preds
        )
    else:  # soft voting
        y_probas = np.array([clf.predict_proba(X_normalized) for clf in fitted_classifiers])
        y_ensemble = np.argmax(np.mean(y_probas, axis=0), axis=1)

    # Compute metric
    results['result'] = y_ensemble
    results['metrics']['score'] = compute_metric(y, y_ensemble, metric)

    return results

################################################################################
# SelfOrganizingMaps
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input contains NaN or infinite values")

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

def compute_distance(X: np.ndarray, Y: np.ndarray, metric: str = 'euclidean', custom_metric: Optional[Callable] = None) -> np.ndarray:
    """Compute distance between data points and map nodes."""
    if custom_metric is not None:
        return custom_metric(X, Y)
    if metric == 'euclidean':
        return np.sqrt(np.sum((X[:, np.newaxis, :] - Y) ** 2, axis=2))
    elif metric == 'manhattan':
        return np.sum(np.abs(X[:, np.newaxis, :] - Y), axis=2)
    elif metric == 'cosine':
        dot_products = np.sum(X[:, np.newaxis, :] * Y, axis=2)
        norms_X = np.linalg.norm(X, axis=1)[:, np.newaxis]
        norms_Y = np.linalg.norm(Y, axis=1)
        return 1 - dot_products / (norms_X * norms_Y + 1e-8)
    elif metric == 'minkowski':
        p = 3
        return np.sum(np.abs(X[:, np.newaxis, :] - Y) ** p, axis=2) ** (1/p)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def initialize_weights(X: np.ndarray, n_nodes: int, init_method: str = 'random') -> np.ndarray:
    """Initialize weights for the SOM."""
    if init_method == 'random':
        return np.random.rand(X.shape[1], n_nodes)
    elif init_method == 'pca':
        _, _, V = np.linalg.svd(X, full_matrices=False)
        return V[:n_nodes].T
    else:
        raise ValueError(f"Unknown initialization method: {init_method}")

def update_weights(X: np.ndarray, distances: np.ndarray, weights: np.ndarray, learning_rate: float, neighborhood_function: Callable) -> np.ndarray:
    """Update weights based on learning rate and neighborhood function."""
    bmu_indices = np.argmin(distances, axis=1)
    for i in range(X.shape[0]):
        bmu = bmu_indices[i]
        influence = neighborhood_function(bmu, distances[i], learning_rate)
        weights[:, bmu] += influence * (X[i] - weights[:, bmu])
    return weights

def gaussian_neighborhood(bmu: int, distances: np.ndarray, learning_rate: float, sigma: float) -> np.ndarray:
    """Gaussian neighborhood function."""
    return learning_rate * np.exp(-np.square(distances - bmu) / (2 * sigma ** 2))

def linear_neighborhood(bmu: int, distances: np.ndarray, learning_rate: float) -> np.ndarray:
    """Linear neighborhood function."""
    return learning_rate * (1 - distances / bmu)

def compute_metrics(X: np.ndarray, weights: np.ndarray, metric: str) -> Dict[str, float]:
    """Compute metrics for the SOM."""
    distances = compute_distance(X, weights, metric)
    bmu_indices = np.argmin(distances, axis=1)
    quantization_error = np.mean(np.min(distances, axis=1))
    topographic_error = np.mean([np.any(bmu_indices == bmu) for bmu in range(weights.shape[1])])
    return {
        'quantization_error': quantization_error,
        'topographic_error': topographic_error
    }

def SelfOrganizingMaps_fit(X: np.ndarray, n_nodes: int = 100, n_iterations: int = 1000,
                          normalization: str = 'standard', metric: str = 'euclidean',
                          custom_metric: Optional[Callable] = None,
                          init_method: str = 'random', learning_rate: float = 0.1,
                          neighborhood_function: Callable = gaussian_neighborhood,
                          sigma: float = 1.0) -> Dict:
    """Fit a Self-Organizing Map to the data."""
    validate_input(X)
    X_normalized = normalize_data(X, normalization)
    weights = initialize_weights(X_normalized, n_nodes, init_method)

    for iteration in range(n_iterations):
        distances = compute_distance(X_normalized, weights, metric, custom_metric)
        learning_rate_iter = learning_rate * (1 - iteration / n_iterations)
        weights = update_weights(X_normalized, distances, weights, learning_rate_iter, neighborhood_function)

    metrics = compute_metrics(X_normalized, weights, metric)
    return {
        'result': weights,
        'metrics': metrics,
        'params_used': {
            'n_nodes': n_nodes,
            'n_iterations': n_iterations,
            'normalization': normalization,
            'metric': metric,
            'init_method': init_method,
            'learning_rate': learning_rate,
            'neighborhood_function': neighborhood_function.__name__ if callable(neighborhood_function) else 'custom',
            'sigma': sigma
        },
        'warnings': []
    }

# Example usage:
# X = np.random.rand(100, 5)
# result = SelfOrganizingMaps_fit(X, n_nodes=20, n_iterations=500)

################################################################################
# RadialBasisFunctions
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate input data for Radial Basis Functions.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,) or (n_samples, n_classes).

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("Inputs must be numpy arrays.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim not in (1, 2):
        raise ValueError("y must be a 1D or 2D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")

def normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Normalize input data using specified method.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    method : str, optional
        Normalization method: 'none', 'standard', 'minmax', 'robust'.

    Returns
    ------
    np.ndarray
        Normalized data.
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

def compute_distance(X: np.ndarray, centers: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """
    Compute distance between samples and centers using specified metric.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    centers : np.ndarray
        Centers array of shape (n_centers, n_features).
    metric : str, optional
        Distance metric: 'euclidean', 'manhattan', 'cosine', 'minkowski'.

    Returns
    ------
    np.ndarray
        Distance matrix of shape (n_samples, n_centers).
    """
    if metric == 'euclidean':
        return np.sqrt(np.sum((X[:, np.newaxis, :] - centers) ** 2, axis=2))
    elif metric == 'manhattan':
        return np.sum(np.abs(X[:, np.newaxis, :] - centers), axis=2)
    elif metric == 'cosine':
        return 1 - np.dot(X, centers.T) / (np.linalg.norm(X, axis=1)[:, np.newaxis] * np.linalg.norm(centers, axis=1))
    elif metric == 'minkowski':
        return np.sum(np.abs(X[:, np.newaxis, :] - centers) ** 3, axis=2) ** (1/3)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def compute_rbf(X: np.ndarray, centers: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Compute Radial Basis Function values.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    centers : np.ndarray
        Centers array of shape (n_centers, n_features).
    gamma : float, optional
        Gamma parameter for RBF.

    Returns
    ------
    np.ndarray
        RBF values of shape (n_samples, n_centers).
    """
    distances = compute_distance(X, centers)
    return np.exp(-gamma * distances ** 2)

def solve_closed_form(Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve for weights using closed form solution.

    Parameters
    ----------
    Phi : np.ndarray
        Design matrix of shape (n_samples, n_centers).
    y : np.ndarray
        Target values array of shape (n_samples,).

    Returns
    ------
    np.ndarray
        Weights array of shape (n_centers,).
    """
    return np.linalg.pinv(Phi) @ y

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, metric: str = 'mse') -> Dict[str, float]:
    """
    Compute specified metrics between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        True target values array of shape (n_samples,).
    y_pred : np.ndarray
        Predicted values array of shape (n_samples,).
    metric : str or callable, optional
        Metric to compute: 'mse', 'mae', 'r2', 'logloss'.

    Returns
    ------
    Dict[str, float]
        Dictionary of computed metrics.
    """
    if callable(metric):
        return {'custom': metric(y_true, y_pred)}

    metrics = {}
    if metric in ['mse', 'mae', 'r2'] or callable(metric):
        if metric == 'mse':
            metrics['mse'] = np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['r2'] = 1 - (ss_res / (ss_tot + 1e-8))
    elif metric == 'logloss':
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        metrics['logloss'] = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

def RadialBasisFunctions_fit(
    X: np.ndarray,
    y: np.ndarray,
    centers: Optional[np.ndarray] = None,
    gamma: float = 1.0,
    normalize_method: str = 'standard',
    distance_metric: str = 'euclidean',
    solver: str = 'closed_form',
    metric: Union[str, Callable] = 'mse',
    **kwargs
) -> Dict:
    """
    Fit Radial Basis Function model to data.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,) or (n_samples, n_classes).
    centers : np.ndarray, optional
        Centers array of shape (n_centers, n_features). If None, use k-means.
    gamma : float, optional
        Gamma parameter for RBF.
    normalize_method : str, optional
        Normalization method: 'none', 'standard', 'minmax', 'robust'.
    distance_metric : str, optional
        Distance metric: 'euclidean', 'manhattan', 'cosine', 'minkowski'.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', 'newton'.
    metric : str or callable, optional
        Metric to compute: 'mse', 'mae', 'r2', 'logloss'.
    **kwargs
        Additional solver-specific parameters.

    Returns
    ------
    Dict
        Dictionary containing:
        - 'result': Fitted model weights.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used for fitting.
        - 'warnings': Any warnings generated during fitting.
    """
    result = {}
    metrics = {}
    params_used = {
        'gamma': gamma,
        'normalize_method': normalize_method,
        'distance_metric': distance_metric,
        'solver': solver
    }
    warnings = []

    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_normalized = normalize_data(X, method=normalize_method)
    params_used['normalization_applied'] = normalize_method

    # Compute RBF features
    if centers is None:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=min(10, X.shape[0] - 1), random_state=42)
        centers = kmeans.fit(X_normalized).cluster_centers_
        warnings.append("Centers were not provided, using k-means to determine them.")
    Phi = compute_rbf(X_normalized, centers, gamma)

    # Solve for weights
    if solver == 'closed_form':
        weights = solve_closed_form(Phi, y)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute predictions and metrics
    y_pred = Phi @ weights
    metrics.update(compute_metrics(y, y_pred, metric))

    return {
        'result': weights,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

# Example usage:
# result = RadialBasisFunctions_fit(X_train, y_train)

################################################################################
# GaussianProcesses
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
    """Normalize input data."""
    if method == 'none':
        return X, y
    elif method == 'standard':
        mean_X = np.mean(X, axis=0)
        std_X = np.std(X, axis=0)
        X_normalized = (X - mean_X) / std_X
        return X_normalized, y
    elif method == 'minmax':
        min_X = np.min(X, axis=0)
        max_X = np.max(X, axis=0)
        X_normalized = (X - min_X) / (max_X - min_X)
        return X_normalized, y
    elif method == 'robust':
        median_X = np.median(X, axis=0)
        iqr_X = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_normalized = (X - median_X) / iqr_X
        return X_normalized, y
    else:
        raise ValueError("Invalid normalization method")

def compute_kernel(X: np.ndarray, X2: Optional[np.ndarray] = None,
                   kernel_func: Callable = lambda x, y: np.exp(-0.5 * np.sum((x - y)**2)),
                   **kernel_params) -> np.ndarray:
    """Compute the kernel matrix."""
    if X2 is None:
        X2 = X
    return np.array([[kernel_func(x, y, **kernel_params) for y in X2] for x in X])

def compute_log_marginal_likelihood(y: np.ndarray, K: np.ndarray,
                                   kernel_func: Callable) -> float:
    """Compute the log marginal likelihood."""
    K_inv = np.linalg.inv(K)
    alpha = np.dot(K_inv, y)
    return -0.5 * (np.dot(y.T, alpha) + np.log(np.linalg.det(K)))

def optimize_hyperparameters(X: np.ndarray, y: np.ndarray,
                            kernel_func: Callable,
                            optimizer: str = 'gradient_descent',
                            tol: float = 1e-3,
                            max_iter: int = 100) -> Dict:
    """Optimize hyperparameters of the kernel function."""
    # Placeholder for optimization logic
    return {'hyperparameters': {}}

def predict(X: np.ndarray, X_train: np.ndarray, y_train: np.ndarray,
            kernel_func: Callable, **kernel_params) -> tuple:
    """Make predictions using Gaussian Processes."""
    K = compute_kernel(X_train, kernel_func=kernel_func, **kernel_params)
    K_s = compute_kernel(X_train, X, kernel_func=kernel_func, **kernel_params)
    K_ss = compute_kernel(X, kernel_func=kernel_func, **kernel_params)

    K_inv = np.linalg.inv(K)
    y_mean = np.dot(np.dot(K_s.T, K_inv), y_train)

    v = np.dot(K_ss - np.dot(np.dot(K_s.T, K_inv), K_s).T, K_inv)
    y_std = np.sqrt(np.diag(v))

    return y_mean, y_std

def GaussianProcesses_fit(X: np.ndarray, y: np.ndarray,
                         kernel_func: Callable = lambda x, y: np.exp(-0.5 * np.sum((x - y)**2)),
                         normalization: str = 'standard',
                         optimizer: str = 'gradient_descent',
                         tol: float = 1e-3,
                         max_iter: int = 100) -> Dict:
    """
    Fit Gaussian Process model to the data.

    Example:
        >>> X = np.random.rand(10, 2)
        >>> y = np.random.rand(10)
        >>> result = GaussianProcesses_fit(X, y)
    """
    validate_inputs(X, y)

    X_norm, y_norm = normalize_data(X, y, method=normalization)

    hyperparams = optimize_hyperparameters(X_norm, y_norm,
                                          kernel_func=kernel_func,
                                          optimizer=optimizer,
                                          tol=tol,
                                          max_iter=max_iter)

    return {
        'result': None,  # Placeholder for actual result
        'metrics': {},
        'params_used': {
            'normalization': normalization,
            'optimizer': optimizer,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

################################################################################
# BayesianNetworks
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data."""
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

def _compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: Union[str, Callable]) -> float:
    """Compute specified metric."""
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

def _fit_bayesian_network(X: np.ndarray, y: np.ndarray,
                         solver: str = "gradient_descent",
                         max_iter: int = 1000,
                         tol: float = 1e-4,
                         learning_rate: float = 0.01) -> Dict:
    """Fit Bayesian Network using specified solver."""
    if solver == "gradient_descent":
        weights = _gradient_descent(X, y, max_iter, tol, learning_rate)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    return {"weights": weights}

def _gradient_descent(X: np.ndarray, y: np.ndarray,
                     max_iter: int = 1000,
                     tol: float = 1e-4,
                     learning_rate: float = 0.01) -> np.ndarray:
    """Gradient descent solver for Bayesian Network."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)

    for _ in range(max_iter):
        gradient = -2 * X.T @ (y - _sigmoid(X @ weights)) / n_samples
        new_weights = weights - learning_rate * gradient

        if np.linalg.norm(new_weights - weights) < tol:
            break
        weights = new_weights

    return weights

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))

def BayesianNetworks_fit(X: np.ndarray, y: np.ndarray,
                        normalization: str = "standard",
                        metric: Union[str, Callable] = "logloss",
                        solver: str = "gradient_descent",
                        max_iter: int = 1000,
                        tol: float = 1e-4,
                        learning_rate: float = 0.01) -> Dict:
    """
    Fit Bayesian Network for non-linear classification.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    normalization : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    metric : str or callable, optional
        Metric to evaluate model performance.
    solver : str, optional
        Solver method ("gradient_descent").
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for stopping criterion.
    learning_rate : float, optional
        Learning rate for gradient descent.

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    _validate_inputs(X, y)
    X_normalized = _normalize_data(X, normalization)

    result = _fit_bayesian_network(X_normalized, y, solver, max_iter, tol, learning_rate)
    y_pred = _sigmoid(X_normalized @ result["weights"])
    metrics = {"metric": _compute_metric(y, y_pred, metric)}

    return {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric,
            "solver": solver,
            "max_iter": max_iter,
            "tol": tol,
            "learning_rate": learning_rate
        },
        "warnings": []
    }

# Example usage:
# X = np.random.rand(100, 5)
# y = np.random.randint(0, 2, size=100)
# result = BayesianNetworks_fit(X, y)
