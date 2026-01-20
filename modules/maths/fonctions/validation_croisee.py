"""
Quantix – Module validation_croisee
Généré automatiquement
Date: 2026-01-09
"""

################################################################################
# k_fold
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(X: np.ndarray, y: np.ndarray, k: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if k <= 1 or not isinstance(k, int):
        raise ValueError("k must be an integer greater than 1")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains NaN or infinite values")
    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("y contains NaN or infinite values")

def _normalize_data(X: np.ndarray, y: np.ndarray,
                   normalization: str = 'none',
                   custom_normalization: Optional[Callable] = None) -> tuple:
    """Normalize data based on user choice."""
    if normalization == 'none':
        return X, y
    elif normalization == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_normalized = (X - X_mean) / X_std
        return X_normalized, y
    elif normalization == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_normalized = (X - X_min) / (X_max - X_min)
        return X_normalized, y
    elif normalization == 'robust':
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_normalized = (X - X_median) / X_iqr
        return X_normalized, y
    elif custom_normalization is not None:
        return custom_normalization(X, y)
    else:
        raise ValueError("Invalid normalization option")

def _compute_metric(y_true: np.ndarray, y_pred: np.ndarray,
                   metric: str = 'mse',
                   custom_metric: Optional[Callable] = None) -> float:
    """Compute metric based on user choice."""
    if custom_metric is not None:
        return custom_metric(y_true, y_pred)

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
        raise ValueError("Invalid metric option")

def _k_fold_split(X: np.ndarray, y: np.ndarray, k: int) -> list:
    """Split data into k folds."""
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append((indices[:start], indices[start:stop]))
        current = stop
    return [(X[train_idx], X[test_idx], y[train_idx], y[test_idx]) for train_idx, test_idx in folds]

def k_fold_fit(X: np.ndarray,
              y: np.ndarray,
              model: Callable,
              k: int = 5,
              normalization: str = 'none',
              metric: str = 'mse',
              custom_normalization: Optional[Callable] = None,
              custom_metric: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation.

    Parameters:
    - X: Input features (numpy array)
    - y: Target values (numpy array)
    - model: Callable model to fit and predict
    - k: Number of folds (default: 5)
    - normalization: Normalization method ('none', 'standard', 'minmax', 'robust')
    - metric: Evaluation metric ('mse', 'mae', 'r2', 'logloss')
    - custom_normalization: Custom normalization function
    - custom_metric: Custom metric function

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings
    """
    _validate_inputs(X, y, k)

    X_norm, y_norm = _normalize_data(X, y, normalization, custom_normalization)
    folds = _k_fold_split(X_norm, y_norm, k)

    results = []
    metrics = []

    for train_idx, test_idx in folds:
        X_train, X_test, y_train, y_test = train_idx, test_idx, y_train, y_test
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metric_value = _compute_metric(y_test, y_pred, metric, custom_metric)
        results.append((X_train, X_test, y_train, y_test))
        metrics.append(metric_value)

    return {
        "result": results,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric,
            "k": k
        },
        "warnings": []
    }

################################################################################
# stratified_k_fold
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union
from collections.abc import Iterable

def _validate_inputs(X: np.ndarray, y: np.ndarray, n_splits: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if n_splits <= 1:
        raise ValueError("n_splits must be greater than 1")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _stratified_split(X: np.ndarray, y: np.ndarray, n_splits: int) -> Iterable[tuple]:
    """Generate stratified k-fold splits."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    for train_index, test_index in skf.split(X, y):
        yield X[train_index], X[test_index], y[train_index], y[test_index]

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute specified metrics."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = float(func(y_true, y_pred))
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def stratified_k_fold_fit(
    X: np.ndarray,
    y: np.ndarray,
    model_func: Callable,
    n_splits: int = 5,
    metric_funcs: Optional[Dict[str, Callable]] = None,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform stratified k-fold cross-validation.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    model_func : Callable
        Function that takes (X_train, y_train) and returns a trained model
    n_splits : int, optional
        Number of folds (default: 5)
    metric_funcs : Dict[str, Callable], optional
        Dictionary of metric names and functions (default: None)
    random_state : int, optional
        Random seed for reproducibility (default: None)

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Set default metrics if none provided
    if metric_funcs is None:
        from sklearn.metrics import mean_squared_error, r2_score
        metric_funcs = {
            'mse': mean_squared_error,
            'r2': r2_score
        }

    # Validate inputs
    _validate_inputs(X, y, n_splits)

    # Initialize results storage
    results = {
        'result': [],
        'metrics': {name: [] for name in metric_funcs.keys()},
        'params_used': {
            'n_splits': n_splits,
            'random_state': random_state
        },
        'warnings': []
    }

    # Perform stratified k-fold cross-validation
    for i, (X_train, X_test, y_train, y_test) in enumerate(_stratified_split(X, y, n_splits)):
        try:
            # Train model
            model = model_func(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Compute metrics
            fold_metrics = _compute_metrics(y_test, y_pred, metric_funcs)
            results['result'].append({
                'fold': i,
                'train_indices': np.where(X == X_train)[0],
                'test_indices': np.where(X == X_test)[0]
            })
            for name, value in fold_metrics.items():
                results['metrics'][name].append(value)

        except Exception as e:
            results['warnings'].append(f"Error in fold {i}: {str(e)}")

    # Calculate mean metrics across folds
    for name in results['metrics']:
        results['metrics'][name] = np.mean(results['metrics'][name])

    return results

# Example usage:
"""
def example_model(X_train, y_train):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

X_example = np.random.rand(100, 5)
y_example = np.random.randint(0, 2, size=100)

results = stratified_k_fold_fit(
    X_example,
    y_example,
    model_func=example_model
)
"""

################################################################################
# leave_one_out
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def leave_one_out_fit(
    X: np.ndarray,
    y: np.ndarray,
    model_func: Callable[[np.ndarray, np.ndarray], Any],
    metric_func: Callable[[np.ndarray, np.ndarray], float] = None,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform leave-one-out cross-validation.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    model_func : Callable[[np.ndarray, np.ndarray], Any]
        Function that fits a model and returns predictions.
    metric_func : Callable[[np.ndarray, np.ndarray], float], optional
        Function to compute the metric. If None, uses mean squared error.
    normalizer : Callable[[np.ndarray], np.ndarray], optional
        Function to normalize the features. If None, no normalization is applied.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    n_samples = X.shape[0]
    metrics = []
    predictions = np.zeros_like(y)
    warnings_list = []

    for i in range(n_samples):
        # Split data
        X_train, X_test = np.delete(X, i, axis=0), X[[i], :]
        y_train, y_test = np.delete(y, i), y[i]

        # Normalize if needed
        if normalizer is not None:
            X_train = normalizer(X_train)
            X_test = normalizer(X_test)

        # Fit model and predict
        try:
            model = model_func(X_train, y_train)
            pred = model.predict(X_test) if hasattr(model, 'predict') else model_func(X_test, y_train)
            predictions[i] = pred
        except Exception as e:
            warnings_list.append(f"Error in fold {i}: {str(e)}")
            continue

        # Compute metric
        if metric_func is None:
            metric = _mean_squared_error(y_test, pred)
        else:
            metric = metric_func(y_test, pred)

        metrics.append(metric)

    result = {
        "result": predictions,
        "metrics": np.mean(metrics) if metrics else None,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric_func": metric_func.__name__ if metric_func else "mse"
        },
        "warnings": warnings_list
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

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

# Example usage:
"""
from sklearn.linear_model import LinearRegression

X = np.random.rand(10, 5)
y = np.random.rand(10)

def model_func(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

result = leave_one_out_fit(X, y, model_func)
print(result)
"""

################################################################################
# train_test_split
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(X: np.ndarray, y: Optional[np.ndarray] = None,
                    test_size: float = 0.25, random_state: Optional[int] = None) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")

def _split_data(X: np.ndarray, y: Optional[np.ndarray] = None,
               test_size: float = 0.25, random_state: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Split data into train and test sets."""
    rng = np.random.RandomState(random_state)
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)

    split_idx = int((1 - test_size) * X.shape[0])
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = None, None
    if y is not None:
        y_train, y_test = y[train_indices], y[test_indices]

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute specified metrics."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(y_true, y_pred)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def train_test_split_fit(X: np.ndarray, y: Optional[np.ndarray] = None,
                        test_size: float = 0.25, random_state: Optional[int] = None,
                        metric_funcs: Dict[str, Callable] = None) -> Dict[str, Any]:
    """
    Split data into train and test sets with optional metric computation.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target vector of shape (n_samples,) or None
    test_size : float, default=0.25
        Proportion of data to use for testing (0 < test_size < 1)
    random_state : Optional[int]
        Random seed for reproducibility
    metric_funcs : Dict[str, Callable]
        Dictionary of metric functions to compute on test set

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - 'result': Split data
        - 'metrics': Computed metrics (if any)
        - 'params_used': Parameters used
        - 'warnings': Any warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> metrics = {'mse': lambda y_true, y_pred: np.mean((y_true - y_pred)**2)}
    >>> result = train_test_split_fit(X, y, test_size=0.3, metric_funcs=metrics)
    """
    # Validate inputs
    _validate_inputs(X, y, test_size, random_state)

    # Split data
    split_result = _split_data(X, y, test_size, random_state)

    # Compute metrics if provided
    metrics = {}
    warnings = []
    if y is not None and metric_funcs is not None:
        try:
            metrics = _compute_metrics(split_result['y_test'], split_result.get('y_pred', np.zeros_like(split_result['y_test'])), metric_funcs)
        except Exception as e:
            warnings.append(f"Metric computation failed: {str(e)}")

    return {
        'result': split_result,
        'metrics': metrics,
        'params_used': {
            'test_size': test_size,
            'random_state': random_state
        },
        'warnings': warnings
    }

################################################################################
# time_series_split
################################################################################

import numpy as np
from typing import Callable, Dict, List, Optional, Union

def time_series_split_fit(
    X: np.ndarray,
    n_splits: int = 5,
    test_size: Optional[int] = None,
    max_train_size: Optional[int] = None,
    shuffle: bool = False,
    random_state: Optional[int] = None
) -> Dict[str, Union[List[np.ndarray], List[np.ndarray], Dict]]:
    """
    Perform time series cross-validation split.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    n_splits : int, default=5
        Number of splits.
    test_size : int, optional
        Size of the test set. If None, it is inferred from n_splits.
    max_train_size : int, optional
        Maximum number of training samples. If None, no limit is applied.
    shuffle : bool, default=False
        Whether to shuffle the data before splitting. Not recommended for time series.
    random_state : int, optional
        Random seed for reproducibility if shuffle is True.

    Returns
    -------
    Dict[str, Union[List[np.ndarray], List[np.ndarray], Dict]]
        Dictionary containing:
        - "train_indices": List of training indices for each split
        - "test_indices": List of test indices for each split
        - "params_used": Dictionary of parameters used

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> splits = time_series_split_fit(X, n_splits=3)
    """
    # Validate inputs
    _validate_inputs(X, n_splits, test_size, max_train_size)

    # Set random state if shuffling
    rng = np.random.RandomState(random_state) if shuffle else None

    # Determine test size
    n_samples = X.shape[0]
    if test_size is None:
        test_size = max(1, n_samples // n_splits)

    # Generate splits
    train_indices = []
    test_indices = []

    for i in range(n_splits):
        # Calculate split indices
        start_train = 0 if i == 0 else (i * n_samples // n_splits)
        end_train = start_train + (n_samples - i * test_size) if max_train_size is None else min(
            start_train + max_train_size, n_samples - test_size
        )
        end_test = end_train + test_size

        # Shuffle if needed
        if shuffle:
            indices = np.arange(n_samples)
            rng.shuffle(indices)
            train_idx = indices[start_train:end_train]
            test_idx = indices[end_train:end_test]
        else:
            train_idx = np.arange(start_train, end_train)
            test_idx = np.arange(end_train, end_test)

        train_indices.append(train_idx)
        test_indices.append(test_idx)

    return {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "params_used": {
            "n_splits": n_splits,
            "test_size": test_size,
            "max_train_size": max_train_size,
            "shuffle": shuffle,
            "random_state": random_state
        }
    }

def _validate_inputs(
    X: np.ndarray,
    n_splits: int,
    test_size: Optional[int],
    max_train_size: Optional[int]
) -> None:
    """Validate input parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if n_splits <= 0:
        raise ValueError("n_splits must be positive")
    if test_size is not None and (test_size <= 0 or test_size >= X.shape[0]):
        raise ValueError("test_size must be between 1 and n_samples-1")
    if max_train_size is not None and (max_train_size <= 0 or max_train_size >= X.shape[0]):
        raise ValueError("max_train_size must be between 1 and n_samples-1")
    if max_train_size is not None and test_size is not None:
        if (max_train_size + test_size) > X.shape[0]:
            raise ValueError("max_train_size + test_size must be <= n_samples")

################################################################################
# shuffle_split
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def shuffle_split_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Perform a shuffle split cross-validation.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    n_splits : int, optional
        Number of splits. Default is 5.
    test_size : float, optional
        Proportion of the dataset to include in the test split. Default is 0.2.
    random_state : Optional[int], optional
        Seed for random number generation. Default is None.
    shuffle : bool, optional
        Whether to shuffle the data before splitting. Default is True.
    stratify : Optional[np.ndarray], optional
        If not None, data is split in a stratified fashion using this as the class labels.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the results of the shuffle split.
    """
    # Validate inputs
    _validate_inputs(X, y, test_size, n_splits)

    # Initialize random number generator
    rng = np.random.RandomState(random_state)

    # Generate splits
    splits = _generate_splits(X, y, n_splits, test_size, shuffle, stratify, rng)

    # Prepare output
    result = {
        "result": splits,
        "metrics": {},
        "params_used": {
            "n_splits": n_splits,
            "test_size": test_size,
            "random_state": random_state,
            "shuffle": shuffle,
            "stratify": stratify is not None
        },
        "warnings": []
    }

    return result

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    n_splits: int
) -> None:
    """
    Validate the inputs for shuffle split.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    test_size : float
        Proportion of the dataset to include in the test split.
    n_splits : int
        Number of splits.

    Raises:
    -------
    ValueError
        If inputs are invalid.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("X and y must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("X and y must not contain infinite values.")

def _generate_splits(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    test_size: float,
    shuffle: bool,
    stratify: Optional[np.ndarray],
    rng: np.random.RandomState
) -> Dict[str, Any]:
    """
    Generate the splits for shuffle split.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    n_splits : int
        Number of splits.
    test_size : float
        Proportion of the dataset to include in the test split.
    shuffle : bool
        Whether to shuffle the data before splitting.
    stratify : Optional[np.ndarray]
        If not None, data is split in a stratified fashion using this as the class labels.
    rng : np.random.RandomState
        Random number generator.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the splits.
    """
    n_samples = X.shape[0]
    test_size_int = int(test_size * n_samples)

    if stratify is not None:
        classes, y_indices = np.unique(stratify, return_inverse=True)
        n_classes = len(classes)

    splits = []
    for _ in range(n_splits):
        if shuffle:
            indices = rng.permutation(n_samples)
            X_shuffled, y_shuffled = X[indices], y[indices]
        else:
            X_shuffled, y_shuffled = X, y

        if stratify is not None:
            test_indices = []
            for cls in classes:
                cls_indices = np.where(y_indices == np.where(classes == cls)[0][0])[0]
                test_indices.extend(rng.choice(cls_indices, size=int(test_size_int / n_classes), replace=False))
            train_indices = np.setdiff1d(np.arange(n_samples), test_indices)
        else:
            test_indices = rng.choice(n_samples, size=test_size_int, replace=False)
            train_indices = np.setdiff1d(np.arange(n_samples), test_indices)

        splits.append({
            "train_indices": train_indices,
            "test_indices": test_indices
        })

    return splits

################################################################################
# group_k_fold
################################################################################

import numpy as np
from typing import Callable, Dict, List, Optional, Union, Any
from sklearn.base import BaseEstimator

def validate_groups(y: np.ndarray, groups: np.ndarray) -> None:
    """Validate input arrays for group k-fold."""
    if len(y) != len(groups):
        raise ValueError("y and groups must have the same length")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or infinite values")
    if not np.all(np.isfinite(groups)):
        raise ValueError("groups contains NaN or infinite values")

def _group_k_fold_indices(n_splits: int, y: np.ndarray, groups: np.ndarray) -> Any:
    """Generate indices for group k-fold cross-validation."""
    from sklearn.model_selection import GroupKFold
    return GroupKFold(n_splits=n_splits).split(y, groups=groups)

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: List[Union[str, Callable[[np.ndarray, np.ndarray], float]]]
) -> Dict[str, float]:
    """Compute specified metrics between true and predicted values."""
    metric_results = {}

    for metric in metrics:
        if isinstance(metric, str):
            if metric == 'mse':
                score = np.mean((y_true - y_pred) ** 2)
            elif metric == 'mae':
                score = np.mean(np.abs(y_true - y_pred))
            elif metric == 'r2':
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
            elif metric == 'logloss':
                score = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            else:
                raise ValueError(f"Unknown metric: {metric}")
        elif callable(metric):
            score = metric(y_true, y_pred)
        else:
            raise ValueError("Metric must be either a string or callable")

        metric_results[str(metric)] = score

    return metric_results

def group_k_fold_fit(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    metrics: List[Union[str, Callable[[np.ndarray, np.ndarray], float]]] = ['mse', 'r2'],
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform group k-fold cross-validation.

    Parameters:
    -----------
    estimator : BaseEstimator
        The model to evaluate.
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    groups : np.ndarray
        Group labels for the samples.
    n_splits : int, optional
        Number of folds (default=5).
    metrics : List[Union[str, Callable]], optional
        Metrics to compute (default=['mse', 'r2']).
    random_state : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    validate_groups(y, groups)

    # Initialize results storage
    results = {
        'result': [],
        'metrics': {metric: [] for metric in metrics},
        'params_used': estimator.get_params(),
        'warnings': []
    }

    # Generate folds
    fold_generator = _group_k_fold_indices(n_splits, y, groups)

    # Perform cross-validation
    for train_idx, test_idx in fold_generator:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit estimator
        estimator.fit(X_train, y_train)

        # Predict and compute metrics
        y_pred = estimator.predict(X_test)
        fold_metrics = compute_metrics(y_test, y_pred, metrics)

        # Store results
        results['result'].append({
            'train_indices': train_idx,
            'test_indices': test_idx,
            'predictions': y_pred
        })

        for metric_name, score in fold_metrics.items():
            results['metrics'][metric_name].append(score)

    # Calculate mean metrics across folds
    for metric_name in results['metrics']:
        results['metrics'][metric_name] = np.mean(results['metrics'][metric_name])

    return results

# Example usage:
"""
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.random.rand(100, 5)
y = np.random.rand(100)
groups = np.random.randint(0, 20, size=100)

estimator = LinearRegression()
result = group_k_fold_fit(estimator, X, y, groups)
"""

################################################################################
# repeated_k_fold
################################################################################

import numpy as np
from typing import Callable, Dict, List, Optional, Union, Any
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: Optional[int] = None
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if n_splits <= 1:
        raise ValueError("n_splits must be greater than 1")
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive")
    if random_state is not None and (not isinstance(random_state, int) or random_state < 0):
        raise ValueError("random_state must be None or a positive integer")

def repeated_k_fold_fit(
    X: np.ndarray,
    y: np.ndarray,
    estimator: BaseEstimator,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: Optional[int] = None,
    scoring: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'accuracy',
    normalize: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    pre_dispatch: int = 1
) -> Dict[str, Any]:
    """
    Perform repeated k-fold cross-validation.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    estimator : BaseEstimator
        The model to evaluate
    n_splits : int, default=5
        Number of folds
    n_repeats : int, default=10
        Number of times cross-validator will be repeated
    random_state : Optional[int], default=None
        Controls the random resampling of the data
    scoring : Union[str, Callable], default='accuracy'
        Scoring method to evaluate the predictions
    normalize : Optional[Callable], default=None
        Function to normalize features before fitting
    pre_dispatch : int, default=1
        Controls the number of jobs to dispatch

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings
    """
    validate_inputs(X, y, n_splits, n_repeats, random_state)

    # Initialize results storage
    results = {
        'result': [],
        'metrics': [],
        'params_used': {
            'n_splits': n_splits,
            'n_repeats': n_repeats,
            'random_state': random_state
        },
        'warnings': []
    }

    # Set up repeated k-fold cross-validation
    rkf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for i in range(n_repeats):
        # Generate folds
        fold_results = []
        for train_index, test_index in rkf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Apply normalization if specified
            X_train_norm = normalize(X_train) if normalize else X_train
            X_test_norm = normalize(X_test) if normalize else X_test

            # Fit estimator and predict
            estimator.fit(X_train_norm, y_train)
            y_pred = estimator.predict(X_test_norm)

            # Calculate score
            if callable(scoring):
                score = scoring(y_test, y_pred)
            else:
                # In a real implementation, you would use sklearn's scoring here
                score = np.mean(y_pred == y_test)  # placeholder for accuracy

            fold_results.append(score)

        results['result'].append(fold_results)
        results['metrics'].append({
            'mean': np.mean(fold_results),
            'std': np.std(fold_results)
        })

    return results

def standard_normalize(X: np.ndarray) -> np.ndarray:
    """Standard normalization (z-score)."""
    return (X - X.mean(axis=0)) / X.std(axis=0)

def minmax_normalize(X: np.ndarray) -> np.ndarray:
    """Min-max normalization."""
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

def robust_normalize(X: np.ndarray) -> np.ndarray:
    """Robust normalization using median and IQR."""
    return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))

# Example usage:
"""
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, size=100)

from sklearn.linear_model import LogisticRegression

results = repeated_k_fold_fit(
    X=X,
    y=y,
    estimator=LogisticRegression(),
    n_splits=5,
    n_repeats=3,
    random_state=42,
    scoring='accuracy',
    normalize=standard_normalize
)
"""

################################################################################
# stratified_shuffle_split
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union, List
from collections.abc import Iterable

def stratified_shuffle_split_fit(
    X: Union[np.ndarray, List],
    y: Union[np.ndarray, List],
    n_splits: int = 5,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True
) -> Dict[str, Any]:
    """
    Perform stratified shuffle split on the input data.

    Parameters:
    -----------
    X : Union[np.ndarray, List]
        Input features.
    y : Union[np.ndarray, List]
        Target labels.
    n_splits : int, optional
        Number of splits (default is 5).
    test_size : float, optional
        Proportion of the dataset to include in the test split (default is 0.2).
    random_state : Optional[int], optional
        Random seed for reproducibility (default is None).
    shuffle : bool, optional
        Whether or not to shuffle the data before splitting (default is True).

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the splits and other relevant information.

    Examples:
    ---------
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([0, 1, 0, 1])
    >>> result = stratified_shuffle_split_fit(X, y)
    """
    # Validate inputs
    X, y = _validate_inputs(X, y)

    # Check test_size
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    # Initialize random state
    rng = np.random.RandomState(random_state)

    # Get unique classes and their counts
    classes, y_indices = np.unique(y, return_inverse=True)
    n_classes = len(classes)

    # Calculate number of samples per class in each split
    n_samples_per_class = _calculate_n_samples_per_class(y, test_size, n_splits)

    # Generate splits
    splits = _generate_stratified_splits(X, y_indices, n_samples_per_class, shuffle, rng)

    return {
        "result": splits,
        "params_used": {
            "n_splits": n_splits,
            "test_size": test_size,
            "random_state": random_state,
            "shuffle": shuffle
        },
        "warnings": []
    }

def _validate_inputs(X: Union[np.ndarray, List], y: Union[np.ndarray, List]) -> tuple:
    """
    Validate the input features and target labels.

    Parameters:
    -----------
    X : Union[np.ndarray, List]
        Input features.
    y : Union[np.ndarray, List]
        Target labels.

    Returns:
    --------
    tuple
        Validated X and y as numpy arrays.
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")

    if len(X.shape) != 2:
        raise ValueError("X must be a 2D array")

    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")

    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

    return X, y

def _calculate_n_samples_per_class(
    y: np.ndarray,
    test_size: float,
    n_splits: int
) -> Dict[int, List[int]]:
    """
    Calculate the number of samples per class in each split.

    Parameters:
    -----------
    y : np.ndarray
        Target labels.
    test_size : float
        Proportion of the dataset to include in the test split.
    n_splits : int
        Number of splits.

    Returns:
    --------
    Dict[int, List[int]]
        A dictionary mapping each class to a list of sample counts per split.
    """
    classes, y_indices = np.unique(y, return_inverse=True)
    n_classes = len(classes)

    # Calculate the number of samples per class in each split
    n_samples_per_class = {}
    for i, cls in enumerate(classes):
        n_samples_cls = np.sum(y == cls)
        n_test_samples = int(test_size * n_samples_cls / n_splits)

        # Ensure at least one sample per class in each split
        if n_test_samples < 1:
            n_test_samples = 1

        # Distribute samples evenly across splits
        remainder = n_samples_cls - n_test_samples * n_splits
        samples_per_split = [n_test_samples] * n_splits

        for i in range(remainder):
            samples_per_split[i] += 1

        n_samples_per_class[cls] = samples_per_split

    return n_samples_per_class

def _generate_stratified_splits(
    X: np.ndarray,
    y_indices: np.ndarray,
    n_samples_per_class: Dict[int, List[int]],
    shuffle: bool,
    rng: np.random.RandomState
) -> List[Dict[str, Any]]:
    """
    Generate stratified splits of the data.

    Parameters:
    -----------
    X : np.ndarray
        Input features.
    y_indices : np.ndarray
        Indices of target labels.
    n_samples_per_class : Dict[int, List[int]]
        Number of samples per class in each split.
    shuffle : bool
        Whether or not to shuffle the data before splitting.
    rng : np.random.RandomState
        Random state for reproducibility.

    Returns:
    --------
    List[Dict[str, Any]]
        A list of dictionaries containing the train and test indices for each split.
    """
    classes = np.unique(y_indices)
    n_classes = len(classes)

    # Initialize splits
    splits = []

    for _ in range(len(n_samples_per_class[classes[0]])):
        train_indices = []
        test_indices = []

        for cls in classes:
            # Get indices of samples belonging to the current class
            cls_indices = np.where(y_indices == cls)[0]

            # Shuffle indices if required
            if shuffle:
                rng.shuffle(cls_indices)

            # Calculate the number of test samples for this class in this split
            n_test_samples = n_samples_per_class[cls][_]

            # Split indices into train and test
            test_idx = cls_indices[:n_test_samples]
            train_idx = cls_indices[n_test_samples:]

            # Append to the respective lists
            test_indices.extend(test_idx)
            train_indices.extend(train_idx)

        # Shuffle the indices if required
        if shuffle:
            rng.shuffle(train_indices)
            rng.shuffle(test_indices)

        # Append the split to the list
        splits.append({
            "train": train_indices,
            "test": test_indices
        })

    return splits

################################################################################
# cross_val_score
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Union, Optional

def cross_val_score_fit(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: Union[str, Callable] = 'mse',
    normalize: str = 'none',
    random_state: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform cross-validation and return evaluation scores.

    Parameters:
    -----------
    estimator : Any
        The model to evaluate.
    X : np.ndarray
        Training data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    cv : int, default=5
        Number of folds for cross-validation.
    scoring : str or callable, default='mse'
        Scoring metric. Can be 'mse', 'mae', 'r2', or a custom callable.
    normalize : str, default='none'
        Normalization method. Can be 'none', 'standard', 'minmax', or 'robust'.
    random_state : int, optional
        Random seed for reproducibility.
    **kwargs :
        Additional parameters passed to the estimator.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples:
    ---------
    >>> from sklearn.linear_model import LinearRegression
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> model = LinearRegression()
    >>> result = cross_val_score_fit(model, X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_normalized = _normalize_data(X, method=normalize)

    # Split data into folds
    fold_sizes = np.full(cv, len(X) // cv, dtype=int)
    fold_sizes[:len(X) % cv] += 1
    indices = np.arange(len(X))
    np.random.seed(random_state)
    np.random.shuffle(indices)

    # Initialize results
    metrics = []
    warnings = []

    for i in range(cv):
        # Split into train and test
        test_indices = indices[np.sum(fold_sizes[:i]):np.sum(fold_sizes[:i+1])]
        train_indices = np.setdiff1d(indices, test_indices)

        X_train, y_train = X_normalized[train_indices], y[train_indices]
        X_test, y_test = X_normalized[test_indices], y[test_indices]

        try:
            # Fit estimator
            estimator.fit(X_train, y_train)

            # Predict and compute metric
            y_pred = estimator.predict(X_test)
            metric_value = _compute_metric(y_test, y_pred, scoring)

            metrics.append(metric_value)
        except Exception as e:
            warnings.append(f"Fold {i+1} failed: {str(e)}")

    # Prepare results
    result = {
        "result": np.mean(metrics) if metrics else None,
        "metrics": metrics,
        "params_used": {
            "cv": cv,
            "scoring": scoring,
            "normalize": normalize,
            "random_state": random_state
        },
        "warnings": warnings if warnings else None
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _normalize_data(X: np.ndarray, method: str = 'none') -> np.ndarray:
    """Normalize data based on the specified method."""
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

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> float:
    """Compute the specified metric between true and predicted values."""
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
    else:
        raise ValueError(f"Unknown metric: {metric}")

################################################################################
# cross_validate
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Union, Optional

def cross_validate_fit(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: int = 5,
    scoring: Union[str, Callable] = 'mse',
    normalize: str = 'none',
    solver: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform cross-validation on a given model.

    Parameters:
    -----------
    model : Callable
        The model to validate. Must have fit and predict methods.
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    cv_splits : int, optional
        Number of cross-validation splits (default: 5).
    scoring : str or Callable, optional
        Scoring metric ('mse', 'mae', 'r2', etc.) or custom callable (default: 'mse').
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'none').
    solver : str, optional
        Solver method for the model (default: None).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_normalized = _apply_normalization(X, normalize)

    # Initialize results dictionary
    results = {
        'result': None,
        'metrics': {},
        'params_used': {
            'cv_splits': cv_splits,
            'scoring': scoring,
            'normalize': normalize,
            'solver': solver
        },
        'warnings': []
    }

    # Perform cross-validation
    metrics = _cross_validate_compute(
        model,
        X_normalized,
        y,
        cv_splits,
        scoring,
        solver,
        **kwargs
    )

    results['metrics'] = metrics

    return results

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate input data dimensions and types.

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

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """
    Apply normalization to the feature matrix.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix.
    method : str
        Normalization method.

    Returns:
    --------
    np.ndarray
        Normalized feature matrix.
    """
    if method == 'standard':
        X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        X_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        X_normalized = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        X_normalized = X
    return X_normalized

def _cross_validate_compute(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: int,
    scoring: Union[str, Callable],
    solver: Optional[str],
    **kwargs
) -> Dict[str, float]:
    """
    Compute cross-validation metrics.

    Parameters:
    -----------
    model : Callable
        The model to validate.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    cv_splits : int
        Number of cross-validation splits.
    scoring : str or Callable
        Scoring metric.
    solver : str, optional
        Solver method for the model.

    Returns:
    --------
    Dict[str, float]
        Dictionary of computed metrics.
    """
    n_samples = X.shape[0]
    fold_size = n_samples // cv_splits
    metrics = {}

    for i in range(cv_splits):
        # Split data into training and validation sets
        val_indices = range(i * fold_size, (i + 1) * fold_size)
        train_indices = list(set(range(n_samples)) - set(val_indices))

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        # Fit the model
        if solver:
            model = model(solver=solver, **kwargs)
        else:
            model = model(**kwargs)

        model.fit(X_train, y_train)

        # Predict and compute metrics
        y_pred = model.predict(X_val)
        metric_value = _compute_metric(y_val, y_pred, scoring)

        metrics[f'fold_{i}'] = metric_value

    # Compute average metric
    avg_metric = np.mean(list(metrics.values()))
    metrics['mean'] = avg_metric

    return metrics

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> float:
    """
    Compute the specified metric.

    Parameters:
    -----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.
    metric : str or Callable
        Metric to compute.

    Returns:
    --------
    float
        Computed metric value.
    """
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

################################################################################
# learning_curve
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray],
    metric: Callable[[np.ndarray, np.ndarray], float]
) -> None:
    """Validate input data and parameters."""
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values.")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or infinite values.")
    if np.any(train_sizes <= 0) or np.any(train_sizes > X.shape[0]):
        raise ValueError("train_sizes must be between 0 and the number of samples.")
    if not callable(normalizer):
        raise TypeError("normalizer must be a callable.")
    if not callable(metric):
        raise TypeError("metric must be a callable.")

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Compute metrics for the given true and predicted values."""
    return {"metric": metric(y_true, y_pred)}

def _normalize_data(
    X: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Normalize the input data using the provided normalizer."""
    return normalizer(X)

def _learning_curve_step(
    X: np.ndarray,
    y: np.ndarray,
    train_size: int,
    normalizer: Callable[[np.ndarray], np.ndarray],
    metric: Callable[[np.ndarray, np.ndarray], float],
    model_fit: Callable[[np.ndarray, np.ndarray], Any],
    model_predict: Callable[[Any, np.ndarray], np.ndarray]
) -> Dict[str, Any]:
    """Compute a single point on the learning curve."""
    indices = np.random.choice(X.shape[0], train_size, replace=False)
    X_train, y_train = X[indices], y[indices]
    X_train_norm = _normalize_data(X_train, normalizer)
    model = model_fit(X_train_norm, y_train)
    X_test_norm = _normalize_data(X, normalizer)
    y_pred = model_predict(model, X_test_norm)
    metrics = _compute_metrics(y, y_pred, metric)
    return {"train_size": train_size, "metrics": metrics}

def learning_curve_fit(
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 5),
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: Callable[[np.ndarray, np.ndarray], float] = lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
    model_fit: Callable[[np.ndarray, np.ndarray], Any] = lambda X, y: None,
    model_predict: Callable[[Any, np.ndarray], np.ndarray] = lambda model, X: np.zeros(X.shape[0]),
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute the learning curve for a given model.

    Parameters:
    -----------
    X : np.ndarray
        Input features.
    y : np.ndarray
        Target values.
    train_sizes : np.ndarray, optional
        Array of training set sizes to evaluate.
    normalizer : Callable[[np.ndarray], np.ndarray], optional
        Function to normalize the input data.
    metric : Callable[[np.ndarray, np.ndarray], float], optional
        Function to compute the evaluation metric.
    model_fit : Callable[[np.ndarray, np.ndarray], Any], optional
        Function to fit the model.
    model_predict : Callable[[Any, np.ndarray], np.ndarray], optional
        Function to predict using the model.
    random_state : Optional[int], optional
        Random seed for reproducibility.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the results of the learning curve.
    """
    if random_state is not None:
        np.random.seed(random_state)

    _validate_inputs(X, y, train_sizes, normalizer, metric)

    results = []
    for size in train_sizes:
        result = _learning_curve_step(
            X, y, int(size), normalizer, metric, model_fit, model_predict
        )
        results.append(result)

    return {
        "result": results,
        "metrics": ["metric"],
        "params_used": {
            "train_sizes": train_sizes.tolist(),
            "normalizer": normalizer.__name__ if hasattr(normalizer, '__name__') else str(normalizer),
            "metric": metric.__name__ if hasattr(metric, '__name__') else str(metric),
            "model_fit": model_fit.__name__ if hasattr(model_fit, '__name__') else str(model_fit),
            "model_predict": model_predict.__name__ if hasattr(model_predict, '__name__') else str(model_predict),
            "random_state": random_state
        },
        "warnings": []
    }

# Example usage:
if __name__ == "__main__":
    # Example data
    X = np.random.rand(100, 5)
    y = np.random.rand(100)

    # Example model functions
    def example_fit(X, y):
        return np.mean(X, axis=0)

    def example_predict(model, X):
        return model + np.random.rand(X.shape[0]) * 0.1

    # Compute learning curve
    result = learning_curve_fit(
        X, y,
        train_sizes=np.linspace(0.1, 1.0, 5),
        normalizer=lambda x: (x - np.mean(x, axis=0)) / np.std(x, axis=0),
        metric=lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)),
        model_fit=example_fit,
        model_predict=example_predict
    )

################################################################################
# validation_curve
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    param_range: np.ndarray,
    cv: int = 5,
    scoring: Optional[Callable] = None
) -> None:
    """Validate input data and parameters."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if cv <= 0:
        raise ValueError("cv must be positive")
    if len(param_range) < 2:
        raise ValueError("param_range must contain at least 2 values")

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scoring: Callable
) -> float:
    """Compute the specified metric."""
    return scoring(y_true, y_pred)

def _split_data(
    X: np.ndarray,
    y: np.ndarray,
    cv: int
) -> list:
    """Split data into cross-validation folds."""
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    fold_sizes = np.full(cv, n_samples // cv, dtype=int)
    fold_sizes[:n_samples % cv] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append((indices[:start], indices[start:stop]))
        current = stop
    return [(X[train], X[test], y[train], y[test]) for train, test in folds]

def _fit_and_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    estimator: Callable,
    param_value: float
) -> Dict[str, Any]:
    """Fit estimator and compute score for a given parameter value."""
    model = estimator(param_value)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        'model': model,
        'y_pred': y_pred
    }

def validation_curve_compute(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Callable,
    param_name: str,
    param_range: np.ndarray,
    cv: int = 5,
    scoring: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Compute cross-validated scores for parameter values.

    Parameters:
    -----------
    X : np.ndarray
        Training data of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    estimator : Callable
        Estimator object that implements fit and predict
    param_name : str
        Name of the parameter to be varied
    param_range : np.ndarray
        Array of parameter values to be tested
    cv : int, optional
        Number of cross-validation folds (default=5)
    scoring : Callable, optional
        Scoring function to evaluate the predictions (default=None)

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings
    """
    _validate_inputs(X, y, param_range, cv, scoring)

    if scoring is None:
        raise ValueError("scoring must be provided")

    folds = _split_data(X, y, cv)
    results = []
    warnings_list = []

    for param_value in param_range:
        fold_scores = []
        for X_train, X_test, y_train, y_test in folds:
            try:
                fit_result = _fit_and_score(X_train, y_train, X_test, y_test, estimator, param_value)
                score = _compute_metric(y_test, fit_result['y_pred'], scoring)
                fold_scores.append(score)
            except Exception as e:
                warnings_list.append(f"Error for param={param_value}: {str(e)}")
                fold_scores.append(np.nan)

        results.append({
            'param_value': param_value,
            'mean_score': np.mean(fold_scores),
            'std_score': np.std(fold_scores)
        })

    return {
        'result': results,
        'metrics': {'scoring': scoring.__name__},
        'params_used': {
            'param_name': param_name,
            'param_range': param_range.tolist(),
            'cv': cv
        },
        'warnings': warnings_list
    }

# Example usage:
"""
from sklearn.linear_model import Ridge

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

X = np.random.rand(100, 5)
y = np.random.rand(100)

result = validation_curve_compute(
    X=X,
    y=y,
    estimator=lambda alpha: Ridge(alpha=alpha),
    param_name='alpha',
    param_range=np.logspace(-3, 3, 7),
    cv=5,
    scoring=mse
)
"""
