"""
Quantix – Module equilibrage_classes
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# undersampling
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List
from sklearn.utils.validation import check_X_y, check_array

def undersampling_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sampling_strategy: Union[str, Dict[int, float]] = 'auto',
    random_state: Optional[int] = None,
    replacement: bool = False,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = 'euclidean',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Callable[[np.ndarray, np.ndarray], float] = 'accuracy',
    **kwargs
) -> Dict:
    """
    Perform undersampling on a dataset to balance class distribution.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    sampling_strategy : str or dict, default='auto'
        Sampling strategy to use. If 'auto', uses majority class as reference.
    random_state : int, optional
        Random seed for reproducibility
    replacement : bool, default=False
        Whether to sample with replacement
    distance_metric : callable or str, default='euclidean'
        Distance metric to use for sampling
    normalizer : callable, optional
        Normalization function to apply before sampling
    metric : callable or str, default='accuracy'
        Metric to evaluate the sampling result
    **kwargs :
        Additional keyword arguments for specific sampling methods

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': undersampled data (X_res, y_res)
        - 'metrics': evaluation metrics
        - 'params_used': parameters used
        - 'warnings': any warnings generated

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = undersampling_fit(X, y, sampling_strategy='auto')
    """
    # Validate inputs
    X, y = check_X_y(X, y)
    classes, counts = np.unique(y, return_counts=True)

    # Initialize warnings
    warnings = []

    # Set default sampling strategy if needed
    if sampling_strategy == 'auto':
        majority_class = classes[np.argmax(counts)]
        sampling_strategy = {c: 1.0 if c == majority_class else float(counts[np.argmax(counts)])/counts[i]
                            for i, c in enumerate(classes)}

    # Apply normalization if specified
    X_normalized = normalizer(X) if normalizer else X

    # Get class indices
    class_indices = {c: np.where(y == c)[0] for c in classes}

    # Perform undersampling
    X_res, y_res = _perform_undersampling(
        X_normalized,
        y,
        class_indices,
        sampling_strategy,
        random_state=random_state,
        replacement=replacement,
        distance_metric=distance_metric
    )

    # Calculate metrics
    metrics = _calculate_metrics(X_res, y_res, metric)

    return {
        'result': (X_res, y_res),
        'metrics': metrics,
        'params_used': {
            'sampling_strategy': sampling_strategy,
            'random_state': random_state,
            'replacement': replacement,
            'distance_metric': distance_metric.__name__ if callable(distance_metric) else distance_metric,
            'normalizer': normalizer.__name__ if normalizer and callable(normalizer) else None,
            'metric': metric.__name__ if callable(metric) else metric
        },
        'warnings': warnings
    }

def _perform_undersampling(
    X: np.ndarray,
    y: np.ndarray,
    class_indices: Dict[int, np.ndarray],
    sampling_strategy: Union[str, Dict[int, float]],
    *,
    random_state: Optional[int] = None,
    replacement: bool = False,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = 'euclidean'
) -> tuple:
    """
    Internal function to perform the actual undersampling.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    class_indices : dict
        Dictionary mapping classes to their indices in y
    sampling_strategy : str or dict
        Sampling strategy to use
    random_state : int, optional
        Random seed for reproducibility
    replacement : bool, default=False
        Whether to sample with replacement
    distance_metric : callable or str
        Distance metric to use for sampling

    Returns
    -------
    tuple
        (X_res, y_res) - undersampled data
    """
    rng = np.random.RandomState(random_state)
    classes = list(class_indices.keys())

    # Determine target counts for each class
    if isinstance(sampling_strategy, dict):
        min_count = min(len(indices) for indices in class_indices.values())
        target_counts = {c: int(min_count * ratio) if c != classes[np.argmax([len(class_indices[c]) for c in classes])]
                        else min_count
                        for c, ratio in sampling_strategy.items()}
    else:
        raise ValueError("Unsupported sampling strategy type")

    # Perform undersampling for each class
    X_res_list = []
    y_res_list = []

    for c in classes:
        indices = class_indices[c]
        target_count = target_counts[c]

        if len(indices) > target_count:
            # For majority classes, perform undersampling
            sampled_indices = rng.choice(indices, size=target_count, replace=replacement)
        else:
            # For minority classes, keep all samples
            sampled_indices = indices

        X_res_list.append(X[sampled_indices])
        y_res_list.append(y[sampled_indices])

    # Concatenate results
    X_res = np.concatenate(X_res_list, axis=0)
    y_res = np.concatenate(y_res_list, axis=0)

    return X_res, y_res

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = 'accuracy'
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for the undersampled data.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    metric : callable or str
        Metric to calculate

    Returns
    -------
    dict
        Dictionary of calculated metrics
    """
    # For simplicity, we'll just calculate class distribution as a metric
    classes, counts = np.unique(y, return_counts=True)
    class_distribution = {c: count for c, count in zip(classes, counts)}

    # If a custom metric is provided, calculate it
    if callable(metric):
        try:
            class_distribution['custom_metric'] = metric(X, y)
        except Exception as e:
            class_distribution['custom_metric_error'] = str(e)

    return class_distribution

# Default distance metrics
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.sum(np.abs(a - b))

# Default normalizers
def standard_normalizer(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(axis=0)) / X.std(axis=0)

def minmax_normalizer(X: np.ndarray) -> np.ndarray:
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# Default metrics
def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_true == y_pred)

################################################################################
# oversampling
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

def _compute_distance(X1: np.ndarray, X2: np.ndarray,
                     distance: str = 'euclidean') -> np.ndarray:
    """Compute pairwise distances between two sets of samples."""
    if distance == 'euclidean':
        return np.sqrt(np.sum((X1[:, np.newaxis] - X2) ** 2, axis=2))
    elif distance == 'manhattan':
        return np.sum(np.abs(X1[:, np.newaxis] - X2), axis=2)
    elif distance == 'cosine':
        dot_product = np.dot(X1, X2.T)
        norm_X1 = np.linalg.norm(X1, axis=1)[:, np.newaxis]
        norm_X2 = np.linalg.norm(X2, axis=1)[np.newaxis, :]
        return 1 - (dot_product / (norm_X1 * norm_X2 + 1e-8))
    elif distance == 'minkowski':
        p = 3
        return np.sum(np.abs(X1[:, np.newaxis] - X2) ** p, axis=2) ** (1/p)
    else:
        raise ValueError(f"Unknown distance metric: {distance}")

def _oversample_random(X_minority: np.ndarray, n_samples: int,
                      random_state: Optional[int] = None) -> np.ndarray:
    """Random oversampling of minority class samples."""
    rng = np.random.RandomState(random_state)
    indices = rng.choice(X_minority.shape[0], size=n_samples, replace=True)
    return X_minority[indices]

def _oversample_smote(X_minority: np.ndarray, X_majority: np.ndarray,
                     n_samples: int, k: int = 5,
                     distance: str = 'euclidean',
                     random_state: Optional[int] = None) -> np.ndarray:
    """SMOTE (Synthetic Minority Over-sampling Technique)."""
    rng = np.random.RandomState(random_state)
    distances = _compute_distance(X_minority, X_majority, distance)
    synthetic_samples = []

    for _ in range(n_samples):
        idx = rng.choice(X_minority.shape[0])
        nn_indices = np.argsort(distances[idx])[:k]
        nn_idx = rng.choice(nn_indices)
        diff = X_majority[nn_idx] - X_minority[idx]
        gap = rng.random()
        synthetic_samples.append(X_minority[idx] + gap * diff)

    return np.array(synthetic_samples)

def oversampling_fit(X: np.ndarray, y: np.ndarray,
                    method: str = 'random',
                    sampling_ratio: float = 1.0,
                    normalize_method: str = 'standard',
                    distance_metric: str = 'euclidean',
                    random_state: Optional[int] = None) -> Dict:
    """
    Oversampling function for class imbalance.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    method : str
        Oversampling method ('random' or 'smote')
    sampling_ratio : float
        Ratio of samples to generate (relative to minority class size)
    normalize_method : str
        Normalization method for features
    distance_metric : str
        Distance metric to use (for SMOTE)
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.array([0]*80 + [1]*20)
    >>> result = oversampling_fit(X, y, method='smote', sampling_ratio=0.5)
    """
    _validate_inputs(X, y)

    # Separate classes
    class_0 = X[y == 0]
    class_1 = X[y == 1]

    # Determine minority and majority classes
    if len(class_0) < len(class_1):
        X_minority = class_0
        X_majority = class_1
    else:
        X_minority = class_1
        X_majority = class_0

    # Normalize data
    X_normalized = _normalize_data(X, normalize_method)

    # Determine number of samples to generate
    n_samples = int(len(X_minority) * sampling_ratio)

    # Perform oversampling
    if method == 'random':
        X_synthetic = _oversample_random(X_minority, n_samples, random_state)
    elif method == 'smote':
        X_synthetic = _oversample_smote(X_minority, X_majority,
                                      n_samples, distance_metric=distance_metric,
                                      random_state=random_state)
    else:
        raise ValueError(f"Unknown oversampling method: {method}")

    # Combine original and synthetic samples
    X_balanced = np.vstack([X, X_synthetic])
    y_balanced = np.hstack([y, np.full(n_samples, X_minority[0][-1])])

    # Calculate metrics
    class_counts = np.bincount(y_balanced)
    imbalance_ratio = min(class_counts) / max(class_counts)

    return {
        'result': {'X_balanced': X_balanced, 'y_balanced': y_balanced},
        'metrics': {'imbalance_ratio': imbalance_ratio},
        'params_used': {
            'method': method,
            'sampling_ratio': sampling_ratio,
            'normalize_method': normalize_method,
            'distance_metric': distance_metric
        },
        'warnings': []
    }

################################################################################
# SMOTE
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple
from sklearn.utils.validation import check_X_y, check_array

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Validate input arrays."""
    X, y = check_X_y(X, y)
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input data contains NaN or infinite values.")
    return X, y

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

def _calculate_distance(X: np.ndarray, Y: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """Calculate distance between two sets of points."""
    if metric == 'euclidean':
        return np.sqrt(np.sum((X[:, np.newaxis] - Y) ** 2, axis=2))
    elif metric == 'manhattan':
        return np.sum(np.abs(X[:, np.newaxis] - Y), axis=2)
    elif metric == 'cosine':
        dot_product = np.dot(X, Y.T)
        norm_X = np.linalg.norm(X, axis=1)[:, np.newaxis]
        norm_Y = np.linalg.norm(Y, axis=1)[np.newaxis, :]
        return 1 - (dot_product / (norm_X * norm_Y + 1e-8))
    elif metric == 'minkowski':
        p = 3
        return np.sum(np.abs(X[:, np.newaxis] - Y) ** p, axis=2) ** (1/p)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _generate_synthetic_samples(X_minority: np.ndarray, k_neighbors: int = 5,
                              distance_metric: str = 'euclidean') -> np.ndarray:
    """Generate synthetic samples using SMOTE algorithm."""
    n_samples = X_minority.shape[0]
    synthetic_samples = np.zeros_like(X_minority)

    for i in range(n_samples):
        distances = _calculate_distance(X_minority[i:i+1], X_minority, distance_metric)
        nearest_indices = np.argsort(distances[0])[1:k_neighbors+1]
        random_index = np.random.randint(0, k_neighbors)
        neighbor = X_minority[nearest_indices[random_index]]
        gap = np.random.rand()

        synthetic_samples[i] = X_minority[i] + gap * (neighbor - X_minority[i])

    return synthetic_samples

def SMOTE_fit(X: np.ndarray, y: np.ndarray,
              normalization_method: str = 'standard',
              distance_metric: str = 'euclidean',
              k_neighbors: int = 5,
              sampling_strategy: float = 1.0) -> Dict:
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance class distribution.

    Parameters:
    - X: Input features array of shape (n_samples, n_features)
    - y: Target labels array of shape (n_samples,)
    - normalization_method: Normalization method ('none', 'standard', 'minmax', 'robust')
    - distance_metric: Distance metric for neighbor calculation ('euclidean', 'manhattan', 'cosine', 'minkowski')
    - k_neighbors: Number of nearest neighbors to consider
    - sampling_strategy: Ratio of synthetic samples to generate for minority class

    Returns:
    - Dictionary containing:
        * 'result': Balanced dataset
        * 'metrics': Performance metrics
        * 'params_used': Parameters used in the computation
        * 'warnings': Any warnings generated during computation

    Example:
    >>> X = np.random.rand(10, 5)
    >>> y = np.array([0]*8 + [1]*2)
    >>> result = SMOTE_fit(X, y)
    """
    # Validate inputs
    X, y = _validate_inputs(X, y)

    # Normalize data
    X_normalized = _normalize_data(X, normalization_method)

    # Separate classes
    classes, counts = np.unique(y, return_counts=True)
    minority_class = classes[np.argmin(counts)]
    majority_class = classes[np.argmax(counts)]

    # Get minority class samples
    X_minority = X_normalized[y == minority_class]

    # Generate synthetic samples
    n_synthetic = int(sampling_strategy * len(X_minority))
    synthetic_samples = _generate_synthetic_samples(X_minority, k_neighbors, distance_metric)

    # Combine original and synthetic samples
    X_balanced = np.vstack([X_normalized, synthetic_samples])
    y_balanced = np.hstack([y, np.full(n_synthetic, minority_class)])

    # Shuffle the dataset
    indices = np.random.permutation(len(X_balanced))
    X_balanced = X_balanced[indices]
    y_balanced = y_balanced[indices]

    # Prepare output
    result = {
        'result': {'X': X_balanced, 'y': y_balanced},
        'metrics': {
            'original_class_distribution': dict(zip(classes, counts)),
            'balanced_class_distribution': np.bincount(y_balanced)
        },
        'params_used': {
            'normalization_method': normalization_method,
            'distance_metric': distance_metric,
            'k_neighbors': k_neighbors,
            'sampling_strategy': sampling_strategy
        },
        'warnings': []
    }

    return result

################################################################################
# ADASYN
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union
from sklearn.neighbors import NearestNeighbors

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: Dict[int, int],
    n_neighbors: int = 5
) -> None:
    """Validate input data and parameters."""
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if len(X.shape) != 2:
        raise ValueError("X must be a 2D array")
    if not all(isinstance(k, int) and isinstance(v, int) for k, v in sampling_strategy.items()):
        raise ValueError("sampling_strategy must be a dictionary with integer keys and values")
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be a positive integer")

def _calculate_synthetic_samples(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: Dict[int, int],
    n_neighbors: int = 5,
    metric: str = 'euclidean',
    random_state: Optional[int] = None
) -> np.ndarray:
    """Calculate synthetic samples using ADASYN algorithm."""
    rng = np.random.RandomState(random_state)
    X_resampled = []
    y_resampled = []

    for class_label, n_samples in sampling_strategy.items():
        if n_samples == 0:
            continue

        X_class = X[y == class_label]
        if len(X_class) == 0:
            continue

        # Calculate density ratio
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        nn.fit(X_class)
        distances, _ = nn.kneighbors(X_class)
        density_ratio = np.mean(distances[:, -1], axis=1)

        # Calculate number of synthetic samples for each minority class sample
        n_synth_per_sample = np.floor(density_ratio * n_samples / len(X_class)).astype(int)

        # Generate synthetic samples
        for i in range(len(X_class)):
            n_synth = n_synth_per_sample[i]
            if n_synth > 0:
                for _ in range(n_synth):
                    # Randomly select a neighbor
                    neighbor_idx = rng.randint(1, n_neighbors)
                    synth_sample = X_class[i] + rng.rand() * (X_class[neighbor_idx] - X_class[i])
                    X_resampled.append(synth_sample)
                    y_resampled.append(class_label)

    return np.array(X_resampled), np.array(y_resampled)

def ADASYN_fit(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: Dict[int, int],
    n_neighbors: int = 5,
    metric: str = 'euclidean',
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, Dict, str]]:
    """
    Apply ADASYN (Adaptive Synthetic Sampling) to balance class distribution.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    sampling_strategy : Dict[int, int]
        Dictionary specifying the desired number of samples for each class
    n_neighbors : int, optional (default=5)
        Number of neighbors to use for density estimation
    metric : str, optional (default='euclidean')
        Distance metric to use for neighbor search
    random_state : int, optional (default=None)
        Random seed for reproducibility

    Returns:
    --------
    result : Dict
        Dictionary containing:
        - 'X_resampled': Resampled feature matrix
        - 'y_resampled': Resampled target vector
        - 'metrics': Dictionary of performance metrics
        - 'params_used': Dictionary of parameters used
        - 'warnings': List of warnings encountered

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.array([0]*80 + [1]*20)
    >>> strategy = {0: 100, 1: 40}
    >>> result = ADASYN_fit(X, y, strategy)
    """
    # Validate inputs
    _validate_inputs(X, y, sampling_strategy, n_neighbors)

    # Initialize warnings
    warnings = []

    # Calculate synthetic samples
    X_synth, y_synth = _calculate_synthetic_samples(
        X, y, sampling_strategy, n_neighbors, metric, random_state
    )

    # Combine original and synthetic samples
    X_resampled = np.vstack([X, X_synth])
    y_resampled = np.hstack([y, y_synth])

    # Calculate class distribution metrics
    original_counts = {k: sum(y == k) for k in np.unique(y)}
    new_counts = {k: sum(y_resampled == k) for k in np.unique(y_resampled)}
    metrics = {
        'original_class_distribution': original_counts,
        'new_class_distribution': new_counts
    }

    # Record parameters used
    params_used = {
        'n_neighbors': n_neighbors,
        'metric': metric,
        'random_state': random_state
    }

    return {
        'result': {'X_resampled': X_resampled, 'y_resampled': y_resampled},
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

################################################################################
# RandomUnderSampler
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List
from sklearn.utils.validation import check_X_y, check_array

def validate_inputs(X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
    """
    Validate input arrays.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : Optional[np.ndarray]
        Target vector.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if y is not None:
        X, y = check_X_y(X, y)
    else:
        X = check_array(X)

    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Input contains NaN or infinite values.")

def _random_undersample(X: np.ndarray, y: np.ndarray,
                        sampling_strategy: Dict[int, int],
                        random_state: Optional[int] = None) -> tuple:
    """
    Perform random undersampling.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    sampling_strategy : Dict[int, int]
        Dictionary specifying the number of samples to keep for each class.
    random_state : Optional[int]
        Random seed.

    Returns
    -------
    tuple
        (X_resampled, y_resampled)
    """
    rng = np.random.RandomState(random_state)

    X_resampled = []
    y_resampled = []

    for class_label, n_samples in sampling_strategy.items():
        class_indices = np.where(y == class_label)[0]
        if len(class_indices) > n_samples:
            selected_indices = rng.choice(class_indices, size=n_samples, replace=False)
        else:
            selected_indices = class_indices
        X_resampled.append(X[selected_indices])
        y_resampled.append(y[selected_indices])

    return np.vstack(X_resampled), np.hstack(y_resampled)

def calculate_class_distribution(y: np.ndarray) -> Dict[int, int]:
    """
    Calculate class distribution.

    Parameters
    ----------
    y : np.ndarray
        Target vector.

    Returns
    -------
    Dict[int, int]
        Class distribution.
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    return dict(zip(unique_classes, counts))

def RandomUnderSampler_fit(X: np.ndarray,
                           y: Optional[np.ndarray] = None,
                           sampling_strategy: Union[str, Dict[int, int]] = 'auto',
                           random_state: Optional[int] = None) -> Dict:
    """
    Perform random undersampling.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : Optional[np.ndarray]
        Target vector. If None, X is assumed to be the target.
    sampling_strategy : Union[str, Dict[int, int]]
        Sampling strategy. 'auto' will balance classes.
    random_state : Optional[int]
        Random seed.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': (X_resampled, y_resampled)
        - 'metrics': {'class_distribution': dict}
        - 'params_used': dict
        - 'warnings': list

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = RandomUnderSampler_fit(X, y, sampling_strategy='auto')
    """
    validate_inputs(X, y)

    if y is None:
        y = X
        X = np.arange(len(y)).reshape(-1, 1)

    class_distribution = calculate_class_distribution(y)
    params_used = {
        'sampling_strategy': sampling_strategy,
        'random_state': random_state
    }

    if isinstance(sampling_strategy, str):
        if sampling_strategy == 'auto':
            min_samples = min(class_distribution.values())
            sampling_strategy = {cls: min_samples for cls in class_distribution}
        else:
            raise ValueError("Invalid sampling_strategy string.")

    X_resampled, y_resampled = _random_undersample(X, y,
                                                  sampling_strategy,
                                                  random_state)

    metrics = {
        'class_distribution': calculate_class_distribution(y_resampled)
    }

    warnings = []

    return {
        'result': (X_resampled, y_resampled),
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

################################################################################
# RandomOverSampler
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_input(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate input data for RandomOverSampler.

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
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains NaN or infinite values.")
    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("y contains NaN or infinite values.")

def _random_oversample(X: np.ndarray, y: np.ndarray,
                      sampling_strategy: Dict[int, int],
                      random_state: Optional[int] = None) -> tuple:
    """
    Perform random oversampling on the minority class(es).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    sampling_strategy : Dict[int, int]
        Dictionary mapping class labels to desired counts.
    random_state : Optional[int], default=None
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (X_resampled, y_resampled) after oversampling.
    """
    rng = np.random.RandomState(random_state)
    X_resampled = []
    y_resampled = []

    for class_label in np.unique(y):
        X_class = X[y == class_label]
        y_class = y[y == class_label]

        n_samples_to_add = sampling_strategy[class_label] - len(y_class)
        if n_samples_to_add > 0:
            indices = rng.choice(len(X_class), size=n_samples_to_add, replace=True)
            X_resampled.append(np.vstack([X_class, X_class[indices]]))
            y_resampled.append(np.concatenate([y_class, y_class[indices]]))
        else:
            X_resampled.append(X_class)
            y_resampled.append(y_class)

    return np.vstack(X_resampled), np.concatenate(y_resampled)

def _compute_sampling_strategy(y: np.ndarray,
                              strategy: str = 'auto',
                              ratio: Optional[float] = None) -> Dict[int, int]:
    """
    Compute the sampling strategy based on input parameters.

    Parameters
    ----------
    y : np.ndarray
        Target vector of shape (n_samples,).
    strategy : str, default='auto'
        Sampling strategy ('auto', 'majority', or 'minority').
    ratio : Optional[float], default=None
        Desired ratio of majority to minority class.

    Returns
    -------
    Dict[int, int]
        Dictionary mapping class labels to desired counts.
    """
    classes, counts = np.unique(y, return_counts=True)
    if strategy == 'auto':
        max_count = np.max(counts)
        return {cls: max_count for cls in classes}
    elif strategy == 'majority':
        min_count = np.min(counts)
        return {cls: min_count if counts[np.where(classes == cls)[0][0]] != max(counts) else min_count for cls in classes}
    elif strategy == 'minority':
        max_count = np.max(counts)
        return {cls: max_count if counts[np.where(classes == cls)[0][0]] != min(counts) else max_count for cls in classes}
    elif ratio is not None:
        majority_class = classes[np.argmax(counts)]
        minority_class = classes[np.argmin(counts)]
        majority_count = counts[np.argmax(counts)]
        minority_count = int(majority_count / ratio)
        return {cls: minority_count if cls == minority_class else majority_count for cls in classes}
    else:
        raise ValueError("Invalid strategy or ratio provided.")

def RandomOverSampler_fit(X: np.ndarray,
                         y: np.ndarray,
                         strategy: str = 'auto',
                         ratio: Optional[float] = None,
                         random_state: Optional[int] = None) -> Dict[str, Any]:
    """
    Perform Random OverSampling on the input data.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    strategy : str, default='auto'
        Sampling strategy ('auto', 'majority', or 'minority').
    ratio : Optional[float], default=None
        Desired ratio of majority to minority class.
    random_state : Optional[int], default=None
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'result': tuple of (X_resampled, y_resampled)
        - 'metrics': dictionary of metrics
        - 'params_used': dictionary of parameters used
        - 'warnings': list of warnings

    Examples
    --------
    >>> X = np.array([[1], [2], [3], [4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> result = RandomOverSampler_fit(X, y)
    """
    validate_input(X, y)

    sampling_strategy = _compute_sampling_strategy(y, strategy, ratio)
    X_resampled, y_resampled = _random_oversample(X, y, sampling_strategy, random_state)

    metrics = {
        'original_class_counts': dict(zip(*np.unique(y, return_counts=True))),
        'resampled_class_counts': dict(zip(*np.unique(y_resampled, return_counts=True)))
    }

    params_used = {
        'strategy': strategy,
        'ratio': ratio,
        'random_state': random_state
    }

    warnings = []

    return {
        'result': (X_resampled, y_resampled),
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

################################################################################
# TomekLinks
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate input data for TomekLinks.

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

def _compute_distance_matrix(X: np.ndarray, metric: str) -> np.ndarray:
    """
    Compute distance matrix using specified metric.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    metric : str
        Distance metric to use.

    Returns
    ------
    np.ndarray
        Distance matrix of shape (n_samples, n_samples).
    """
    if metric == 'euclidean':
        return np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
    elif metric == 'manhattan':
        return np.sum(np.abs(X[:, np.newaxis] - X), axis=2)
    elif metric == 'cosine':
        return 1 - np.dot(X, X.T) / (np.linalg.norm(X, axis=1)[:, np.newaxis] * np.linalg.norm(X, axis=1))
    elif metric == 'minkowski':
        return np.sum(np.abs(X[:, np.newaxis] - X) ** 3, axis=2) ** (1/3)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def _find_tomek_links(X: np.ndarray, y: np.ndarray, distance_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Find Tomek links in the dataset.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    distance_matrix : np.ndarray
        Precomputed distance matrix.

    Returns
    ------
    Dict[str, np.ndarray]
        Dictionary containing indices of Tomek links and their neighbors.
    """
    tomek_links = []
    for i in range(len(X)):
        # Find the nearest neighbor of opposite class
        mask = y != y[i]
        if np.any(mask):
            nearest_idx = np.argmin(distance_matrix[i][mask])
            nearest_i = np.where(mask)[0][nearest_idx]
            # Check if the pair forms a Tomek link
            if distance_matrix[i][nearest_i] < np.min(distance_matrix[nearest_i]) and \
               distance_matrix[i][nearest_i] < np.min(distance_matrix[i]):
                tomek_links.append((i, nearest_i))
    return {'tomek_links': np.array(tomek_links)}

def TomekLinks_fit(X: np.ndarray, y: np.ndarray,
                   metric: str = 'euclidean',
                   custom_metric: Optional[Callable] = None) -> Dict[str, Union[np.ndarray, Dict[str, str]]]:
    """
    Apply Tomek Links method for class balancing.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    metric : str, optional
        Distance metric to use. Default is 'euclidean'.
    custom_metric : Callable, optional
        Custom distance metric function.

    Returns
    ------
    Dict[str, Union[np.ndarray, Dict[str, str]]]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    _validate_inputs(X, y)

    if custom_metric is not None:
        distance_matrix = custom_metric(X)
    else:
        distance_matrix = _compute_distance_matrix(X, metric)

    result = _find_tomek_links(X, y, distance_matrix)

    return {
        'result': result,
        'metrics': {},
        'params_used': {'metric': metric if custom_metric is None else 'custom'},
        'warnings': []
    }

# Example usage:
# X = np.random.rand(100, 5)
# y = np.random.randint(0, 2, size=100)
# result = TomekLinks_fit(X, y, metric='euclidean')

################################################################################
# NearMiss
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate input data for NearMiss algorithm.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)

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

def compute_distance(X: np.ndarray, distance_metric: str = 'euclidean',
                    custom_distance: Optional[Callable] = None) -> Callable:
    """
    Return a distance function based on user choice.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    distance_metric : str
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski')
    custom_distance : Optional[Callable]
        Custom distance function if needed

    Returns
    -------
    Callable
        Distance function
    """
    def euclidean(a, b):
        return np.linalg.norm(a - b)

    def manhattan(a, b):
        return np.sum(np.abs(a - b))

    def cosine(a, b):
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def minkowski(a, b, p=3):
        return np.sum(np.abs(a - b)**p)**(1/p)

    if custom_distance is not None:
        return custom_distance
    elif distance_metric == 'euclidean':
        return euclidean
    elif distance_metric == 'manhattan':
        return manhattan
    elif distance_metric == 'cosine':
        return cosine
    elif distance_metric == 'minkowski':
        return lambda a, b: minkowski(a, b)
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

def near_miss_algorithm(X: np.ndarray, y: np.ndarray,
                       sampling_ratio: float = 0.5,
                       distance_metric: str = 'euclidean',
                       custom_distance: Optional[Callable] = None) -> np.ndarray:
    """
    NearMiss undersampling algorithm implementation.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    sampling_ratio : float
        Ratio of majority class to keep
    distance_metric : str
        Distance metric to use
    custom_distance : Optional[Callable]
        Custom distance function

    Returns
    -------
    np.ndarray
        Indices of samples to keep
    """
    validate_inputs(X, y)

    # Identify majority and minority classes
    unique_classes, counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2:
        raise ValueError("Need at least two classes for undersampling")

    majority_class = unique_classes[np.argmax(counts)]
    minority_class = unique_classes[np.argmin(counts)]

    # Get indices for each class
    maj_idx = np.where(y == majority_class)[0]
    min_idx = np.where(y == minority_class)[0]

    # Compute distances between majority and minority samples
    distance_func = compute_distance(X, distance_metric, custom_distance)
    distances = np.zeros((len(maj_idx), len(min_idx)))

    for i, maj_sample in enumerate(X[maj_idx]):
        for j, min_sample in enumerate(X[min_idx]):
            distances[i, j] = distance_func(maj_sample, min_sample)

    # For each minority sample, find the k nearest majority samples
    k = int(sampling_ratio * len(maj_idx))
    if k == 0:
        return np.array([])

    # Find the majority samples that are farthest from any minority sample
    mean_distances = np.mean(distances, axis=1)
    farthest_maj_idx = np.argsort(mean_distances)[-k:]

    return maj_idx[farthest_maj_idx]

def NearMiss_fit(X: np.ndarray, y: np.ndarray,
                sampling_ratio: float = 0.5,
                distance_metric: str = 'euclidean',
                custom_distance: Optional[Callable] = None,
                random_state: Optional[int] = None) -> Dict:
    """
    NearMiss undersampling algorithm wrapper function.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    sampling_ratio : float
        Ratio of majority class to keep
    distance_metric : str
        Distance metric to use
    custom_distance : Optional[Callable]
        Custom distance function
    random_state : Optional[int]
        Random seed for reproducibility

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': indices of samples to keep
        - 'metrics': dictionary of metrics
        - 'params_used': parameters used
        - 'warnings': any warnings
    """
    if random_state is not None:
        np.random.seed(random_state)

    result = near_miss_algorithm(X, y, sampling_ratio,
                               distance_metric, custom_distance)

    # Calculate metrics
    unique_classes, counts = np.unique(y, return_counts=True)
    majority_class = unique_classes[np.argmax(counts)]
    minority_count = np.min(counts)

    metrics = {
        'original_class_distribution': {cls: cnt for cls, cnt in zip(unique_classes, counts)},
        'new_class_distribution': {
            majority_class: len(result),
            unique_classes[np.argmin(counts)]: minority_count
        },
        'sampling_ratio_used': len(result) / counts[np.argmax(counts)]
    }

    params_used = {
        'sampling_ratio': sampling_ratio,
        'distance_metric': distance_metric,
        'custom_distance': custom_distance is not None
    }

    warnings = []
    if len(result) == 0:
        warnings.append("No samples selected - check sampling ratio")

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

################################################################################
# SMOTEENN
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Validate input arrays."""
    X, y = check_X_y(X, y)
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input X contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Input y contains NaN or infinite values.")
    return X, y

def _normalize_data(X: np.ndarray,
                    normalization: str = 'standard',
                    custom_normalizer: Optional[Callable] = None) -> np.ndarray:
    """Normalize data based on user choice."""
    if custom_normalizer is not None:
        return custom_normalizer(X)
    if normalization == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif normalization == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif normalization == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        return X

def _calculate_metrics(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       metric: str = 'mse',
                       custom_metric: Optional[Callable] = None) -> Dict[str, float]:
    """Calculate metrics based on user choice."""
    if custom_metric is not None:
        return {'custom': custom_metric(y_true, y_pred)}
    metrics = {}
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
    return metrics

def _smote_sampling(X: np.ndarray,
                    y: np.ndarray,
                    sampling_strategy: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Perform SMOTE oversampling."""
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(sampling_strategy=sampling_strategy)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

def _enn_sampling(X: np.ndarray,
                  y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform ENN undersampling."""
    from imblearn.under_sampling import EditedNearestNeighbours
    enn = EditedNearestNeighbours()
    X_res, y_res = enn.fit_resample(X, y)
    return X_res, y_res

def SMOTEENN_fit(X: np.ndarray,
                 y: np.ndarray,
                 normalization: str = 'standard',
                 metric: str = 'mse',
                 custom_normalizer: Optional[Callable] = None,
                 custom_metric: Optional[Callable] = None) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Perform SMOTEENN class balancing.

    Parameters:
    -----------
    X : np.ndarray
        Input features.
    y : np.ndarray
        Target labels.
    normalization : str, optional (default='standard')
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : str, optional (default='mse')
        Metric for evaluation: 'mse', 'mae', 'r2', or 'logloss'.
    custom_normalizer : Callable, optional
        Custom normalization function.
    custom_metric : Callable, optional
        Custom metric function.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    X, y = _validate_inputs(X, y)

    # Normalize data
    X_norm = _normalize_data(X, normalization, custom_normalizer)

    # Perform SMOTEENN
    X_smote, y_smote = _smote_sampling(X_norm, y)
    X_res, y_res = _enn_sampling(X_smote, y_smote)

    # Calculate metrics
    metrics = _calculate_metrics(y, y_res[:, 0], metric, custom_metric)

    # Prepare output
    result = {
        'X_resampled': X_res,
        'y_resampled': y_res
    }

    params_used = {
        'normalization': normalization,
        'metric': metric,
        'custom_normalizer': custom_normalizer is not None,
        'custom_metric': custom_metric is not None
    }

    warnings = {
        'warnings': []
    }

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

################################################################################
# SMOTETomek
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    X, y = check_X_y(X, y)
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input data contains NaN or infinite values.")
    if len(np.unique(y)) < 2:
        raise ValueError("The target array must have at least two classes.")

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

def _calculate_distance(X: np.ndarray, y: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """Calculate pairwise distances between samples."""
    if metric == 'euclidean':
        return np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
    elif metric == 'manhattan':
        return np.sum(np.abs(X[:, np.newaxis] - X), axis=2)
    elif metric == 'cosine':
        dot_products = np.dot(X, X.T)
        norms = np.sqrt(np.sum(X ** 2, axis=1))
        return 1 - dot_products / (norms[:, np.newaxis] * norms)
    elif metric == 'minkowski':
        p = 3
        return np.sum(np.abs(X[:, np.newaxis] - X) ** p, axis=2) ** (1/p)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _smote_oversampling(X: np.ndarray, y: np.ndarray, sampling_strategy: float = 1.0) -> tuple:
    """Perform SMOTE oversampling."""
    minority_mask = y == np.unique(y)[0]
    majority_mask = y == np.unique(y)[1]

    n_minority = np.sum(minority_mask)
    n_majority = np.sum(majority_mask)

    n_samples_to_generate = int(sampling_strategy * (n_majority - n_minority))

    if n_samples_to_generate <= 0:
        return X, y

    distances = _calculate_distance(X[minority_mask], X)
    nearest_indices = np.argsort(distances, axis=1)[:, 1]

    synthetic_samples = []
    for i in range(n_minority):
        nearest_idx = nearest_indices[i]
        if nearest_idx >= n_minority:
            continue
        diff = X[nearest_idx] - X[i]
        gap = np.random.rand() * 1.0
        synthetic_sample = X[i] + gap * diff
        synthetic_samples.append(synthetic_sample)

    synthetic_samples = np.array(synthetic_samples)[:n_samples_to_generate]
    X_synthetic = np.vstack([X, synthetic_samples])
    y_synthetic = np.hstack([y, np.zeros(n_samples_to_generate) + np.unique(y)[0]])

    return X_synthetic, y_synthetic

def _tomek_links_undersampling(X: np.ndarray, y: np.ndarray) -> tuple:
    """Perform Tomek Links undersampling."""
    distances = _calculate_distance(X, X)
    minority_mask = y == np.unique(y)[0]
    majority_mask = y == np.unique(y)[1]

    tomek_links = []
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            if (minority_mask[i] and majority_mask[j]) or (majority_mask[i] and minority_mask[j]):
                if distances[i, j] == 0:
                    tomek_links.append((i, j))

    X_filtered = np.delete(X, [idx for pair in tomek_links for idx in pair], axis=0)
    y_filtered = np.delete(y, [idx for pair in tomek_links for idx in pair])

    return X_filtered, y_filtered

def SMOTETomek_fit(X: np.ndarray,
                   y: np.ndarray,
                   sampling_strategy: float = 1.0,
                   normalize_method: str = 'standard',
                   distance_metric: str = 'euclidean') -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Perform SMOTE-Tomek ensemble method for class balancing.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    sampling_strategy : float, optional
        Sampling strategy for SMOTE oversampling. Default is 1.0.
    normalize_method : str, optional
        Normalization method for features. Default is 'standard'.
    distance_metric : str, optional
        Distance metric for SMOTE and Tomek Links. Default is 'euclidean'.

    Returns
    -------
    dict
        Dictionary containing:
        - "result": Balanced feature matrix and target vector.
        - "metrics": Dictionary of computed metrics.
        - "params_used": Dictionary of parameters used.
        - "warnings": List of warnings encountered.

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.hstack([np.zeros(80), np.ones(20)])
    >>> result = SMOTETomek_fit(X, y)
    """
    _validate_inputs(X, y)

    warnings = []
    params_used = {
        'sampling_strategy': sampling_strategy,
        'normalize_method': normalize_method,
        'distance_metric': distance_metric
    }

    X_normalized = _normalize_data(X, normalize_method)

    # SMOTE Oversampling
    X_smote, y_smote = _smote_oversampling(X_normalized, y, sampling_strategy)

    # Tomek Links Undersampling
    X_tomek, y_tomek = _tomek_links_undersampling(X_smote, y_smote)

    metrics = {
        'class_distribution_before': np.bincount(y),
        'class_distribution_after': np.bincount(y_tomek)
    }

    return {
        'result': (X_tomek, y_tomek),
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

################################################################################
# class_weight
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(y: np.ndarray) -> None:
    """Validate input array."""
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if np.any(np.isnan(y)):
        raise ValueError("y contains NaN values")

def _compute_class_weights(
    y: np.ndarray,
    normalization: str = 'none',
    metric: Union[str, Callable] = 'inverse_frequency',
    custom_metric: Optional[Callable] = None
) -> np.ndarray:
    """Compute class weights based on specified metric and normalization."""
    classes, counts = np.unique(y, return_counts=True)
    class_proportions = counts / len(y)

    if callable(metric):
        weights = metric(class_proportions)
    elif metric == 'inverse_frequency':
        weights = 1. / class_proportions
    elif metric == 'sqrt_inverse_frequency':
        weights = 1. / np.sqrt(class_proportions)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if normalization == 'none':
        pass
    elif normalization == 'standard':
        weights = (weights - np.mean(weights)) / np.std(weights)
    elif normalization == 'minmax':
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
    elif normalization == 'robust':
        weights = (weights - np.median(weights)) / (np.percentile(weights, 75) - np.percentile(weights, 25))
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    return weights[np.argsort(classes)]

def _compute_metrics(
    y: np.ndarray,
    weights: np.ndarray
) -> Dict[str, float]:
    """Compute metrics for the computed class weights."""
    classes = np.unique(y)
    counts = np.bincount(y.astype(int))
    class_proportions = counts / len(y)

    metrics = {
        'weighted_entropy': -np.sum(weights * class_proportions * np.log(class_proportions + 1e-10)),
        'max_weight': np.max(weights),
        'min_weight': np.min(weights)
    }
    return metrics

def class_weight_fit(
    y: np.ndarray,
    normalization: str = 'none',
    metric: Union[str, Callable] = 'inverse_frequency',
    custom_metric: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Compute class weights for imbalanced classification problems.

    Parameters:
    -----------
    y : np.ndarray
        Target labels (1D array)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, optional
        Metric to compute weights ('inverse_frequency', 'sqrt_inverse_frequency')
    custom_metric : callable, optional
        Custom function to compute weights

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': computed class weights (np.ndarray)
        - 'metrics': computed metrics (dict)
        - 'params_used': parameters used (dict)
        - 'warnings': warnings (list)

    Example:
    --------
    >>> y = np.array([0, 0, 1, 1, 1, 2])
    >>> result = class_weight_fit(y, normalization='standard', metric='inverse_frequency')
    """
    _validate_inputs(y)

    warnings = []
    params_used = {
        'normalization': normalization,
        'metric': metric if not callable(metric) else 'custom',
        'custom_metric_used': bool(custom_metric)
    }

    if custom_metric is not None:
        metric = custom_metric

    weights = _compute_class_weights(y, normalization, metric)
    metrics = _compute_metrics(y, weights)

    return {
        'result': weights,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

################################################################################
# imbalanced_learn
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate input arrays for imbalanced learning.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)

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

def standardize(X: np.ndarray) -> np.ndarray:
    """
    Standardize features by removing the mean and scaling to unit variance.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)

    Returns
    ------
    np.ndarray
        Standardized feature matrix
    """
    return (X - X.mean(axis=0)) / X.std(axis=0)

def minmax_scale(X: np.ndarray) -> np.ndarray:
    """
    Scale features to a given range, usually [0, 1].

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)

    Returns
    ------
    np.ndarray
        Scaled feature matrix
    """
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

def calculate_class_weights(y: np.ndarray, method: str = 'balanced') -> np.ndarray:
    """
    Calculate class weights for imbalanced learning.

    Parameters
    ----------
    y : np.ndarray
        Target vector of shape (n_samples,)
    method : str
        Weighting method ('balanced' or 'custom')

    Returns
    ------
    np.ndarray
        Class weights of shape (n_classes,)
    """
    classes, counts = np.unique(y, return_counts=True)
    if method == 'balanced':
        weights = 1. / counts
    else:
        raise ValueError("Only 'balanced' method is currently supported")
    return weights[np.argsort(classes)]

def resample_data(X: np.ndarray, y: np.ndarray, method: str = 'oversampling',
                  sampling_ratio: float = 1.0) -> tuple:
    """
    Resample data to balance class distribution.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    method : str
        Resampling method ('oversampling' or 'undersampling')
    sampling_ratio : float
        Ratio for resampling

    Returns
    ------
    tuple
        (X_resampled, y_resampled)
    """
    classes = np.unique(y)
    X_resampled = []
    y_resampled = []

    for cls in classes:
        idx = np.where(y == cls)[0]
        X_cls = X[idx]
        y_cls = y[idx]

        if method == 'oversampling':
            n_samples = int(len(idx) * sampling_ratio)
            if len(idx) < n_samples:
                idx_oversample = np.random.choice(idx, size=n_samples - len(idx), replace=True)
                X_cls_oversample = X[idx_oversample]
                y_cls_oversample = y[idx_oversample]

                X_resampled.append(np.vstack([X_cls, X_cls_oversample]))
                y_resampled.append(np.hstack([y_cls, y_cls_oversample]))
            else:
                X_resampled.append(X_cls)
                y_resampled.append(y_cls)

        elif method == 'undersampling':
            n_samples = int(len(idx) / sampling_ratio)
            if len(idx) > n_samples:
                idx_undersample = np.random.choice(idx, size=n_samples, replace=False)
                X_resampled.append(X[idx_undersample])
                y_resampled.append(y[idx_undersample])
            else:
                X_resampled.append(X_cls)
                y_resampled.append(y_cls)

    return np.vstack(X_resampled), np.hstack(y_resampled)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """
    Calculate evaluation metrics for imbalanced learning.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted target values
    metric_funcs : Dict[str, Callable]
        Dictionary of metric functions

    Returns
    ------
    Dict[str, float]
        Dictionary of calculated metrics
    """
    return {name: func(y_true, y_pred) for name, func in metric_funcs.items()}

def imbalanced_learn_fit(X: np.ndarray, y: np.ndarray,
                         normalization: str = 'standard',
                         resampling_method: str = None,
                         sampling_ratio: float = 1.0,
                         metric_funcs: Dict[str, Callable] = None) -> Dict[str, Any]:
    """
    Main function for imbalanced learning with configurable options.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalization : str
        Normalization method ('none', 'standard', 'minmax')
    resampling_method : str
        Resampling method ('oversampling', 'undersampling') or None
    sampling_ratio : float
        Ratio for resampling
    metric_funcs : Dict[str, Callable]
        Dictionary of metric functions

    Returns
    ------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    validate_inputs(X, y)

    # Initialize output dictionary
    result = {
        'result': None,
        'metrics': {},
        'params_used': {
            'normalization': normalization,
            'resampling_method': resampling_method,
            'sampling_ratio': sampling_ratio
        },
        'warnings': []
    }

    # Apply normalization if specified
    if normalization == 'standard':
        X = standardize(X)
    elif normalization == 'minmax':
        X = minmax_scale(X)

    # Resample data if specified
    if resampling_method is not None:
        X, y = resample_data(X, y, method=resampling_method,
                            sampling_ratio=sampling_ratio)

    # Calculate class weights
    class_weights = calculate_class_weights(y)
    result['params_used']['class_weights'] = dict(zip(np.unique(y), class_weights))

    # Here you would typically fit a model and get predictions
    # For this example, we'll just return the processed data

    result['result'] = {
        'X_processed': X,
        'y_processed': y
    }

    # Calculate metrics if provided
    if metric_funcs is not None:
        result['metrics'] = calculate_metrics(y, y, metric_funcs)

    return result

# Example usage:
"""
from sklearn.metrics import accuracy_score, f1_score

def custom_metric(y_true, y_pred):
    return np.mean(y_true == y_pred)

X = np.random.rand(100, 5)
y = np.random.randint(0, 2, size=100)

metrics = {
    'accuracy': accuracy_score,
    'f1': f1_score,
    'custom': custom_metric
}

result = imbalanced_learn_fit(
    X=X,
    y=y,
    normalization='standard',
    resampling_method='oversampling',
    sampling_ratio=1.5,
    metric_funcs=metrics
)
"""

################################################################################
# resampling
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def resampling_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sampling_strategy: str = 'auto',
    replacement: bool = False,
    random_state: Optional[int] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'accuracy',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    **kwargs
) -> Dict[str, Union[Dict, float, np.ndarray]]:
    """
    Fit a resampling method to balance class distribution.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    sampling_strategy : str or dict
        Strategy to use for resampling. Can be 'auto', 'majority', 'minority',
        'not majority', 'not minority' or a dictionary.
    replacement : bool
        Whether to sample with replacement.
    random_state : int, optional
        Random seed for reproducibility.
    metric : str or callable
        Metric to evaluate resampling quality. Can be 'accuracy', 'f1', or a custom callable.
    normalizer : callable, optional
        Function to normalize features before resampling.

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': Resampled data (X_res, y_res)
        - 'metrics': Evaluation metrics
        - 'params_used': Parameters used
        - 'warnings': Any warnings generated

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = resampling_fit(X, y, sampling_strategy='minority')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set random state
    rng = np.random.RandomState(random_state)

    # Normalize features if requested
    X_norm = _apply_normalization(X, normalizer)

    # Determine sampling strategy
    if isinstance(sampling_strategy, str):
        strategy = _determine_sampling_strategy(y, sampling_strategy)
    else:
        strategy = sampling_strategy

    # Perform resampling
    X_res, y_res = _perform_resampling(X_norm, y, strategy, replacement, rng)

    # Calculate metrics
    metrics = _calculate_metrics(y_res, metric)

    return {
        'result': (X_res, y_res),
        'metrics': metrics,
        'params_used': {
            'sampling_strategy': sampling_strategy,
            'replacement': replacement,
            'random_state': random_state
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalization(
    X: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Apply normalization to features."""
    if normalizer is None:
        return X
    try:
        return normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _determine_sampling_strategy(
    y: np.ndarray,
    strategy: str
) -> Dict[int, int]:
    """Determine sampling strategy based on input."""
    classes, counts = np.unique(y, return_counts=True)
    if strategy == 'auto':
        majority_class = classes[np.argmax(counts)]
        return {c: max(counts) for c in classes}
    elif strategy == 'majority':
        majority_class = classes[np.argmax(counts)]
        return {c: counts[np.argmax(counts)] if c == majority_class else len(y) for c in classes}
    elif strategy == 'minority':
        minority_class = classes[np.argmin(counts)]
        return {c: counts[np.argmax(counts)] if c == minority_class else len(y) for c in classes}
    elif strategy == 'not majority':
        majority_class = classes[np.argmax(counts)]
        return {c: counts[np.argmin(counts)] if c != majority_class else len(y) for c in classes}
    elif strategy == 'not minority':
        minority_class = classes[np.argmin(counts)]
        return {c: counts[np.argmax(counts)] if c != minority_class else len(y) for c in classes}
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

def _perform_resampling(
    X: np.ndarray,
    y: np.ndarray,
    strategy: Dict[int, int],
    replacement: bool,
    rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray]:
    """Perform resampling according to strategy."""
    X_res = []
    y_res = []

    for cls, target_size in strategy.items():
        mask = (y == cls)
        X_cls = X[mask]
        y_cls = y[mask]

        if len(X_cls) >= target_size:
            indices = rng.choice(len(X_cls), size=target_size, replace=False)
        else:
            indices = rng.choice(len(X_cls), size=target_size, replace=True)

        X_res.append(X_cls[indices])
        y_res.append(y_cls[indices])

    return np.concatenate(X_res), np.concatenate(y_res)

def _calculate_metrics(
    y: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    classes, counts = np.unique(y, return_counts=True)
    proportions = {cls: count/len(y) for cls, count in zip(classes, counts)}

    if callable(metric):
        return {'custom_metric': metric(y, y)}
    elif metric == 'accuracy':
        return {'class_proportions': proportions}
    elif metric == 'f1':
        # This would require predicted labels, so we just return proportions
        return {'class_proportions': proportions}
    else:
        raise ValueError(f"Unknown metric: {metric}")

################################################################################
# data_augmentation
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def data_augmentation_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    distance_metric: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Main function for data augmentation to balance classes.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target labels array of shape (n_samples,)
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize features
    distance_metric : str
        Distance metric for augmentation ('euclidean', 'manhattan', 'cosine')
    solver : str
        Solver method ('closed_form', 'gradient_descent', 'newton')
    regularization : Optional[str]
        Regularization type (None, 'l1', 'l2', 'elasticnet')
    max_iter : int
        Maximum iterations for iterative solvers
    tol : float
        Tolerance for convergence
    random_state : Optional[int]
        Random seed for reproducibility
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = normalizer(X)

    # Get class distribution
    class_counts = np.bincount(y)
    n_classes = len(class_counts)

    # Initialize results dictionary
    results = {
        'result': None,
        'metrics': {},
        'params_used': {
            'normalizer': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
            'distance_metric': distance_metric,
            'solver': solver,
            'regularization': regularization,
            'max_iter': max_iter,
            'tol': tol
        },
        'warnings': []
    }

    # Check if augmentation is needed
    if np.all(class_counts == class_counts[0]):
        results['warnings'].append('All classes are already balanced')
        return results

    # Select solver
    if solver == 'closed_form':
        augmented_data = _closed_form_solver(X_normalized, y)
    elif solver == 'gradient_descent':
        augmented_data = _gradient_descent_solver(X_normalized, y,
                                                 distance_metric=distance_metric,
                                                 regularization=regularization,
                                                 max_iter=max_iter,
                                                 tol=tol)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    results['result'] = augmented_data
    _calculate_metrics(results, X_normalized, y, custom_metric)

    return results

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

def _closed_form_solver(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed form solution for data augmentation."""
    # Simple implementation - in practice would use more sophisticated methods
    unique_classes = np.unique(y)
    max_count = np.max(np.bincount(y))
    augmented_data = []

    for cls in unique_classes:
        class_mask = (y == cls)
        X_cls = X[class_mask]
        n_augment = max_count - len(X_cls)

        if n_augment > 0:
            # Simple duplication for demonstration
            augmented_samples = X_cls[np.random.randint(0, len(X_cls), size=n_augment)]
            augmented_data.append(augmented_samples)

    if augmented_data:
        return np.vstack([X] + augmented_data)
    return X

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    *,
    distance_metric: str = 'euclidean',
    regularization: Optional[str] = None,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> np.ndarray:
    """Gradient descent solver for data augmentation."""
    # Implementation would depend on specific distance metric and regularization
    # This is a placeholder implementation
    return _closed_form_solver(X, y)

def _calculate_metrics(
    results: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> None:
    """Calculate and store metrics."""
    if custom_metric is not None:
        results['metrics']['custom'] = custom_metric(X, y)

    # Add other metrics as needed

################################################################################
# cost_sensitive_learning
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    class_costs: Optional[np.ndarray] = None
) -> None:
    """
    Validate input data and parameters.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    sample_weights : Optional[np.ndarray]
        Sample weights of shape (n_samples,)
    class_costs : Optional[np.ndarray]
        Class costs vector of shape (n_classes,)

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
    if sample_weights is not None:
        if X.shape[0] != sample_weights.shape[0]:
            raise ValueError("X and sample_weights must have the same number of samples")
        if np.any(sample_weights < 0):
            raise ValueError("sample_weights must be non-negative")
    if class_costs is not None:
        unique_classes = np.unique(y)
        if class_costs.shape[0] != len(unique_classes):
            raise ValueError("class_costs must match number of unique classes")
        if np.any(class_costs <= 0):
            raise ValueError("class_costs must be positive")

def normalize_data(
    X: np.ndarray,
    method: str = 'standard',
    custom_normalizer: Optional[Callable] = None
) -> np.ndarray:
    """
    Normalize feature matrix.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    method : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    custom_normalizer : Optional[Callable]
        Custom normalization function

    Returns
    ------
    np.ndarray
        Normalized feature matrix
    """
    if custom_normalizer is not None:
        return custom_normalizer(X)

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

def compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = 'mse',
    custom_metric: Optional[Callable] = None,
    sample_weights: Optional[np.ndarray] = None
) -> float:
    """
    Compute evaluation metric.

    Parameters
    ----------
    y_true : np.ndarray
        True target values of shape (n_samples,)
    y_pred : np.ndarray
        Predicted values of shape (n_samples,)
    metric : str
        Metric to compute ('mse', 'mae', 'r2', 'logloss')
    custom_metric : Optional[Callable]
        Custom metric function
    sample_weights : Optional[np.ndarray]
        Sample weights of shape (n_samples,)

    Returns
    ------
    float
        Computed metric value
    """
    if custom_metric is not None:
        return custom_metric(y_true, y_pred)

    if sample_weights is not None:
        weights = sample_weights
    else:
        weights = np.ones_like(y_true)

    if metric == 'mse':
        return np.mean(weights * (y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(weights * np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum(weights * (y_true - y_pred) ** 2)
        ss_tot = np.sum(weights * (y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / (ss_tot + 1e-8)
    elif metric == 'logloss':
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        return -np.mean(weights * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def cost_sensitive_learning_fit(
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    class_costs: Optional[np.ndarray] = None,
    normalizer: str = 'standard',
    custom_normalizer: Optional[Callable] = None,
    metric: str = 'mse',
    custom_metric: Optional[Callable] = None,
    solver: str = 'closed_form',
    custom_solver: Optional[Callable] = None,
    regularization: str = 'none',
    reg_param: float = 1.0,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> Dict[str, Any]:
    """
    Fit a cost-sensitive learning model.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    sample_weights : Optional[np.ndarray]
        Sample weights of shape (n_samples,)
    class_costs : Optional[np.ndarray]
        Class costs vector of shape (n_classes,)
    normalizer : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    custom_normalizer : Optional[Callable]
        Custom normalization function
    metric : str
        Evaluation metric ('mse', 'mae', 'r2', 'logloss')
    custom_metric : Optional[Callable]
        Custom metric function
    solver : str
        Solver method ('closed_form', 'gradient_descent')
    custom_solver : Optional[Callable]
        Custom solver function
    regularization : str
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    reg_param : float
        Regularization parameter
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations

    Returns
    ------
    Dict[str, Any]
        Dictionary containing:
        - 'result': Model parameters
        - 'metrics': Computed metrics
        - 'params_used': Parameters used in fitting
        - 'warnings': Any warnings generated
    """
    # Validate inputs
    validate_inputs(X, y, sample_weights, class_costs)

    # Initialize output dictionary
    result = {
        'result': None,
        'metrics': {},
        'params_used': {
            'normalizer': normalizer if custom_normalizer is None else 'custom',
            'metric': metric if custom_metric is None else 'custom',
            'solver': solver if custom_solver is None else 'custom',
            'regularization': regularization,
            'reg_param': reg_param
        },
        'warnings': []
    }

    # Normalize data
    X_norm = normalize_data(X, normalizer, custom_normalizer)

    # Apply sample weights if provided
    if sample_weights is not None:
        X_norm = X_norm * np.sqrt(sample_weights[:, np.newaxis])
        y = y * sample_weights

    # Apply class costs if provided
    if class_costs is not None:
        unique_classes = np.unique(y)
        cost_dict = {cls: cost for cls, cost in zip(unique_classes, class_costs)}
        y = np.array([cost_dict[cls] for cls in y])

    # Solve the problem
    if custom_solver is not None:
        params = custom_solver(X_norm, y)
    elif solver == 'closed_form':
        XtX = np.dot(X_norm.T, X_norm)
        if regularization == 'l1':
            # Lasso solution would require different approach
            raise NotImplementedError("L1 regularization not implemented for closed form")
        elif regularization == 'l2':
            reg_matrix = reg_param * np.eye(XtX.shape[0])
            XtX += reg_matrix
        elif regularization == 'elasticnet':
            # ElasticNet solution would require different approach
            raise NotImplementedError("ElasticNet regularization not implemented for closed form")
        params = np.linalg.solve(XtX, np.dot(X_norm.T, y))
    elif solver == 'gradient_descent':
        # Simple gradient descent implementation
        n_samples, n_features = X_norm.shape
        params = np.zeros(n_features)
        learning_rate = 0.01

        for _ in range(max_iter):
            grad = np.dot(X_norm.T, (np.dot(X_norm, params) - y)) / n_samples
            if regularization == 'l1':
                grad += reg_param * np.sign(params)
            elif regularization == 'l2':
                grad += 2 * reg_param * params
            elif regularization == 'elasticnet':
                grad += reg_param * (np.sign(params) + 2 * params)

            new_params = params - learning_rate * grad
            if np.linalg.norm(new_params - params) < tol:
                break
            params = new_params
    else:
        raise ValueError(f"Unknown solver: {solver}")

    result['result'] = params

    # Compute metrics
    y_pred = np.dot(X_norm, params)
    result['metrics']['training_metric'] = compute_metric(y, y_pred, metric, custom_metric)

    return result
