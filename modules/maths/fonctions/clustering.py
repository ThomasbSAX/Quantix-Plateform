"""
Quantix – Module clustering
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# algorithmes
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data for clustering algorithms."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or infinite values")

def _normalize_data(data: np.ndarray, method: str = 'standard') -> np.ndarray:
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

def _compute_distance(data: np.ndarray, centers: np.ndarray,
                     metric: str = 'euclidean') -> np.ndarray:
    """Compute distance between data points and cluster centers."""
    if metric == 'euclidean':
        return np.sqrt(np.sum((data[:, np.newaxis, :] - centers) ** 2, axis=2))
    elif metric == 'manhattan':
        return np.sum(np.abs(data[:, np.newaxis, :] - centers), axis=2)
    elif metric == 'cosine':
        return 1 - np.sum(data[:, np.newaxis, :] * centers, axis=2) / (
            np.linalg.norm(data, axis=1)[:, np.newaxis] *
            np.linalg.norm(centers, axis=1) + 1e-8)
    elif metric == 'minkowski':
        return np.sum(np.abs(data[:, np.newaxis, :] - centers) ** 3, axis=2) ** (1/3)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _initialize_centers(data: np.ndarray, n_clusters: int,
                       method: str = 'random') -> np.ndarray:
    """Initialize cluster centers using specified method."""
    if method == 'random':
        indices = np.random.choice(data.shape[0], n_clusters, replace=False)
        return data[indices]
    elif method == 'kmeans++':
        centers = [data[np.random.randint(data.shape[0])]]
        for _ in range(1, n_clusters):
            dist_sq = np.min(_compute_distance(data[np.newaxis, :], np.array(centers), 'euclidean'), axis=0)
            probs = dist_sq / np.sum(dist_sq)
            cumulative_probs = np.cumsum(probs)
            r = np.random.rand()
            index = np.searchsorted(cumulative_probs, r)
            centers.append(data[index])
        return np.array(centers)
    else:
        raise ValueError(f"Unknown initialization method: {method}")

def _compute_metrics(data: np.ndarray, labels: np.ndarray,
                     centers: np.ndarray, metric: str) -> Dict[str, float]:
    """Compute clustering metrics."""
    distances = _compute_distance(data, centers, metric)
    correct_assignments = np.argmin(distances, axis=1) == labels
    metrics = {
        'inertia': np.sum(np.min(distances, axis=1) ** 2),
        'accuracy': np.mean(correct_assignments)
    }
    return metrics

def algorithmes_fit(data: np.ndarray, n_clusters: int,
                    normalization: str = 'standard',
                    distance_metric: str = 'euclidean',
                    initialization: str = 'kmeans++',
                    max_iterations: int = 300,
                    tolerance: float = 1e-4) -> Dict[str, Any]:
    """
    Perform clustering using specified parameters.

    Parameters:
    - data: Input data as numpy array
    - n_clusters: Number of clusters to form
    - normalization: Normalization method ('none', 'standard', 'minmax', 'robust')
    - distance_metric: Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski')
    - initialization: Initialization method ('random', 'kmeans++')
    - max_iterations: Maximum number of iterations
    - tolerance: Tolerance for convergence

    Returns:
    Dictionary containing results, metrics, parameters used and warnings
    """
    _validate_inputs(data)
    normalized_data = _normalize_data(data, normalization)
    centers = _initialize_centers(normalized_data, n_clusters, initialization)

    for _ in range(max_iterations):
        distances = _compute_distance(normalized_data, centers, distance_metric)
        labels = np.argmin(distances, axis=1)

        new_centers = np.array([normalized_data[labels == k].mean(axis=0)
                               for k in range(n_clusters)])
        new_centers = np.nan_to_num(new_centers)

        if np.linalg.norm(centers - new_centers) < tolerance:
            break

        centers = new_centers

    metrics = _compute_metrics(normalized_data, labels, centers, distance_metric)

    return {
        'result': {
            'labels': labels,
            'centers': centers
        },
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'distance_metric': distance_metric,
            'initialization': initialization
        },
        'warnings': []
    }

# Example usage:
# result = algorithmes_fit(data=np.random.rand(100, 5), n_clusters=3)

################################################################################
# distances
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_inputs(X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
    """Validate input arrays for distance calculations."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if y is not None and np.any(np.isnan(y)):
        raise ValueError("y contains NaN values")

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

def compute_distance(X: np.ndarray, y: Optional[np.ndarray] = None,
                    metric: str = 'euclidean',
                    custom_metric: Optional[Callable] = None) -> np.ndarray:
    """Compute pairwise distances between samples."""
    if custom_metric is not None:
        return custom_metric(X, y)

    X = np.asarray(X)
    if y is not None:
        y = np.asarray(y)

    n_samples = X.shape[0]
    distances = np.zeros((n_samples, n_samples))

    if metric == 'euclidean':
        for i in range(n_samples):
            distances[i] = np.sqrt(np.sum((X - X[i])**2, axis=1))
    elif metric == 'manhattan':
        for i in range(n_samples):
            distances[i] = np.sum(np.abs(X - X[i]), axis=1)
    elif metric == 'cosine':
        for i in range(n_samples):
            dot_products = np.dot(X, X[i])
            norms = np.linalg.norm(X, axis=1) * np.linalg.norm(X[i])
            distances[i] = 1 - (dot_products / (norms + 1e-8))
    elif metric == 'minkowski':
        p = 3
        for i in range(n_samples):
            distances[i] = np.sum(np.abs(X - X[i])**p, axis=1)**(1/p)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

    return distances

def distances_fit(X: np.ndarray, y: Optional[np.ndarray] = None,
                normalization: str = 'standard',
                metric: str = 'euclidean',
                custom_metric: Optional[Callable] = None,
                **kwargs) -> Dict:
    """Compute distances between samples with various options.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values if available
    normalization : str, default='standard'
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str, default='euclidean'
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski')
    custom_metric : Optional[Callable]
        Custom distance function if needed
    **kwargs :
        Additional parameters for the distance computation

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': computed distances matrix
        - 'metrics': dictionary of metrics
        - 'params_used': parameters used in computation
        - 'warnings': any warnings generated

    Example
    -------
    >>> X = np.random.rand(10, 5)
    >>> result = distances_fit(X, normalization='standard', metric='euclidean')
    """
    # Validate inputs
    validate_inputs(X, y)

    # Initialize output dictionary
    output = {
        'result': None,
        'metrics': {},
        'params_used': {
            'normalization': normalization,
            'metric': metric
        },
        'warnings': []
    }

    try:
        # Normalize data
        X_normalized = normalize_data(X, normalization)

        # Compute distances
        distances = compute_distance(X_normalized, y, metric, custom_metric)

        # Store results
        output['result'] = distances

    except Exception as e:
        output['warnings'].append(str(e))

    return output

################################################################################
# evaluation
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def evaluation_fit(
    data: np.ndarray,
    labels: np.ndarray,
    distance_metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    normalization: Optional[str] = None,
    metrics: Union[str, list, Callable[[np.ndarray, np.ndarray], float]] = 'silhouette',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluate clustering results using specified metrics and normalization.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    labels : np.ndarray
        Cluster labels of shape (n_samples,)
    distance_metric : str or callable, optional
        Distance metric to use ('euclidean', 'manhattan', 'cosine', etc.)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metrics : str, list or callable, optional
        Metrics to compute ('silhouette', 'davies_bouldin', etc.)
    custom_metric : callable, optional
        Custom metric function

    Returns:
    --------
    dict
        Dictionary containing evaluation results, metrics and parameters used
    """
    # Validate inputs
    _validate_inputs(data, labels)

    # Normalize data if specified
    normalized_data = _apply_normalization(data, normalization)

    # Compute distance matrix based on selected metric
    distance_matrix = _compute_distance_matrix(
        normalized_data,
        labels,
        distance_metric
    )

    # Compute selected metrics
    results = _compute_metrics(
        distance_matrix,
        labels,
        metrics,
        custom_metric
    )

    # Prepare output dictionary
    return {
        'result': results['metrics'],
        'metrics': list(results['metrics'].keys()),
        'params_used': {
            'distance_metric': distance_metric,
            'normalization': normalization
        },
        'warnings': results.get('warnings', [])
    }

def _validate_inputs(data: np.ndarray, labels: np.ndarray) -> None:
    """Validate input data and labels."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if not isinstance(labels, np.ndarray):
        raise TypeError("Labels must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")
    if labels.ndim != 1:
        raise ValueError("Labels must be a 1D array")
    if len(data) != len(labels):
        raise ValueError("Data and labels must have the same number of samples")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")

def _apply_normalization(
    data: np.ndarray,
    method: Optional[str]
) -> np.ndarray:
    """Apply specified normalization to data."""
    if method is None or method == 'none':
        return data

    normalized = data.copy()
    if method == 'standard':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized = (data - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        normalized = (data - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized

def _compute_distance_matrix(
    data: np.ndarray,
    labels: np.ndarray,
    metric: Union[str, Callable]
) -> np.ndarray:
    """Compute distance matrix using specified metric."""
    n_samples = data.shape[0]

    if callable(metric):
        distance_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                distance_matrix[i, j] = metric(data[i], data[j])
    else:
        if metric == 'euclidean':
            distance_matrix = np.sqrt(np.sum((data[:, np.newaxis] - data) ** 2, axis=2))
        elif metric == 'manhattan':
            distance_matrix = np.sum(np.abs(data[:, np.newaxis] - data), axis=2)
        elif metric == 'cosine':
            dot_products = np.dot(data, data.T)
            norms = np.sqrt(np.sum(data**2, axis=1))
            distance_matrix = 1 - (dot_products / np.outer(norms, norms))
        else:
            raise ValueError(f"Unknown distance metric: {metric}")

    return distance_matrix

def _compute_metrics(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    metrics: Union[str, list],
    custom_metric: Optional[Callable]
) -> Dict[str, Any]:
    """Compute specified clustering metrics."""
    results = {'metrics': {}, 'warnings': []}

    if isinstance(metrics, str):
        metrics = [metrics]

    if 'silhouette' in metrics:
        results['metrics']['silhouette'] = _compute_silhouette(distance_matrix, labels)

    if 'davies_bouldin' in metrics:
        results['metrics']['davies_bouldin'] = _compute_davies_bouldin(distance_matrix, labels)

    if custom_metric is not None:
        results['metrics']['custom'] = custom_metric(distance_matrix, labels)

    return results

def _compute_silhouette(
    distance_matrix: np.ndarray,
    labels: np.ndarray
) -> float:
    """Compute silhouette score."""
    n_samples = distance_matrix.shape[0]
    unique_labels = np.unique(labels)
    silhouette_scores = np.zeros(n_samples)

    for i in range(n_samples):
        # Compute a(i)
        cluster_i = labels == labels[i]
        if np.sum(cluster_i) > 1:
            a_i = np.mean(distance_matrix[i, cluster_i])
        else:
            a_i = 0

        # Compute b(i)
        min_b_i = float('inf')
        for j in unique_labels:
            if j != labels[i]:
                cluster_j = labels == j
                b_i = np.mean(distance_matrix[i, cluster_j])
                if b_i < min_b_i:
                    min_b_i = b_i

        # Compute silhouette score
        if max(a_i, min_b_i) != 0:
            silhouette_scores[i] = (min_b_i - a_i) / max(a_i, min_b_i)
        else:
            silhouette_scores[i] = 0

    return np.mean(silhouette_scores)

def _compute_davies_bouldin(
    distance_matrix: np.ndarray,
    labels: np.ndarray
) -> float:
    """Compute Davies-Bouldin index."""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Compute intra-cluster distances
    intra_distances = np.zeros(n_clusters)
    for i, label in enumerate(unique_labels):
        cluster_mask = labels == label
        if np.sum(cluster_mask) > 1:
            intra_distances[i] = np.mean(distance_matrix[cluster_mask, :][:, cluster_mask])

    # Compute inter-cluster distances
    inter_distances = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            cluster_i = labels == unique_labels[i]
            cluster_j = labels == unique_labels[j]
            inter_distances[i, j] = np.mean(distance_matrix[cluster_i, :][:, cluster_j])
            inter_distances[j, i] = inter_distances[i, j]

    # Compute Davies-Bouldin index
    db_index = 0.0
    for i in range(n_clusters):
        max_ratio = 0.0
        for j in range(n_clusters):
            if i != j:
                ratio = (intra_distances[i] + intra_distances[j]) / inter_distances[i, j]
                if ratio > max_ratio:
                    max_ratio = ratio
        db_index += max_ratio

    return db_index / n_clusters if n_clusters > 1 else float('inf')

################################################################################
# normalisation
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data for normalization."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def _standard_normalization(data: np.ndarray) -> np.ndarray:
    """Apply standard normalization (z-score)."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / (std + 1e-8)

def _minmax_normalization(data: np.ndarray) -> np.ndarray:
    """Apply min-max normalization."""
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    return (data - min_vals) / (max_vals - min_vals + 1e-8)

def _robust_normalization(data: np.ndarray) -> np.ndarray:
    """Apply robust normalization using median and IQR."""
    median = np.median(data, axis=0)
    q75, q25 = np.percentile(data, [75, 25], axis=0)
    iqr = q75 - q25
    return (data - median) / (iqr + 1e-8)

def _custom_normalization(data: np.ndarray, func: Callable) -> np.ndarray:
    """Apply custom normalization function."""
    return func(data)

def normalisation_fit(
    data: np.ndarray,
    method: str = "standard",
    custom_func: Optional[Callable] = None,
    metric: str = "mse",
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Apply normalization to data for clustering purposes.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_samples, n_features)
    method : str, optional
        Normalization method: "none", "standard", "minmax", "robust"
    custom_func : Callable, optional
        Custom normalization function if method="custom"
    metric : str, optional
        Metric to evaluate normalization quality: "mse", "mae", "r2"
    **kwargs
        Additional parameters for the normalization method

    Returns
    -------
    dict
        Dictionary containing:
        - "result": normalized data
        - "metrics": evaluation metrics
        - "params_used": parameters used
        - "warnings": any warnings

    Example
    -------
    >>> data = np.random.rand(100, 5)
    >>> result = normalisation_fit(data, method="standard")
    """
    _validate_inputs(data)

    warnings = []
    params_used = {
        "method": method,
        "custom_func": custom_func is not None,
        **kwargs
    }

    if method == "none":
        normalized_data = data.copy()
    elif method == "standard":
        normalized_data = _standard_normalization(data)
    elif method == "minmax":
        normalized_data = _minmax_normalization(data)
    elif method == "robust":
        normalized_data = _robust_normalization(data)
    elif method == "custom" and custom_func is not None:
        normalized_data = _custom_normalization(data, custom_func)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Calculate metrics
    original_mean = np.mean(data, axis=0)
    normalized_mean = np.mean(normalized_data, axis=0)
    original_std = np.std(data, axis=0)
    normalized_std = np.std(normalized_data, axis=0)

    metrics = {}
    if metric == "mse":
        metrics["mse"] = np.mean((normalized_data - data) ** 2)
    elif metric == "mae":
        metrics["mae"] = np.mean(np.abs(normalized_data - data))
    elif metric == "r2":
        ss_total = np.sum((data - original_mean) ** 2)
        ss_residual = np.sum((normalized_data - data) ** 2)
        metrics["r2"] = 1 - (ss_residual / ss_total)

    # Check for potential issues
    if np.any(np.isnan(normalized_data)):
        warnings.append("Normalization resulted in NaN values")
    if np.any(np.isinf(normalized_data)):
        warnings.append("Normalization resulted in infinite values")

    return {
        "result": normalized_data,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

################################################################################
# preprocessing
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def preprocessing_fit(
    data: np.ndarray,
    normalization: str = 'standard',
    distance_metric: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    custom_metric: Optional[Callable] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Main preprocessing function for clustering.

    Parameters:
    -----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features)
    normalization : str, optional
        Normalization method: 'none', 'standard', 'minmax', 'robust'
    distance_metric : str or callable, optional
        Distance metric: 'euclidean', 'manhattan', 'cosine', 'minkowski' or custom callable
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', 'newton', 'coordinate_descent'
    custom_metric : callable, optional
        Custom metric function if needed
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> data = np.random.rand(100, 5)
    >>> result = preprocessing_fit(data, normalization='standard', distance_metric='euclidean')
    """
    # Validate inputs
    _validate_inputs(data, normalization, distance_metric)

    # Initialize random state if provided
    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    # Normalize data
    normalized_data = _apply_normalization(data, normalization)

    # Get distance function
    distance_func = _get_distance_function(distance_metric, rng)

    # Solve using selected method
    if solver == 'closed_form':
        result = _solve_closed_form(normalized_data, distance_func)
    elif solver == 'gradient_descent':
        result = _solve_gradient_descent(normalized_data, distance_func, tol, max_iter)
    elif solver == 'newton':
        result = _solve_newton(normalized_data, distance_func, tol, max_iter)
    elif solver == 'coordinate_descent':
        result = _solve_coordinate_descent(normalized_data, distance_func, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(result, normalized_data, distance_func)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'distance_metric': distance_metric,
            'solver': solver,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

def _validate_inputs(
    data: np.ndarray,
    normalization: str,
    distance_metric: Union[str, Callable]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be 2-dimensional")
    if np.isnan(data).any():
        raise ValueError("Data contains NaN values")
    if np.isinf(data).any():
        raise ValueError("Data contains infinite values")

    valid_normalizations = ['none', 'standard', 'minmax', 'robust']
    if normalization not in valid_normalizations:
        raise ValueError(f"Unknown normalization: {normalization}")

    if isinstance(distance_metric, str):
        valid_distances = ['euclidean', 'manhattan', 'cosine', 'minkowski']
        if distance_metric not in valid_distances:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

def _apply_normalization(
    data: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply selected normalization to data."""
    if method == 'none':
        return data.copy()

    normalized = data.copy()
    if method == 'standard':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized = (data - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        normalized = (data - median) / (iqr + 1e-8)

    return normalized

def _get_distance_function(
    metric: Union[str, Callable],
    rng: np.random.RandomState
) -> Callable:
    """Get distance function based on metric specification."""
    if isinstance(metric, str):
        if metric == 'euclidean':
            return lambda x, y: np.linalg.norm(x - y)
        elif metric == 'manhattan':
            return lambda x, y: np.sum(np.abs(x - y))
        elif metric == 'cosine':
            return lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        elif metric == 'minkowski':
            return lambda x, y: np.sum(np.abs(x - y)**3)**(1/3)
    else:
        return metric

    raise ValueError(f"Unknown distance metric: {metric}")

def _solve_closed_form(
    data: np.ndarray,
    distance_func: Callable
) -> Dict[str, Any]:
    """Solve using closed form solution."""
    # This is a placeholder for actual implementation
    centers = data[np.random.choice(data.shape[0], size=min(5, data.shape[0]), replace=False)]
    return {'centers': centers}

def _solve_gradient_descent(
    data: np.ndarray,
    distance_func: Callable,
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Solve using gradient descent."""
    # This is a placeholder for actual implementation
    centers = data[np.random.choice(data.shape[0], size=min(5, data.shape[0]), replace=False)]
    for _ in range(max_iter):
        # Update centers
        pass
    return {'centers': centers}

def _solve_newton(
    data: np.ndarray,
    distance_func: Callable,
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Solve using Newton's method."""
    # This is a placeholder for actual implementation
    centers = data[np.random.choice(data.shape[0], size=min(5, data.shape[0]), replace=False)]
    return {'centers': centers}

def _solve_coordinate_descent(
    data: np.ndarray,
    distance_func: Callable,
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Solve using coordinate descent."""
    # This is a placeholder for actual implementation
    centers = data[np.random.choice(data.shape[0], size=min(5, data.shape[0]), replace=False)]
    return {'centers': centers}

def _calculate_metrics(
    result: Dict[str, Any],
    data: np.ndarray,
    distance_func: Callable
) -> Dict[str, float]:
    """Calculate clustering metrics."""
    # This is a placeholder for actual implementation
    return {
        'inertia': 0.0,
        'silhouette_score': 0.0
    }

################################################################################
# visualisation
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def visualisation_fit(
    data: np.ndarray,
    labels: np.ndarray,
    distance_metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    normalization: str = 'none',
    cluster_centers: Optional[np.ndarray] = None,
    custom_metrics: Optional[Dict[str, Callable]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Fonction principale pour la visualisation des résultats de clustering.

    Parameters:
    -----------
    data : np.ndarray
        Données d'entrée pour le clustering.
    labels : np.ndarray
        Étiquettes des clusters obtenues après clustering.
    distance_metric : str or callable, optional
        Métrique de distance à utiliser ('euclidean', 'manhattan', 'cosine', etc.) ou une fonction personnalisée.
    normalization : str, optional
        Type de normalisation ('none', 'standard', 'minmax', 'robust').
    cluster_centers : np.ndarray, optional
        Centres des clusters. Si None, ils seront calculés.
    custom_metrics : dict, optional
        Dictionnaire de métriques personnalisées à utiliser.

    Returns:
    --------
    dict
        Dictionnaire contenant les résultats, métriques, paramètres utilisés et avertissements.
    """
    # Validation des entrées
    _validate_inputs(data, labels)

    # Normalisation des données
    normalized_data = _apply_normalization(data, normalization)

    # Calcul des centres de clusters si non fournis
    if cluster_centers is None:
        cluster_centers = _compute_cluster_centers(normalized_data, labels)

    # Calcul des métriques
    metrics = _compute_metrics(normalized_data, labels, cluster_centers, distance_metric, custom_metrics)

    # Visualisation des résultats
    visualization_result = _plot_clusters(normalized_data, labels, cluster_centers)

    return {
        "result": visualization_result,
        "metrics": metrics,
        "params_used": {
            "distance_metric": distance_metric,
            "normalization": normalization
        },
        "warnings": []
    }

def _validate_inputs(data: np.ndarray, labels: np.ndarray) -> None:
    """
    Valide les entrées pour la visualisation.

    Parameters:
    -----------
    data : np.ndarray
        Données d'entrée.
    labels : np.ndarray
        Étiquettes des clusters.

    Raises:
    -------
    ValueError
        Si les dimensions ou types ne sont pas valides.
    """
    if not isinstance(data, np.ndarray) or not isinstance(labels, np.ndarray):
        raise ValueError("Les données et les étiquettes doivent être des tableaux NumPy.")
    if data.shape[0] != labels.shape[0]:
        raise ValueError("Le nombre de points doit correspondre au nombre d'étiquettes.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Les données ne doivent pas contenir de NaN ou d'inf.")

def _apply_normalization(data: np.ndarray, normalization: str) -> np.ndarray:
    """
    Applique la normalisation spécifiée aux données.

    Parameters:
    -----------
    data : np.ndarray
        Données d'entrée.
    normalization : str
        Type de normalisation ('none', 'standard', 'minmax', 'robust').

    Returns:
    --------
    np.ndarray
        Données normalisées.
    """
    if normalization == 'none':
        return data
    elif normalization == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif normalization == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif normalization == 'robust':
        return (data - np.median(data, axis=0)) / (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0))
    else:
        raise ValueError(f"Normalisation non reconnue: {normalization}")

def _compute_cluster_centers(data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Calcule les centres des clusters.

    Parameters:
    -----------
    data : np.ndarray
        Données d'entrée.
    labels : np.ndarray
        Étiquettes des clusters.

    Returns:
    --------
    np.ndarray
        Centres des clusters.
    """
    unique_labels = np.unique(labels)
    cluster_centers = np.array([data[labels == label].mean(axis=0) for label in unique_labels])
    return cluster_centers

def _compute_metrics(
    data: np.ndarray,
    labels: np.ndarray,
    cluster_centers: np.ndarray,
    distance_metric: Union[str, Callable],
    custom_metrics: Optional[Dict[str, Callable]]
) -> Dict[str, float]:
    """
    Calcule les métriques de clustering.

    Parameters:
    -----------
    data : np.ndarray
        Données d'entrée.
    labels : np.ndarray
        Étiquettes des clusters.
    cluster_centers : np.ndarray
        Centres des clusters.
    distance_metric : str or callable
        Métrique de distance à utiliser.
    custom_metrics : dict, optional
        Dictionnaire de métriques personnalisées.

    Returns:
    --------
    dict
        Dictionnaire des métriques calculées.
    """
    metrics = {}

    # Calcul de la distance moyenne intra-cluster
    if isinstance(distance_metric, str):
        distance_func = _get_distance_function(distance_metric)
    else:
        distance_func = distance_metric

    intra_cluster_distances = []
    for label in np.unique(labels):
        cluster_data = data[labels == label]
        center = cluster_centers[label]
        distances = [distance_func(point, center) for point in cluster_data]
        intra_cluster_distances.extend(distances)

    metrics['mean_intra_cluster_distance'] = np.mean(intra_cluster_distances)

    # Calcul des métriques personnalisées si fournies
    if custom_metrics:
        for name, metric_func in custom_metrics.items():
            metrics[name] = metric_func(data, labels)

    return metrics

def _get_distance_function(metric_name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Retourne la fonction de distance spécifiée.

    Parameters:
    -----------
    metric_name : str
        Nom de la métrique de distance.

    Returns:
    --------
    callable
        Fonction de distance.
    """
    if metric_name == 'euclidean':
        return lambda x, y: np.linalg.norm(x - y)
    elif metric_name == 'manhattan':
        return lambda x, y: np.sum(np.abs(x - y))
    elif metric_name == 'cosine':
        return lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    elif metric_name == 'minkowski':
        return lambda x, y: np.sum(np.abs(x - y) ** 3) ** (1/3)
    else:
        raise ValueError(f"Métrique de distance non reconnue: {metric_name}")

def _plot_clusters(data: np.ndarray, labels: np.ndarray, cluster_centers: np.ndarray) -> Dict[str, Any]:
    """
    Génère la visualisation des clusters.

    Parameters:
    -----------
    data : np.ndarray
        Données d'entrée.
    labels : np.ndarray
        Étiquettes des clusters.
    cluster_centers : np.ndarray
        Centres des clusters.

    Returns:
    --------
    dict
        Dictionnaire contenant les résultats de la visualisation.
    """
    # Exemple simplifié de visualisation
    visualization_result = {
        'data': data.tolist(),
        'labels': labels.tolist(),
        'cluster_centers': cluster_centers.tolist()
    }
    return visualization_result
