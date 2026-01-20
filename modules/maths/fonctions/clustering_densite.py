"""
Quantix – Module clustering_densite
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# DBSCAN
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input contains NaN or infinite values")

def default_distance_metric(x: np.ndarray, y: np.ndarray) -> float:
    """Default Euclidean distance metric."""
    return np.linalg.norm(x - y)

def DBSCAN_fit(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: Union[str, Callable] = "euclidean",
    normalize: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform DBSCAN clustering.

    Parameters:
    - X: Input data (n_samples, n_features)
    - eps: Maximum distance between two samples for one to be considered in the neighborhood of the other
    - min_samples: Number of samples in a neighborhood for a point to be considered as a core point
    - metric: Distance metric ('euclidean', 'manhattan', 'cosine', or custom callable)
    - normalize: Normalization method ('none', 'standard', 'minmax', 'robust')

    Returns:
    - Dictionary containing clustering results, metrics, parameters used, and warnings
    """
    # Validate input
    validate_input(X)

    # Normalize data if specified
    if normalize is not None:
        X = _normalize_data(X, method=normalize)

    # Set default metric if string provided
    if isinstance(metric, str):
        if metric == "euclidean":
            distance_func = default_distance_metric
        elif metric == "manhattan":
            distance_func = lambda x, y: np.sum(np.abs(x - y))
        elif metric == "cosine":
            distance_func = lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        distance_func = metric
    else:
        raise TypeError("Metric must be a string or callable")

    # Perform DBSCAN clustering
    labels = _dbscan_core(X, eps, min_samples, distance_func)

    # Calculate metrics
    metrics = _calculate_metrics(X, labels)

    return {
        "result": {"labels": labels},
        "metrics": metrics,
        "params_used": {
            "eps": eps,
            "min_samples": min_samples,
            "metric": metric if isinstance(metric, str) else "custom",
            "normalize": normalize
        },
        "warnings": []
    }

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize data using specified method."""
    if method == "standard":
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == "minmax":
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == "robust":
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    return X

def _dbscan_core(
    X: np.ndarray,
    eps: float,
    min_samples: int,
    distance_func: Callable
) -> np.ndarray:
    """Core DBSCAN algorithm implementation."""
    n_samples = X.shape[0]
    labels = np.full(n_samples, -1, dtype=int)  # -1: noise, 0: unclassified
    cluster_id = 0

    for i in range(n_samples):
        if labels[i] != -1:
            continue  # Already classified

        neighbors = _region_query(X, i, eps, distance_func)
        if len(neighbors) < min_samples:
            labels[i] = -1  # Mark as noise
        else:
            cluster_id += 1
            labels[i] = cluster_id
            _expand_cluster(X, i, neighbors, eps, min_samples, distance_func, labels, cluster_id)

    return labels

def _region_query(
    X: np.ndarray,
    query_index: int,
    eps: float,
    distance_func: Callable
) -> np.ndarray:
    """Find all points within eps distance of the query point."""
    distances = np.array([distance_func(X[query_index], x) for x in X])
    return np.where(distances <= eps)[0]

def _expand_cluster(
    X: np.ndarray,
    query_index: int,
    neighbors: np.ndarray,
    eps: float,
    min_samples: int,
    distance_func: Callable,
    labels: np.ndarray,
    cluster_id: int
) -> None:
    """Expand the cluster from the query point."""
    i = 0
    while i < len(neighbors):
        neighbor_idx = neighbors[i]

        if labels[neighbor_idx] == -1:
            labels[neighbor_idx] = cluster_id  # Change noise to border point

        if labels[neighbor_idx] == -1 or labels[neighbor_idx] != cluster_id:
            labels[neighbor_idx] = cluster_id
            new_neighbors = _region_query(X, neighbor_idx, eps, distance_func)

            if len(new_neighbors) >= min_samples:
                neighbors = np.concatenate((neighbors, new_neighbors))

        i += 1

def _calculate_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Calculate clustering metrics."""
    n_clusters = len(np.unique(labels))
    if -1 in labels:
        n_clusters -= 1

    return {
        "n_clusters": n_clusters,
        "n_noise_points": np.sum(labels == -1)
    }

################################################################################
# OPTICS
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("Input X must be a 2D array.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input X must not contain NaN or infinite values.")

def default_distance_metric(x: np.ndarray, y: np.ndarray) -> float:
    """Default Euclidean distance metric."""
    return np.linalg.norm(x - y)

def compute_reachability_distance(
    X: np.ndarray,
    idx1: int,
    idx2: int,
    core_distances: np.ndarray,
    distance_metric: Callable[[np.ndarray, np.ndarray], float]
) -> float:
    """Compute reachability distance between two points."""
    if core_distances[idx1] > distance_metric(X[idx1], X[idx2]):
        return core_distances[idx1]
    else:
        return distance_metric(X[idx1], X[idx2])

def OPTICS_fit(
    X: np.ndarray,
    min_samples: int = 5,
    max_eps: float = np.inf,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = default_distance_metric,
    normalize: Optional[str] = None
) -> Dict[str, Any]:
    """
    OPTICS clustering algorithm.

    Parameters:
    - X: Input data (n_samples, n_features)
    - min_samples: Minimum number of samples in a neighborhood for a point to be considered as a core point.
    - max_eps: Maximum distance between two samples for one to be considered in the neighborhood of the other.
    - distance_metric: Distance metric function (default is Euclidean).
    - normalize: Normalization method ('standard', 'minmax', 'robust') or None.

    Returns:
    - Dictionary containing clustering results, metrics, parameters used, and warnings.
    """
    validate_inputs(X)

    if normalize == 'standard':
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalize == 'minmax':
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalize == 'robust':
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))

    n_samples = X.shape[0]
    reachability_distances = np.full(n_samples, np.inf)
    core_distances = np.full(n_samples, np.inf)
    ordering = []
    cluster_ordering = []

    for i in range(n_samples):
        neighbors = []
        for j in range(n_samples):
            if distance_metric(X[i], X[j]) <= max_eps:
                neighbors.append(j)

        if len(neighbors) >= min_samples:
            core_distances[i] = np.max([distance_metric(X[i], X[j]) for j in neighbors])
        else:
            core_distances[i] = np.inf

    current_point = 0
    while current_point < n_samples:
        if core_distances[current_point] == np.inf:
            ordering.append(current_point)
            current_point += 1
        else:
            seed_set = [current_point]
            ordering.append(current_point)
            cluster_ordering.append(len(ordering) - 1)

            while seed_set:
                current_seed = seed_set.pop(0)
                neighbors = []
                for j in range(n_samples):
                    if distance_metric(X[current_seed], X[j]) <= max_eps:
                        neighbors.append(j)

                for neighbor in neighbors:
                    if core_distances[neighbor] != np.inf and neighbor not in ordering:
                        reachability_distances[neighbor] = compute_reachability_distance(
                            X, current_seed, neighbor, core_distances, distance_metric
                        )
                        seed_set.append(neighbor)
                        ordering.append(neighbor)
                        cluster_ordering.append(len(ordering) - 1)

    result = {
        "ordering": np.array(ordering),
        "reachability_distances": reachability_distances,
        "core_distances": core_distances,
        "cluster_ordering": np.array(cluster_ordering)
    }

    metrics = {
        "n_clusters": len(np.unique(cluster_ordering)),
        "silhouette_score": None  # Can be computed if needed
    }

    params_used = {
        "min_samples": min_samples,
        "max_eps": max_eps,
        "distance_metric": distance_metric.__name__ if hasattr(distance_metric, '__name__') else "custom",
        "normalize": normalize
    }

    warnings = []

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

# Example usage:
# X = np.random.rand(100, 2)
# result = OPTICS_fit(X, min_samples=5, max_eps=0.5)

################################################################################
# HDBSCAN
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(X: np.ndarray) -> None:
    """Validate input data for HDBSCAN clustering.

    Args:
        X: Input data array of shape (n_samples, n_features).

    Raises:
        ValueError: If input data is invalid.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("Input data must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("Input data must be 2-dimensional.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input data contains NaN or infinite values.")

def default_distance_metric(x: np.ndarray, y: np.ndarray) -> float:
    """Default Euclidean distance metric.

    Args:
        x: First data point.
        y: Second data point.

    Returns:
        Euclidean distance between x and y.
    """
    return np.linalg.norm(x - y)

def compute_mutual_reachability_distance(
    X: np.ndarray,
    distance_metric: Callable[[np.ndarray, np.ndarray], float],
    min_cluster_size: int = 5,
    alpha: float = 1.0
) -> np.ndarray:
    """Compute mutual reachability distance matrix.

    Args:
        X: Input data array.
        distance_metric: Distance metric function.
        min_cluster_size: Minimum cluster size.
        alpha: Parameter for distance scaling.

    Returns:
        Mutual reachability distance matrix.
    """
    n_samples = X.shape[0]
    mrd_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            # Compute pairwise distances
            dist_ij = distance_metric(X[i], X[j])

            # Find minpts neighbors
            distances_i = np.array([distance_metric(X[i], x) for x in X])
            distances_j = np.array([distance_metric(X[j], x) for x in X])

            # Sort distances and find minpts neighbors
            sorted_distances_i = np.sort(distances_i)
            sorted_distances_j = np.sort(distances_j)

            # Compute mutual reachability distance
            if i == j:
                mrd_matrix[i, j] = 0.0
            else:
                # Use the larger of the two core distances
                core_dist_i = sorted_distances_i[min_cluster_size - 1]
                core_dist_j = sorted_distances_j[min_cluster_size - 1]

                mrd_matrix[i, j] = max(dist_ij, core_dist_i, core_dist_j)
                mrd_matrix[j, i] = mrd_matrix[i, j]

    return mrd_matrix

def extract_dense_regions(
    mrd_matrix: np.ndarray,
    min_cluster_size: int,
    min_samples: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Extract dense regions from mutual reachability distance matrix.

    Args:
        mrd_matrix: Mutual reachability distance matrix.
        min_cluster_size: Minimum cluster size.
        min_samples: Minimum number of samples in a neighborhood.

    Returns:
        Dictionary containing cluster labels and other information.
    """
    if min_samples is None:
        min_samples = min_cluster_size

    n_samples = mrd_matrix.shape[0]
    labels = np.zeros(n_samples, dtype=int)
    cluster_id = 1

    for i in range(n_samples):
        if labels[i] == 0:
            # Find neighbors
            neighbors = np.where(mrd_matrix[i] <= mrd_matrix[i, i])[0]
            if len(neighbors) >= min_samples:
                # Expand cluster
                labels[neighbors] = cluster_id
                cluster_id += 1

    return {"labels": labels, "n_clusters": np.max(labels) if np.max(labels) > 0 else 0}

def merge_clusters(
    labels: np.ndarray,
    mrd_matrix: np.ndarray,
    min_cluster_size: int
) -> np.ndarray:
    """Merge clusters based on mutual reachability distance.

    Args:
        labels: Cluster labels.
        mrd_matrix: Mutual reachability distance matrix.
        min_cluster_size: Minimum cluster size.

    Returns:
        Merged cluster labels.
    """
    n_clusters = np.max(labels) if np.max(labels) > 0 else 0
    if n_clusters == 0:
        return labels

    # Create adjacency matrix between clusters
    adjacency = np.zeros((n_clusters, n_clusters))

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if labels[i] != labels[j]:
                cluster_i = labels[i]
                cluster_j = labels[j]
                adjacency[cluster_i - 1, cluster_j - 1] = mrd_matrix[i, j]

    # Merge clusters based on adjacency
    merged_labels = labels.copy()
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            if adjacency[i, j] <= mrd_matrix[np.where(labels == i + 1)[0][0], np.where(labels == j + 1)[0][0]]:
                merged_labels[labels == j + 1] = i + 1

    return merged_labels

def HDBSCAN_fit(
    X: np.ndarray,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = default_distance_metric,
    min_cluster_size: int = 5,
    alpha: float = 1.0,
    min_samples: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, int, str]]:
    """Perform HDBSCAN clustering on input data.

    Args:
        X: Input data array of shape (n_samples, n_features).
        distance_metric: Distance metric function.
        min_cluster_size: Minimum cluster size.
        alpha: Parameter for distance scaling.
        min_samples: Minimum number of samples in a neighborhood.

    Returns:
        Dictionary containing clustering results, metrics, and parameters used.
    """
    validate_input(X)

    # Compute mutual reachability distance matrix
    mrd_matrix = compute_mutual_reachability_distance(
        X, distance_metric, min_cluster_size, alpha
    )

    # Extract dense regions
    result = extract_dense_regions(mrd_matrix, min_cluster_size, min_samples)

    # Merge clusters
    merged_labels = merge_clusters(result["labels"], mrd_matrix, min_cluster_size)

    # Calculate metrics
    n_noise = np.sum(merged_labels == 0)
    silhouette_score = calculate_silhouette_score(X, merged_labels, distance_metric)

    return {
        "result": {"labels": merged_labels},
        "metrics": {
            "n_clusters": np.max(merged_labels) if np.max(merged_labels) > 0 else 0,
            "n_noise": n_noise,
            "silhouette_score": silhouette_score
        },
        "params_used": {
            "min_cluster_size": min_cluster_size,
            "alpha": alpha,
            "min_samples": min_samples if min_samples is not None else min_cluster_size
        },
        "warnings": []
    }

def calculate_silhouette_score(
    X: np.ndarray,
    labels: np.ndarray,
    distance_metric: Callable[[np.ndarray, np.ndarray], float]
) -> float:
    """Calculate silhouette score for clustering results.

    Args:
        X: Input data array.
        labels: Cluster labels.
        distance_metric: Distance metric function.

    Returns:
        Silhouette score.
    """
    n_samples = X.shape[0]
    silhouette_scores = np.zeros(n_samples)

    for i in range(n_samples):
        if labels[i] == -1:
            continue

        # Compute a(i)
        cluster_mask = (labels == labels[i])
        if np.sum(cluster_mask) > 1:
            a_i = np.mean([distance_metric(X[i], X[j]) for j in range(n_samples) if cluster_mask[j]])
        else:
            a_i = 0.0

        # Compute b(i)
        other_clusters = np.unique(labels[labels != labels[i]])
        if len(other_clusters) == 0:
            b_i = 0.0
        else:
            min_b = float('inf')
            for cluster in other_clusters:
                cluster_mask = (labels == cluster)
                if np.sum(cluster_mask) > 0:
                    b_i = np.mean([distance_metric(X[i], X[j]) for j in range(n_samples) if cluster_mask[j]])
                    if b_i < min_b:
                        min_b = b_i
            b_i = min_b

        # Compute silhouette score for sample i
        if a_i < b_i:
            silhouette_scores[i] = 1 - (a_i / b_i)
        else:
            silhouette_scores[i] = (b_i / a_i) - 1

    return np.mean(silhouette_scores)

# Example usage:
"""
import numpy as np
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Run HDBSCAN clustering
result = HDBSCAN_fit(X, min_cluster_size=5)

# Access results
labels = result["result"]["labels"]
n_clusters = result["metrics"]["n_clusters"]
silhouette_score = result["metrics"]["silhouette_score"]

print(f"Number of clusters: {n_clusters}")
print(f"Silhouette score: {silhouette_score:.3f}")
"""

################################################################################
# densite_estimation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def densite_estimation_fit(
    data: np.ndarray,
    bandwidth: float = 1.0,
    kernel: Callable[[np.ndarray], np.ndarray] = lambda x: (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2),
    normalization: str = 'none',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric_params: Optional[Dict] = None,
    custom_kernel_params: Optional[Dict] = None
) -> Dict:
    """
    Estimate the density of data points using kernel density estimation.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_samples, n_features).
    bandwidth : float, optional
        Bandwidth for the kernel density estimation.
    kernel : Callable[[np.ndarray], np.ndarray], optional
        Kernel function. Default is Gaussian.
    normalization : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Distance metric: 'euclidean', 'manhattan', 'cosine', or custom callable.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', etc.
    regularization : Optional[str], optional
        Regularization method: 'none', 'l1', 'l2', or 'elasticnet'.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric_params : Optional[Dict], optional
        Parameters for custom metric function.
    custom_kernel_params : Optional[Dict], optional
        Parameters for custom kernel function.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Estimated density values.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used in the estimation.
        - 'warnings': Any warnings encountered.

    Examples
    --------
    >>> data = np.random.randn(100, 2)
    >>> result = densite_estimation_fit(data, bandwidth=0.5)
    """
    # Validate inputs
    _validate_inputs(data, normalization)

    # Normalize data if required
    normalized_data = _normalize_data(data, normalization)

    # Prepare parameters dictionary
    params_used = {
        'bandwidth': bandwidth,
        'kernel': kernel.__name__ if hasattr(kernel, '__name__') else 'custom',
        'normalization': normalization,
        'metric': metric if isinstance(metric, str) else 'custom',
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'max_iter': max_iter
    }

    # Compute density estimation based on solver choice
    if solver == 'closed_form':
        result = _closed_form_density_estimation(normalized_data, bandwidth, kernel)
    else:
        raise ValueError(f"Solver '{solver}' not implemented.")

    # Compute metrics
    metrics = _compute_metrics(result, normalized_data, metric, custom_metric_params)

    # Prepare output dictionary
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': []
    }

    return output

def _validate_inputs(data: np.ndarray, normalization: str) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values.")
    if normalization not in ['none', 'standard', 'minmax', 'robust']:
        raise ValueError("Invalid normalization method.")

def _normalize_data(data: np.ndarray, normalization: str) -> np.ndarray:
    """Normalize data based on the specified method."""
    if normalization == 'none':
        return data
    elif normalization == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif normalization == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif normalization == 'robust':
        median = np.median(data, axis=0)
        iqr = np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0)
        return (data - median) / iqr
    else:
        raise ValueError("Invalid normalization method.")

def _closed_form_density_estimation(data: np.ndarray, bandwidth: float, kernel: Callable) -> np.ndarray:
    """Compute density estimation using closed-form solution."""
    n_samples = data.shape[0]
    density = np.zeros(n_samples)
    for i in range(n_samples):
        distances = np.linalg.norm(data - data[i], axis=1)
        weights = kernel(distances / bandwidth)
        density[i] = np.sum(weights) / (n_samples * bandwidth**data.shape[1])
    return density

def _compute_metrics(
    result: np.ndarray,
    data: np.ndarray,
    metric: Union[str, Callable],
    custom_metric_params: Optional[Dict]
) -> Dict:
    """Compute metrics for the density estimation."""
    metrics = {}
    if isinstance(metric, str):
        if metric == 'euclidean':
            distances = np.linalg.norm(data - data.mean(axis=0), axis=1)
        elif metric == 'manhattan':
            distances = np.sum(np.abs(data - data.mean(axis=0)), axis=1)
        elif metric == 'cosine':
            normalized_data = data / np.linalg.norm(data, axis=1, keepdims=True)
            distances = 1 - np.sum(normalized_data * normalized_data.mean(axis=0), axis=1)
        else:
            raise ValueError(f"Metric '{metric}' not implemented.")
    else:
        distances = metric(data, data.mean(axis=0), **(custom_metric_params or {}))

    metrics['mean_distance'] = np.mean(distances)
    metrics['std_distance'] = np.std(distances)
    return metrics

################################################################################
# noyau_gaussien
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X must not contain NaN or infinite values")

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

def _compute_distance(X: np.ndarray, Y: Optional[np.ndarray] = None,
                     metric: str = 'euclidean',
                     custom_metric: Optional[Callable] = None) -> np.ndarray:
    """Compute distance matrix between points."""
    if Y is None:
        Y = X

    if custom_metric is not None:
        return np.array([[custom_metric(x, y) for y in Y] for x in X])

    if metric == 'euclidean':
        return np.sqrt(np.sum((X[:, np.newaxis] - Y) ** 2, axis=2))
    elif metric == 'manhattan':
        return np.sum(np.abs(X[:, np.newaxis] - Y), axis=2)
    elif metric == 'cosine':
        dot_products = np.dot(X, Y.T)
        norms = np.sqrt(np.sum(X**2, axis=1))[:, np.newaxis] * np.sqrt(np.sum(Y**2, axis=1))
        return 1 - dot_products / (norms + 1e-8)
    elif metric == 'minkowski':
        p = 3
        return np.sum(np.abs(X[:, np.newaxis] - Y) ** p, axis=2) ** (1/p)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _gaussian_kernel(distance_matrix: np.ndarray, bandwidth: float) -> np.ndarray:
    """Compute Gaussian kernel values."""
    return np.exp(-0.5 * (distance_matrix / bandwidth) ** 2)

def _compute_density(X: np.ndarray, bandwidth: float,
                    metric: str = 'euclidean',
                    custom_metric: Optional[Callable] = None) -> np.ndarray:
    """Compute density estimates for each point."""
    distance_matrix = _compute_distance(X, metric=metric, custom_metric=custom_metric)
    kernel_values = _gaussian_kernel(distance_matrix, bandwidth)
    return np.sum(kernel_values, axis=1)

def noyau_gaussien_fit(X: np.ndarray,
                      bandwidth: float = 1.0,
                      metric: str = 'euclidean',
                      custom_metric: Optional[Callable] = None,
                      normalization: str = 'standard') -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Fit Gaussian kernel density estimation.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    bandwidth : float, optional
        Bandwidth parameter for the Gaussian kernel (default=1.0)
    metric : str, optional
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski') (default='euclidean')
    custom_metric : Callable, optional
        Custom distance function if needed
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default='standard')

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': density estimates for each point
        - 'metrics': dictionary of computed metrics
        - 'params_used': parameters used in the computation
        - 'warnings': any warnings generated during computation

    Example
    -------
    >>> X = np.random.rand(10, 2)
    >>> result = noyau_gaussien_fit(X, bandwidth=0.5)
    """
    # Validate inputs
    _validate_inputs(X)

    # Normalize data
    X_normalized = _normalize_data(X, method=normalization)

    # Compute density estimates
    densities = _compute_density(X_normalized, bandwidth, metric, custom_metric)

    # Prepare output
    output = {
        'result': densities,
        'metrics': {},
        'params_used': {
            'bandwidth': bandwidth,
            'metric': metric,
            'normalization': normalization
        },
        'warnings': []
    }

    return output

################################################################################
# rayon_epsilon
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
    """Normalize data based on specified method."""
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

def compute_distance_matrix(X: np.ndarray, distance_metric: Union[str, Callable]) -> np.ndarray:
    """Compute distance matrix based on specified metric."""
    if isinstance(distance_metric, str):
        if distance_metric == 'euclidean':
            return np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
        elif distance_metric == 'manhattan':
            return np.sum(np.abs(X[:, np.newaxis] - X), axis=2)
        elif distance_metric == 'cosine':
            dot_products = np.dot(X, X.T)
            norms = np.sqrt(np.sum(X ** 2, axis=1))
            return 1 - (dot_products / np.outer(norms, norms))
        elif distance_metric == 'minkowski':
            p = 3
            return np.sum(np.abs(X[:, np.newaxis] - X) ** p, axis=2) ** (1/p)
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
    else:
        return np.array([[distance_metric(x, y) for y in X] for x in X])

def find_optimal_epsilon(distances: np.ndarray, min_points: int = 5) -> float:
    """Find optimal epsilon value using k-distance graph."""
    n_samples = distances.shape[0]
    k_distances = np.sort(distances, axis=1)[:, min_points - 1]
    epsilon = np.median(k_distances)
    return epsilon

def rayon_epsilon_fit(
    X: np.ndarray,
    min_points: int = 5,
    distance_metric: Union[str, Callable] = 'euclidean',
    normalization: str = 'standard'
) -> Dict[str, Union[float, Dict, np.ndarray]]:
    """
    Compute the optimal epsilon radius for DBSCAN clustering.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    min_points : int, optional
        Minimum number of points to form a dense region (default: 5).
    distance_metric : str or callable, optional
        Distance metric to use (default: 'euclidean').
    normalization : str, optional
        Normalization method to apply (default: 'standard').

    Returns:
    --------
    result : dict
        Dictionary containing the computed epsilon value and additional information.
    """
    validate_input(X)
    X_normalized = normalize_data(X, method=normalization)
    distance_matrix = compute_distance_matrix(X_normalized, distance_metric)
    epsilon = find_optimal_epsilon(distance_matrix, min_points)

    return {
        "result": epsilon,
        "metrics": {},
        "params_used": {
            "min_points": min_points,
            "distance_metric": distance_metric,
            "normalization": normalization
        },
        "warnings": []
    }

# Example usage:
# X = np.random.rand(100, 2)
# result = rayon_epsilon_fit(X, min_points=5, distance_metric='euclidean', normalization='standard')

################################################################################
# min_pts
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_input(data: np.ndarray) -> None:
    """Validate input data for min_pts clustering."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or infinite values.")

def normalize_data(
    data: np.ndarray,
    method: str = "standard",
    custom_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize data using specified method or custom function."""
    if custom_func is not None:
        return custom_func(data)

    if method == "none":
        return data
    elif method == "standard":
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def compute_distance(
    data: np.ndarray,
    metric: str = "euclidean",
    custom_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> np.ndarray:
    """Compute pairwise distances between data points."""
    if custom_func is not None:
        return np.array([[custom_func(x, y) for y in data] for x in data])

    n = data.shape[0]
    distances = np.zeros((n, n))

    if metric == "euclidean":
        for i in range(n):
            distances[i] = np.linalg.norm(data - data[i], axis=1)
    elif metric == "manhattan":
        for i in range(n):
            distances[i] = np.sum(np.abs(data - data[i]), axis=1)
    elif metric == "cosine":
        for i in range(n):
            dot_products = np.dot(data, data[i])
            norms = np.linalg.norm(data, axis=1) * np.linalg.norm(data[i])
            distances[i] = 1 - (dot_products / (norms + 1e-8))
    elif metric == "minkowski":
        p = 3
        for i in range(n):
            distances[i] = np.sum(np.abs(data - data[i])**p, axis=1)**(1/p)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

    return distances

def find_core_points(
    distances: np.ndarray,
    eps: float,
    min_pts: int
) -> np.ndarray:
    """Identify core points based on distance and min_pts parameters."""
    core_points = np.zeros(distances.shape[0], dtype=bool)
    for i in range(distances.shape[0]):
        neighbors = np.sum(distances[i] <= eps)
        if neighbors >= min_pts:
            core_points[i] = True
    return core_points

def expand_cluster(
    data: np.ndarray,
    distances: np.ndarray,
    eps: float,
    min_pts: int,
    cluster_id: int,
    core_points: np.ndarray,
    labels: np.ndarray
) -> None:
    """Expand a cluster by adding density-reachable points."""
    queue = np.where(core_points)[0]

    while len(queue) > 0:
        current_point = queue[0]
        queue = queue[1:]

        neighbors = np.where(distances[current_point] <= eps)[0]

        for neighbor in neighbors:
            if labels[neighbor] == -1:  # Not yet assigned to a cluster
                labels[neighbor] = cluster_id

            if core_points[neighbor]:
                queue = np.append(queue, neighbor)

def min_pts_fit(
    data: np.ndarray,
    eps: float,
    min_pts: int,
    normalize_method: str = "standard",
    distance_metric: str = "euclidean",
    custom_normalize: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Perform density-based clustering using the min_pts algorithm.

    Parameters:
    - data: Input data as a 2D numpy array.
    - eps: Maximum distance between two points for one to be considered in the neighborhood of the other.
    - min_pts: Minimum number of points required to form a dense region.
    - normalize_method: Normalization method for the data ("none", "standard", "minmax", "robust").
    - distance_metric: Distance metric to use ("euclidean", "manhattan", "cosine", "minkowski").
    - custom_normalize: Custom normalization function.
    - custom_distance: Custom distance function.

    Returns:
    - A dictionary containing the clustering results, metrics, and parameters used.
    """
    # Validate input
    validate_input(data)

    # Normalize data
    normalized_data = normalize_data(
        data,
        method=normalize_method,
        custom_func=custom_normalize
    )

    # Compute distances
    distances = compute_distance(
        normalized_data,
        metric=distance_metric,
        custom_func=custom_distance
    )

    # Initialize labels
    n = data.shape[0]
    labels = -1 * np.ones(n, dtype=int)
    cluster_id = 0

    # Find core points
    core_points = find_core_points(distances, eps, min_pts)

    # Perform clustering
    for i in range(n):
        if core_points[i] and labels[i] == -1:
            labels[i] = cluster_id
            expand_cluster(
                normalized_data,
                distances,
                eps,
                min_pts,
                cluster_id,
                core_points,
                labels
            )
            cluster_id += 1

    # Calculate metrics
    n_clusters = len(np.unique(labels[labels != -1]))
    noise_points = np.sum(labels == -1)

    metrics = {
        "n_clusters": n_clusters,
        "noise_points": noise_points,
        "silhouette_score": None  # Placeholder for future implementation
    }

    params_used = {
        "eps": eps,
        "min_pts": min_pts,
        "normalize_method": normalize_method if custom_normalize is None else "custom",
        "distance_metric": distance_metric if custom_distance is None else "custom"
    }

    warnings = {
        "data_normalized": normalize_method != "none",
        "custom_functions_used": custom_normalize is not None or custom_distance is not None
    }

    return {
        "result": labels,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

# Example usage:
# result = min_pts_fit(data=np.random.rand(100, 5), eps=0.5, min_pts=5)

################################################################################
# cluster_density_based
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def cluster_density_based_fit(
    data: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: Union[str, Callable] = 'euclidean',
    normalization: Optional[str] = None,
    distance_threshold: float = 0.5,
    custom_distance_func: Optional[Callable] = None
) -> Dict:
    """
    Perform density-based clustering on the input data.

    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (n_samples, n_features)
    eps : float, optional
        The maximum distance between two samples for one to be considered in the neighborhood of the other.
    min_samples : int, optional
        The number of samples in a neighborhood for a point to be considered as a core point.
    metric : str or callable, optional
        The distance metric to use. Can be 'euclidean', 'manhattan', 'cosine', or a custom callable.
    normalization : str, optional
        Normalization method. Can be 'standard', 'minmax', or None.
    distance_threshold : float, optional
        Threshold for density-based clustering.
    custom_distance_func : callable, optional
        Custom distance function if metric is not predefined.

    Returns:
    --------
    dict
        A dictionary containing the clustering results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, eps, min_samples)

    # Normalize data if specified
    normalized_data = _normalize_data(data, normalization)

    # Compute distance matrix
    distance_matrix = _compute_distance_matrix(
        normalized_data,
        metric=metric,
        custom_distance_func=custom_distance_func
    )

    # Perform density-based clustering
    labels = _density_based_clustering(distance_matrix, eps, min_samples, distance_threshold)

    # Compute metrics
    metrics = _compute_metrics(data, labels)

    # Prepare output
    result = {
        'result': {
            'labels': labels,
            'n_clusters': len(np.unique(labels))
        },
        'metrics': metrics,
        'params_used': {
            'eps': eps,
            'min_samples': min_samples,
            'metric': metric,
            'normalization': normalization
        },
        'warnings': []
    }

    return result

def _validate_inputs(data: np.ndarray, eps: float, min_samples: int) -> None:
    """Validate the input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive.")

def _normalize_data(data: np.ndarray, method: Optional[str]) -> np.ndarray:
    """Normalize the data based on the specified method."""
    if method is None:
        return data
    elif method == 'standard':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_distance_matrix(
    data: np.ndarray,
    metric: Union[str, Callable],
    custom_distance_func: Optional[Callable]
) -> np.ndarray:
    """Compute the distance matrix using the specified metric."""
    n_samples = data.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    if custom_distance_func is not None:
        for i in range(n_samples):
            for j in range(n_samples):
                distance_matrix[i, j] = custom_distance_func(data[i], data[j])
    elif metric == 'euclidean':
        for i in range(n_samples):
            for j in range(n_samples):
                distance_matrix[i, j] = np.linalg.norm(data[i] - data[j])
    elif metric == 'manhattan':
        for i in range(n_samples):
            for j in range(n_samples):
                distance_matrix[i, j] = np.sum(np.abs(data[i] - data[j]))
    elif metric == 'cosine':
        for i in range(n_samples):
            for j in range(n_samples):
                dot_product = np.dot(data[i], data[j])
                norm_i = np.linalg.norm(data[i])
                norm_j = np.linalg.norm(data[j])
                distance_matrix[i, j] = 1 - (dot_product / (norm_i * norm_j + 1e-8))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return distance_matrix

def _density_based_clustering(
    distance_matrix: np.ndarray,
    eps: float,
    min_samples: int,
    distance_threshold: float
) -> np.ndarray:
    """Perform density-based clustering using the distance matrix."""
    n_samples = distance_matrix.shape[0]
    labels = np.zeros(n_samples, dtype=int)
    cluster_id = 0

    for i in range(n_samples):
        if labels[i] != 0:
            continue

        neighbors = np.where(distance_matrix[i] <= eps)[0]
        if len(neighbors) < min_samples:
            labels[i] = -1  # Noise point
        else:
            cluster_id += 1
            labels[i] = cluster_id
            queue = neighbors.tolist()

            while queue:
                j = queue.pop(0)
                if labels[j] == -1:  # Change noise to border point
                    labels[j] = cluster_id
                if labels[j] != 0:
                    continue

                labels[j] = cluster_id
                new_neighbors = np.where(distance_matrix[j] <= eps)[0]
                if len(new_neighbors) >= min_samples:
                    queue.extend(new_neighbors)

    return labels

def _compute_metrics(data: np.ndarray, labels: np.ndarray) -> Dict:
    """Compute clustering metrics."""
    n_clusters = len(np.unique(labels))
    if n_clusters <= 1:
        return {
            'n_clusters': n_clusters,
            'silhouette_score': np.nan,
            'davies_bouldin_score': np.nan
        }

    from sklearn.metrics import silhouette_score, davies_bouldin_score

    try:
        silhouette = silhouette_score(data, labels)
    except:
        silhouette = np.nan

    try:
        davies_bouldin = davies_bouldin_score(data, labels)
    except:
        davies_bouldin = np.nan

    return {
        'n_clusters': n_clusters,
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin
    }

################################################################################
# noise_points
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def noise_points_fit(
    data: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: Union[str, Callable] = "euclidean",
    normalize: str = "none",
    distance_threshold: float = 0.5,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Identify noise points in a dataset using density-based clustering.

    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features).
    eps : float, optional
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples : int, optional
        The number of samples in a neighborhood for a point to be considered as a core point.
    metric : str or callable, optional
        The distance metric to use. Can be 'euclidean', 'manhattan', 'cosine', or a custom callable.
    normalize : str, optional
        Normalization method. Can be 'none', 'standard', 'minmax', or 'robust'.
    distance_threshold : float, optional
        Threshold for considering a point as noise.
    custom_metric : callable, optional
        Custom distance metric function.

    Returns
    -------
    Dict
        A dictionary containing:
        - "result": Array of noise points.
        - "metrics": Dictionary of computed metrics.
        - "params_used": Dictionary of parameters used.
        - "warnings": List of warnings.

    Examples
    --------
    >>> data = np.random.rand(100, 2)
    >>> result = noise_points_fit(data, eps=0.5, min_samples=5)
    """
    # Validate inputs
    _validate_inputs(data, eps, min_samples)

    # Normalize data if required
    normalized_data = _normalize_data(data, normalize)

    # Compute distances
    distances = _compute_distances(normalized_data, metric, custom_metric)

    # Identify noise points
    noise_points = _identify_noise_points(distances, eps, min_samples, distance_threshold)

    # Compute metrics
    metrics = _compute_metrics(data, noise_points)

    # Prepare output
    result_dict = {
        "result": noise_points,
        "metrics": metrics,
        "params_used": {
            "eps": eps,
            "min_samples": min_samples,
            "metric": metric,
            "normalize": normalize,
            "distance_threshold": distance_threshold
        },
        "warnings": []
    }

    return result_dict

def _validate_inputs(data: np.ndarray, eps: float, min_samples: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive.")

def _normalize_data(data: np.ndarray, method: str) -> np.ndarray:
    """Normalize data based on the specified method."""
    if method == "none":
        return data
    elif method == "standard":
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_distances(data: np.ndarray, metric: Union[str, Callable], custom_metric: Optional[Callable]) -> np.ndarray:
    """Compute pairwise distances between data points."""
    n_samples = data.shape[0]
    distances = np.zeros((n_samples, n_samples))

    if custom_metric is not None:
        for i in range(n_samples):
            for j in range(n_samples):
                distances[i, j] = custom_metric(data[i], data[j])
    elif metric == "euclidean":
        for i in range(n_samples):
            distances[i] = np.linalg.norm(data - data[i], axis=1)
    elif metric == "manhattan":
        for i in range(n_samples):
            distances[i] = np.sum(np.abs(data - data[i]), axis=1)
    elif metric == "cosine":
        for i in range(n_samples):
            distances[i] = 1 - np.dot(data, data[i]) / (np.linalg.norm(data, axis=1) * np.linalg.norm(data[i]))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return distances

def _identify_noise_points(distances: np.ndarray, eps: float, min_samples: int, distance_threshold: float) -> np.ndarray:
    """Identify noise points based on density criteria."""
    n_samples = distances.shape[0]
    core_points = np.zeros(n_samples, dtype=bool)

    for i in range(n_samples):
        neighbors = distances[i] <= eps
        if np.sum(neighbors) >= min_samples:
            core_points[i] = True

    noise_points = np.where(np.sum(distances <= distance_threshold, axis=1) < min_samples)[0]
    return noise_points

def _compute_metrics(data: np.ndarray, noise_points: np.ndarray) -> Dict:
    """Compute metrics for the noise points."""
    return {
        "noise_ratio": len(noise_points) / data.shape[0],
        "noise_count": len(noise_points)
    }

################################################################################
# border_points
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def border_points_fit(
    data: np.ndarray,
    density_estimator: Callable[[np.ndarray], np.ndarray],
    distance_metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    threshold: Optional[float] = None,
    min_samples: int = 5,
    normalization: str = 'none',
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Identify border points in a dataset based on density estimation.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_samples, n_features)
    density_estimator : Callable
        Function that estimates density for each point in the dataset.
    distance_metric : str or Callable
        Distance metric to use for neighbor search. Can be 'euclidean', 'manhattan',
        'cosine', or a custom callable.
    threshold : float, optional
        Density threshold below which points are considered border points. If None,
        min_samples is used instead.
    min_samples : int
        Minimum number of samples in a neighborhood for a point to be considered not a border.
    normalization : str
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    custom_normalization : Callable, optional
        Custom normalization function if needed.
    **kwargs :
        Additional keyword arguments passed to the density estimator.

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': array of border points indices
        - 'metrics': dictionary of computed metrics
        - 'params_used': dictionary of parameters used
        - 'warnings': list of warnings encountered

    Examples
    --------
    >>> data = np.random.rand(100, 2)
    >>> def density_estimator(x):
    ...     return np.exp(-np.sum(x**2, axis=1))
    >>> result = border_points_fit(data, density_estimator)
    """
    # Validate inputs
    _validate_inputs(data, normalization)

    # Normalize data if needed
    normalized_data = _apply_normalization(
        data, normalization, custom_normalization)

    # Compute density for each point
    densities = density_estimator(normalized_data, **kwargs)

    # Identify border points
    border_indices = _identify_border_points(
        densities, threshold, min_samples,
        distance_metric, normalized_data)

    # Compute metrics
    metrics = _compute_metrics(border_indices, densities)

    return {
        'result': border_indices,
        'metrics': metrics,
        'params_used': {
            'distance_metric': distance_metric,
            'threshold': threshold,
            'min_samples': min_samples,
            'normalization': normalization
        },
        'warnings': []
    }

def _validate_inputs(
    data: np.ndarray,
    normalization: str
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Input data must be 2-dimensional")
    if normalization not in ['none', 'standard', 'minmax', 'robust']:
        raise ValueError("Invalid normalization method")

def _apply_normalization(
    data: np.ndarray,
    normalization: str,
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Apply specified normalization to the data."""
    if custom_normalization is not None:
        return custom_normalization(data)

    if normalization == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif normalization == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif normalization == 'robust':
        return (data - np.median(data, axis=0)) / (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0))
    return data.copy()

def _identify_border_points(
    densities: np.ndarray,
    threshold: Optional[float],
    min_samples: int,
    distance_metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    data: np.ndarray
) -> np.ndarray:
    """Identify border points based on density and neighborhood."""
    if threshold is not None:
        return np.where(densities < threshold)[0]

    # For each point, find neighbors within some distance
    if isinstance(distance_metric, str):
        distances = _compute_distance_matrix(data, distance_metric)
    else:
        distances = np.array([[distance_metric(x, y) for y in data] for x in data])

    # Count neighbors above some density threshold
    neighbor_counts = np.sum(distances < 0.5, axis=1)  # Using 0.5 as example threshold

    return np.where(neighbor_counts < min_samples)[0]

def _compute_distance_matrix(
    data: np.ndarray,
    metric: str
) -> np.ndarray:
    """Compute pairwise distance matrix."""
    if metric == 'euclidean':
        return np.sqrt(((data[:, np.newaxis] - data) ** 2).sum(axis=2))
    elif metric == 'manhattan':
        return np.abs(data[:, np.newaxis] - data).sum(axis=2)
    elif metric == 'cosine':
        return 1 - np.dot(data, data.T) / (np.linalg.norm(data, axis=1)[:, np.newaxis] * np.linalg.norm(data, axis=1))
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _compute_metrics(
    border_indices: np.ndarray,
    densities: np.ndarray
) -> Dict[str, float]:
    """Compute metrics about the border points."""
    return {
        'border_points_ratio': len(border_indices) / len(densities),
        'average_border_density': np.mean(densities[border_indices])
    }

################################################################################
# core_points
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input contains NaN or inf values")

def normalize_data(
    X: np.ndarray,
    method: str = "standard",
    custom_func: Optional[Callable] = None
) -> np.ndarray:
    """Normalize data using specified method or custom function."""
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

def compute_distance(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    metric: str = "euclidean",
    custom_func: Optional[Callable] = None
) -> np.ndarray:
    """Compute distance matrix between points."""
    if Y is None:
        Y = X

    if custom_func is not None:
        return custom_func(X, Y)

    n = X.shape[0]
    m = Y.shape[0]
    dist_matrix = np.zeros((n, m))

    if metric == "euclidean":
        for i in range(n):
            dist_matrix[i] = np.sqrt(np.sum((X[i] - Y) ** 2, axis=1))
    elif metric == "manhattan":
        for i in range(n):
            dist_matrix[i] = np.sum(np.abs(X[i] - Y), axis=1)
    elif metric == "cosine":
        for i in range(n):
            dot_product = np.dot(X[i], Y.T)
            norm_x = np.linalg.norm(X[i])
            norm_y = np.linalg.norm(Y, axis=1)
            dist_matrix[i] = 1 - (dot_product / (norm_x * norm_y + 1e-8))
    elif metric == "minkowski":
        p = 3
        for i in range(n):
            dist_matrix[i] = np.sum(np.abs(X[i] - Y) ** p, axis=1) ** (1/p)

    return dist_matrix

def find_core_points(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    distance_matrix: Optional[np.ndarray] = None
) -> np.ndarray:
    """Identify core points based on density criteria."""
    if distance_matrix is None:
        distance_matrix = compute_distance(X)

    n = X.shape[0]
    core_points_mask = np.zeros(n, dtype=bool)

    for i in range(n):
        neighbors = distance_matrix[i] <= eps
        if np.sum(neighbors) >= min_samples:
            core_points_mask[i] = True

    return core_points_mask

def core_points_fit(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    normalize_method: str = "standard",
    distance_metric: str = "euclidean",
    custom_normalize: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Identify core points in a dataset using density-based clustering.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    eps : float
        Maximum distance between two samples for one to be considered in the neighborhood of the other
    min_samples : int
        Number of samples in a neighborhood for a point to be considered as a core point
    normalize_method : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    distance_metric : str
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski')
    custom_normalize : Callable, optional
        Custom normalization function
    custom_distance : Callable, optional
        Custom distance function

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate input
    validate_input(X)

    # Normalize data
    X_norm = normalize_data(
        X,
        method=normalize_method,
        custom_func=custom_normalize
    )

    # Compute distance matrix
    dist_matrix = compute_distance(
        X_norm,
        metric=distance_metric,
        custom_func=custom_distance
    )

    # Find core points
    core_points_mask = find_core_points(
        X_norm,
        eps=eps,
        min_samples=min_samples,
        distance_matrix=dist_matrix
    )

    # Prepare output
    result = {
        "core_points_indices": np.where(core_points_mask)[0],
        "core_points_mask": core_points_mask,
        "distance_matrix": dist_matrix
    }

    metrics = {
        "n_core_points": np.sum(core_points_mask),
        "mean_distance": np.mean(dist_matrix[dist_matrix != 0])
    }

    params_used = {
        "eps": eps,
        "min_samples": min_samples,
        "normalize_method": normalize_method if custom_normalize is None else "custom",
        "distance_metric": distance_metric if custom_distance is None else "custom"
    }

    warnings = []

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

# Example usage:
"""
X = np.random.rand(100, 5)
result = core_points_fit(X, eps=0.3, min_samples=10)
print(result['result']['core_points_indices'])
"""

################################################################################
# reachability_distance
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def validate_inputs(X: np.ndarray, Y: np.ndarray) -> None:
    """Validate input arrays for reachability distance calculation."""
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("Input arrays must be 2-dimensional")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Input arrays must have the same number of features")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input array X contains NaN or infinite values")
    if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
        raise ValueError("Input array Y contains NaN or infinite values")

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

def compute_distance(X: np.ndarray, Y: np.ndarray,
                    metric: str = 'euclidean',
                    custom_metric: Optional[Callable] = None) -> np.ndarray:
    """Compute distance between points using specified metric."""
    if custom_metric is not None:
        return custom_metric(X, Y)

    if metric == 'euclidean':
        return np.sqrt(np.sum((X[:, np.newaxis] - Y) ** 2, axis=2))
    elif metric == 'manhattan':
        return np.sum(np.abs(X[:, np.newaxis] - Y), axis=2)
    elif metric == 'cosine':
        dot_products = np.dot(X, Y.T)
        norms = np.sqrt(np.sum(X**2, axis=1))[:, np.newaxis] * np.sqrt(np.sum(Y**2, axis=1))
        return 1 - dot_products / (norms + 1e-8)
    elif metric == 'minkowski':
        p = 3
        return np.sum(np.abs(X[:, np.newaxis] - Y) ** p, axis=2) ** (1/p)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def reachability_distance_fit(X: np.ndarray,
                            Y: Optional[np.ndarray] = None,
                            normalize_method: str = 'standard',
                            distance_metric: str = 'euclidean',
                            custom_distance: Optional[Callable] = None,
                            **kwargs) -> Dict[str, Any]:
    """
    Compute reachability distance between points in X and Y.

    Parameters:
    -----------
    X : np.ndarray
        Input array of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Target array of shape (m_samples, n_features). If None, uses X.
    normalize_method : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    distance_metric : str
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski')
    custom_distance : Optional[Callable]
        Custom distance function if needed

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> result = reachability_distance_fit(X)
    """
    # Validate inputs
    if Y is None:
        Y = X

    validate_inputs(X, Y)

    # Normalize data
    X_norm = normalize_data(X, method=normalize_method)
    Y_norm = normalize_data(Y, method=normalize_method)

    # Compute distances
    distances = compute_distance(X_norm, Y_norm,
                               metric=distance_metric,
                               custom_metric=custom_distance)

    # Prepare output
    result = {
        'result': distances,
        'metrics': {
            'normalization_method': normalize_method,
            'distance_metric': distance_metric if custom_distance is None else 'custom'
        },
        'params_used': {
            'normalize_method': normalize_method,
            'distance_metric': distance_metric if custom_distance is None else 'custom',
            'X_shape': X.shape,
            'Y_shape': Y.shape
        },
        'warnings': []
    }

    return result

def _internal_reachability_distance(X: np.ndarray,
                                  Y: np.ndarray,
                                  normalize_method: str = 'standard',
                                  distance_metric: str = 'euclidean') -> np.ndarray:
    """Internal function for reachability distance calculation."""
    X_norm = normalize_data(X, method=normalize_method)
    Y_norm = normalize_data(Y, method=normalize_method)
    return compute_distance(X_norm, Y_norm, metric=distance_metric)

def _validate_normalization_method(method: str) -> None:
    """Validate normalization method."""
    valid_methods = ['none', 'standard', 'minmax', 'robust']
    if method not in valid_methods:
        raise ValueError(f"Normalization method must be one of {valid_methods}")

def _validate_distance_metric(metric: str) -> None:
    """Validate distance metric."""
    valid_metrics = ['euclidean', 'manhattan', 'cosine', 'minkowski']
    if metric not in valid_metrics:
        raise ValueError(f"Distance metric must be one of {valid_metrics}")

################################################################################
# core_distance
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input contains NaN or infinite values")

def normalize_data(
    X: np.ndarray,
    method: str = "standard",
    custom_norm: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize data based on specified method."""
    if custom_norm is not None:
        return custom_norm(X)

    X_normalized = X.copy()
    if method == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_normalized = (X - mean) / std
    elif method == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_normalized = (X - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_normalized = (X - median) / (iqr + 1e-8)
    elif method == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_normalized

def compute_distance_matrix(
    X: np.ndarray,
    metric: str = "euclidean",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> np.ndarray:
    """Compute distance matrix based on specified metric."""
    n_samples = X.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))

    if custom_metric is not None:
        for i in range(n_samples):
            for j in range(n_samples):
                dist_matrix[i, j] = custom_metric(X[i], X[j])
        return dist_matrix

    if metric == "euclidean":
        for i in range(n_samples):
            for j in range(i, n_samples):
                dist = np.linalg.norm(X[i] - X[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
    elif metric == "manhattan":
        for i in range(n_samples):
            for j in range(i, n_samples):
                dist = np.sum(np.abs(X[i] - X[j]))
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
    elif metric == "cosine":
        for i in range(n_samples):
            for j in range(i, n_samples):
                dot_product = np.dot(X[i], X[j])
                norm_i = np.linalg.norm(X[i])
                norm_j = np.linalg.norm(X[j])
                dist = 1 - (dot_product / (norm_i * norm_j + 1e-8))
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
    elif metric == "minkowski":
        p = 3
        for i in range(n_samples):
            for j in range(i, n_samples):
                dist = np.sum(np.abs(X[i] - X[j]) ** p) ** (1/p)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

    return dist_matrix

def compute_core_distance(
    X: np.ndarray,
    eps: float = 0.5,
    min_pts: int = 5
) -> np.ndarray:
    """Compute core distance for each point."""
    dist_matrix = compute_distance_matrix(X)
    n_samples = X.shape[0]
    core_distances = np.zeros(n_samples)

    for i in range(n_samples):
        neighbors = dist_matrix[i] <= eps
        if np.sum(neighbors) >= min_pts:
            core_distances[i] = np.partition(dist_matrix[i], min_pts - 1)[min_pts - 1]
        else:
            core_distances[i] = np.inf

    return core_distances

def core_distance_fit(
    X: np.ndarray,
    eps: float = 0.5,
    min_pts: int = 5,
    normalization_method: str = "standard",
    distance_metric: str = "euclidean",
    custom_norm: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, Union[str, float]]]]:
    """
    Compute core distances for density-based clustering.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    eps : float, optional
        Maximum distance between two points to be considered in the same neighborhood
    min_pts : int, optional
        Minimum number of points required to form a dense region
    normalization_method : str, optional
        Normalization method for input data ("none", "standard", "minmax", "robust")
    distance_metric : str, optional
        Distance metric to use ("euclidean", "manhattan", "cosine", "minkowski")
    custom_norm : callable, optional
        Custom normalization function
    custom_metric : callable, optional
        Custom distance metric function

    Returns
    -------
    dict
        Dictionary containing:
        - "result": array of core distances
        - "metrics": dictionary of computed metrics
        - "params_used": dictionary of parameters used
        - "warnings": list of warnings

    Example
    -------
    >>> X = np.random.rand(10, 2)
    >>> result = core_distance_fit(X, eps=0.5, min_pts=3)
    """
    # Validate input
    validate_input(X)

    # Normalize data
    X_normalized = normalize_data(
        X,
        method=normalization_method,
        custom_norm=custom_norm
    )

    # Compute core distances
    core_distances = compute_core_distance(
        X_normalized,
        eps=eps,
        min_pts=min_pts
    )

    # Prepare output
    result = {
        "result": core_distances,
        "metrics": {},
        "params_used": {
            "eps": eps,
            "min_pts": min_pts,
            "normalization_method": normalization_method,
            "distance_metric": distance_metric
        },
        "warnings": []
    }

    return result

################################################################################
# density_reachable
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def density_reachable_fit(
    data: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: Union[str, Callable] = 'euclidean',
    normalization: Optional[str] = None,
    distance_params: Optional[Dict] = None
) -> Dict:
    """
    Compute density reachability for clustering based on density.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    eps : float, optional
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples : int, optional
        The number of samples in a neighborhood for a point to be considered as a core point.
    metric : str or callable, optional
        The distance metric to use. Can be 'euclidean', 'manhattan', 'cosine', or a custom callable.
    normalization : str, optional
        Normalization method. Can be 'none', 'standard', 'minmax', or 'robust'.
    distance_params : dict, optional
        Additional parameters for the distance metric.

    Returns:
    --------
    Dict containing:
        - 'result': Density reachability matrix.
        - 'metrics': Dictionary of computed metrics.
        - 'params_used': Parameters used in the computation.
        - 'warnings': List of warnings encountered during computation.

    Example:
    --------
    >>> data = np.random.rand(10, 2)
    >>> result = density_reachable_fit(data, eps=0.5, min_samples=3)
    """
    # Validate inputs
    validate_inputs(data, eps, min_samples)

    # Normalize data if required
    normalized_data = apply_normalization(data, normalization)

    # Compute distance matrix
    distance_matrix = compute_distance_matrix(normalized_data, metric, distance_params)

    # Compute density reachability
    reachability_matrix = compute_density_reachability(distance_matrix, eps, min_samples)

    # Prepare output
    output = {
        'result': reachability_matrix,
        'metrics': {'eps': eps, 'min_samples': min_samples},
        'params_used': {
            'metric': metric,
            'normalization': normalization,
            'distance_params': distance_params
        },
        'warnings': []
    }

    return output

def validate_inputs(
    data: np.ndarray,
    eps: float,
    min_samples: int
) -> None:
    """
    Validate input data and parameters.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix.
    eps : float
        The maximum distance between two samples.
    min_samples : int
        The number of samples in a neighborhood for a point to be considered as a core point.

    Raises:
    -------
    ValueError
        If inputs are invalid.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array.")
    if eps <= 0:
        raise ValueError("eps must be a positive number.")
    if min_samples <= 0:
        raise ValueError("min_samples must be a positive integer.")

def apply_normalization(
    data: np.ndarray,
    normalization: Optional[str]
) -> np.ndarray:
    """
    Apply normalization to the data.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix.
    normalization : str, optional
        Normalization method.

    Returns:
    --------
    np.ndarray
        Normalized data matrix.
    """
    if normalization is None:
        return data

    normalized_data = data.copy()

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

    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    return normalized_data

def compute_distance_matrix(
    data: np.ndarray,
    metric: Union[str, Callable],
    params: Optional[Dict]
) -> np.ndarray:
    """
    Compute the distance matrix for the given data.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix.
    metric : str or callable
        The distance metric to use.
    params : dict, optional
        Additional parameters for the distance metric.

    Returns:
    --------
    np.ndarray
        Distance matrix.
    """
    n_samples = data.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    if callable(metric):
        for i in range(n_samples):
            for j in range(n_samples):
                distance_matrix[i, j] = metric(data[i], data[j], **params)
    else:
        if metric == 'euclidean':
            distance_matrix = np.sqrt(np.sum((data[:, np.newaxis] - data) ** 2, axis=2))
        elif metric == 'manhattan':
            distance_matrix = np.sum(np.abs(data[:, np.newaxis] - data), axis=2)
        elif metric == 'cosine':
            dot_products = np.dot(data, data.T)
            norms = np.sqrt(np.sum(data ** 2, axis=1))
            distance_matrix = 1 - (dot_products / np.outer(norms, norms))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return distance_matrix

def compute_density_reachability(
    distance_matrix: np.ndarray,
    eps: float,
    min_samples: int
) -> np.ndarray:
    """
    Compute the density reachability matrix.

    Parameters:
    -----------
    distance_matrix : np.ndarray
        Precomputed distance matrix.
    eps : float
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples : int
        The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    --------
    np.ndarray
        Density reachability matrix.
    """
    n_samples = distance_matrix.shape[0]
    reachability_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        # Count the number of neighbors within eps distance
        neighbors = np.sum(distance_matrix[i] <= eps)
        if neighbors >= min_samples:
            # If core point, set reachability to distance
            reachability_matrix[i] = distance_matrix[i]
        else:
            # If not core point, set reachability to infinity
            reachability_matrix[i] = np.inf

    return reachability_matrix

################################################################################
# density_connectivity
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def density_connectivity_fit(
    data: np.ndarray,
    metric: str = 'euclidean',
    distance_threshold: float = 0.5,
    min_samples: int = 5,
    normalize: str = 'none',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict, str]]:
    """
    Compute density connectivity clustering on input data.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_samples, n_features)
    metric : str or callable, optional
        Distance metric to use. Can be 'euclidean', 'manhattan', 'cosine',
        or a custom callable function.
    distance_threshold : float, optional
        Threshold for density connectivity
    min_samples : int, optional
        Minimum number of samples in a neighborhood to consider it dense
    normalize : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'
    custom_metric : callable, optional
        Custom distance metric function

    Returns
    -------
    dict
        Dictionary containing:
        - 'labels': cluster labels for each sample
        - 'metrics': dictionary of computed metrics
        - 'params_used': parameters actually used
        - 'warnings': any warnings generated

    Examples
    --------
    >>> data = np.random.rand(100, 5)
    >>> result = density_connectivity_fit(data, metric='euclidean', distance_threshold=0.3)
    """
    # Validate inputs
    _validate_inputs(data, metric, normalize)

    # Normalize data if requested
    normalized_data = _apply_normalization(data, normalize)

    # Prepare metric function
    distance_func = _get_distance_function(metric, custom_metric)

    # Compute density connectivity
    labels = _compute_density_connectivity(
        normalized_data,
        distance_func,
        distance_threshold,
        min_samples
    )

    # Compute metrics
    metrics = _compute_metrics(normalized_data, labels)

    return {
        'result': {'labels': labels},
        'metrics': metrics,
        'params_used': {
            'metric': metric if custom_metric is None else 'custom',
            'distance_threshold': distance_threshold,
            'min_samples': min_samples,
            'normalize': normalize
        },
        'warnings': []
    }

def _validate_inputs(
    data: np.ndarray,
    metric: str,
    normalize: str
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Input data must be 2-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

    valid_metrics = ['euclidean', 'manhattan', 'cosine']
    if metric not in valid_metrics and not callable(metric):
        raise ValueError(f"Metric must be one of {valid_metrics} or a callable function")

    valid_normalizations = ['none', 'standard', 'minmax', 'robust']
    if normalize not in valid_normalizations:
        raise ValueError(f"Normalization must be one of {valid_normalizations}")

def _apply_normalization(
    data: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply specified normalization to data."""
    if method == 'none':
        return data

    normalized = np.array(data, dtype=np.float64)

    if method == 'standard':
        mean = np.mean(normalized, axis=0)
        std = np.std(normalized, axis=0)
        normalized = (normalized - mean) / std
    elif method == 'minmax':
        min_val = np.min(normalized, axis=0)
        max_val = np.max(normalized, axis=0)
        normalized = (normalized - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(normalized, axis=0)
        iqr = np.subtract(*np.percentile(normalized, [75, 25], axis=0))
        normalized = (normalized - median) / iqr

    return normalized

def _get_distance_function(
    metric: str,
    custom_metric: Optional[Callable]
) -> Callable:
    """Get distance function based on metric parameter."""
    if custom_metric is not None:
        return custom_metric

    def euclidean(a, b):
        return np.linalg.norm(a - b)

    def manhattan(a, b):
        return np.sum(np.abs(a - b))

    def cosine(a, b):
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    metric_functions = {
        'euclidean': euclidean,
        'manhattan': manhattan,
        'cosine': cosine
    }

    return metric_functions[metric]

def _compute_density_connectivity(
    data: np.ndarray,
    distance_func: Callable,
    threshold: float,
    min_samples: int
) -> np.ndarray:
    """Compute density connectivity clustering."""
    n_samples = data.shape[0]
    labels = np.zeros(n_samples, dtype=int)
    cluster_id = 1

    for i in range(n_samples):
        if labels[i] == 0:  # Unassigned point
            # Find all points within threshold distance
            neighbors = []
            for j in range(n_samples):
                if i != j:
                    dist = distance_func(data[i], data[j])
                    if dist <= threshold:
                        neighbors.append(j)

            # Check if neighborhood is dense enough
            if len(neighbors) >= min_samples:
                labels[i] = cluster_id
                # Assign same cluster to all neighbors in this dense region
                for j in neighbors:
                    if labels[j] == 0:
                        labels[j] = cluster_id
                cluster_id += 1

    return labels

def _compute_metrics(
    data: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """Compute clustering metrics."""
    n_clusters = len(np.unique(labels))
    if n_clusters <= 1:
        return {'n_clusters': n_clusters}

    # Silhouette score
    silhouette = _compute_silhouette(data, labels)

    return {
        'n_clusters': n_clusters,
        'silhouette_score': silhouette
    }

def _compute_silhouette(
    data: np.ndarray,
    labels: np.ndarray
) -> float:
    """Compute silhouette score for clustering."""
    n_samples = data.shape[0]
    scores = np.zeros(n_samples)

    for i in range(n_samples):
        # Compute a(i) - mean intra-cluster distance
        cluster_mask = (labels == labels[i])
        if np.sum(cluster_mask) > 1:
            a = np.mean([np.linalg.norm(data[i] - data[j])
                        for j in range(n_samples) if cluster_mask[j]])
        else:
            a = 0

        # Compute b(i) - mean nearest-cluster distance
        other_clusters = np.unique(labels)
        other_clusters = other_clusters[other_clusters != labels[i]]
        b_values = []

        for c in other_clusters:
            cluster_mask = (labels == c)
            if np.sum(cluster_mask) > 0:
                b_values.append(np.mean([np.linalg.norm(data[i] - data[j])
                                        for j in range(n_samples) if cluster_mask[j]]))
        b = min(b_values) if len(b_values) > 0 else 0

        # Compute silhouette score for this sample
        if max(a, b) != 0:
            scores[i] = (b - a) / max(a, b)
        else:
            scores[i] = 0

    return np.mean(scores) if n_samples > 1 else 0
