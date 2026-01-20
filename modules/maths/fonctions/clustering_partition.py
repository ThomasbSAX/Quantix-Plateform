"""
Quantix – Module clustering_partition
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# k_means
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_input(X: np.ndarray, n_clusters: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X must not contain NaN or infinite values")
    if n_clusters <= 0:
        raise ValueError("n_clusters must be a positive integer")
    if n_clusters > X.shape[0]:
        raise ValueError("n_clusters cannot be greater than the number of samples")

def normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """Normalize the input data."""
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
    """Compute the distance between data points and cluster centers."""
    if metric == 'euclidean':
        return np.sqrt(np.sum((X[:, np.newaxis] - centers) ** 2, axis=2))
    elif metric == 'manhattan':
        return np.sum(np.abs(X[:, np.newaxis] - centers), axis=2)
    elif metric == 'cosine':
        dot_products = np.dot(X, centers.T)
        norms = np.linalg.norm(X, axis=1)[:, np.newaxis] * np.linalg.norm(centers, axis=1)
        return 1 - dot_products / (norms + 1e-8)
    elif metric == 'minkowski':
        p = 3
        return np.sum(np.abs(X[:, np.newaxis] - centers) ** p, axis=2) ** (1/p)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def initialize_centers(X: np.ndarray, n_clusters: int, method: str = 'random') -> np.ndarray:
    """Initialize cluster centers."""
    if method == 'random':
        indices = np.random.choice(X.shape[0], n_clusters, replace=False)
        return X[indices]
    elif method == 'kmeans++':
        centers = [X[np.random.randint(X.shape[0])]]
        for _ in range(1, n_clusters):
            dist_sq = np.min(np.sum((X[:, np.newaxis] - centers) ** 2, axis=2), axis=1)
            probs = dist_sq / np.sum(dist_sq)
            cumulative_probs = np.cumsum(probs)
            r = np.random.rand()
            new_center_idx = np.searchsorted(cumulative_probs, r)
            centers.append(X[new_center_idx])
        return np.array(centers)
    else:
        raise ValueError(f"Unknown initialization method: {method}")

def assign_clusters(distances: np.ndarray) -> np.ndarray:
    """Assign data points to the nearest cluster."""
    return np.argmin(distances, axis=1)

def update_centers(X: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Update cluster centers based on current assignments."""
    centers = np.zeros((n_clusters, X.shape[1]))
    for k in range(n_clusters):
        centers[k] = np.mean(X[labels == k], axis=0)
    return centers

def compute_metrics(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> Dict[str, float]:
    """Compute clustering metrics."""
    distances = compute_distance(X, centers)
    intra_cluster_distances = np.min(distances, axis=1)
    total_distance = np.sum(intra_cluster_distances)

    metrics = {
        'inertia': total_distance,
        'silhouette_score': compute_silhouette(X, labels, centers)
    }
    return metrics

def compute_silhouette(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """Compute the silhouette score."""
    distances = compute_distance(X, centers)
    intra_cluster_distances = np.min(distances, axis=1)

    silhouette_scores = []
    for i in range(X.shape[0]):
        a = np.mean([distances[i][j] for j in range(len(labels)) if labels[j] == labels[i] and i != j])
        b = np.inf
        for k in range(len(np.unique(labels))):
            if k == labels[i]:
                continue
            b_temp = np.mean([distances[i][j] for j in range(len(labels)) if labels[j] == k])
            if b_temp < b:
                b = b_temp
        silhouette_scores.append((b - a) / max(a, b))
    return np.mean(silhouette_scores)

def k_means_fit(X: np.ndarray,
                n_clusters: int = 8,
                max_iter: int = 300,
                tol: float = 1e-4,
                normalize_method: str = 'standard',
                distance_metric: str = 'euclidean',
                init_method: str = 'kmeans++') -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]:
    """
    Perform K-means clustering.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    n_clusters : int, optional
        Number of clusters to form (default is 8).
    max_iter : int, optional
        Maximum number of iterations (default is 300).
    tol : float, optional
        Tolerance for convergence (default is 1e-4).
    normalize_method : str, optional
        Normalization method (default is 'standard').
    distance_metric : str, optional
        Distance metric to use (default is 'euclidean').
    init_method : str, optional
        Method for initializing cluster centers (default is 'kmeans++').

    Returns
    -------
    dict
        A dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate input
    validate_input(X, n_clusters)

    # Normalize data
    X_normalized = normalize_data(X, method=normalize_method)

    # Initialize centers
    centers = initialize_centers(X_normalized, n_clusters, method=init_method)

    # Main K-means loop
    for _ in range(max_iter):
        old_centers = centers.copy()

        # Compute distances and assign clusters
        distances = compute_distance(X_normalized, centers, metric=distance_metric)
        labels = assign_clusters(distances)

        # Update centers
        centers = update_centers(X_normalized, labels, n_clusters)

        # Check for convergence
        if np.linalg.norm(centers - old_centers) < tol:
            break

    # Compute metrics
    metrics = compute_metrics(X_normalized, labels, centers)

    # Prepare output
    result = {
        'labels': labels,
        'centers': centers
    }

    params_used = {
        'n_clusters': n_clusters,
        'max_iter': max_iter,
        'tol': tol,
        'normalize_method': normalize_method,
        'distance_metric': distance_metric,
        'init_method': init_method
    }

    warnings = []

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

################################################################################
# k_medoids
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_input(X: np.ndarray, k: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X must not contain NaN or infinite values")
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if k > X.shape[0]:
        raise ValueError("k cannot be greater than the number of samples")

def default_distance_metric(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Default Euclidean distance metric."""
    return np.sqrt(np.sum((X[:, np.newaxis] - Y) ** 2, axis=2))

def k_medoids_fit(
    X: np.ndarray,
    k: int,
    distance_metric: Callable[[np.ndarray, np.ndarray], np.ndarray] = default_distance_metric,
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform k-medoids clustering.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    k : int
        Number of clusters
    distance_metric : callable
        Distance metric function (default: Euclidean)
    max_iter : int
        Maximum number of iterations (default: 300)
    tol : float
        Tolerance for convergence (default: 1e-4)
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary containing:
        - 'medoids': array of medoid indices
        - 'labels': array of cluster labels
        - 'inertia': total within-cluster distance
        - 'iterations': number of iterations performed
    """
    validate_input(X, k)

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    medoids = np.random.choice(n_samples, size=k, replace=False)
    labels = np.zeros(n_samples, dtype=int)

    for _ in range(max_iter):
        # Assign each point to the nearest medoid
        distances = distance_metric(X, X[medoids])
        labels = np.argmin(distances, axis=1)

        # Update medoids by finding points that minimize total distance
        new_medoids = np.zeros_like(medoids)
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                # Calculate distances between all points in cluster
                cluster_distances = distance_metric(cluster_points, cluster_points)
                # Find point with minimum total distance to all other points in cluster
                total_distances = np.sum(cluster_distances, axis=1)
                new_medoids[i] = cluster_points[np.argmin(total_distances)]

        # Check for convergence
        if np.array_equal(medoids, new_medoids):
            break

        medoids = new_medoids

    # Calculate inertia (total within-cluster distance)
    distances = distance_metric(X, X[medoids])
    inertia = np.sum(np.min(distances, axis=1))

    return {
        'medoids': medoids,
        'labels': labels,
        'inertia': inertia,
        'iterations': _ + 1
    }

def compute_silhouette_score(
    X: np.ndarray,
    labels: np.ndarray,
    distance_metric: Callable[[np.ndarray, np.ndarray], np.ndarray] = default_distance_metric
) -> float:
    """
    Compute silhouette score for cluster evaluation.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    labels : np.ndarray
        Array of cluster labels
    distance_metric : callable
        Distance metric function (default: Euclidean)

    Returns:
    --------
    float
        Silhouette score
    """
    n_samples = X.shape[0]
    silhouette_scores = np.zeros(n_samples)

    for i in range(n_samples):
        # Calculate a(i) - average distance to other points in the same cluster
        cluster_points = X[labels == labels[i]]
        if len(cluster_points) > 1:
            distances = distance_metric(X[i:i+1], cluster_points)
            a_i = np.mean(distances[0][labels == labels[i]])
        else:
            a_i = 0

        # Calculate b(i) - minimum average distance to points in other clusters
        min_b_i = float('inf')
        for j in range(np.max(labels) + 1):
            if j != labels[i]:
                other_cluster_points = X[labels == j]
                distances = distance_metric(X[i:i+1], other_cluster_points)
                b_i = np.mean(distances[0])
                if b_i < min_b_i:
                    min_b_i = b_i

        # Calculate silhouette score for this sample
        if max(a_i, min_b_i) != 0:
            silhouette_scores[i] = (min_b_i - a_i) / max(a_i, min_b_i)
        else:
            silhouette_scores[i] = 0

    return np.mean(silhouette_scores)

# Example usage:
"""
X = np.random.rand(100, 5)  # Random data
result = k_medoids_fit(X, k=3)
silhouette = compute_silhouette_score(X, result['labels'])
"""

################################################################################
# fuzzy_c_means
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def fuzzy_c_means_fit(
    X: np.ndarray,
    n_clusters: int = 3,
    m: float = 2.0,
    max_iter: int = 100,
    tol: float = 1e-4,
    distance_metric: Union[str, Callable] = 'euclidean',
    normalization: Optional[str] = None,
    random_state: Optional[int] = None
) -> Dict:
    """
    Perform fuzzy c-means clustering on the input data.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    n_clusters : int, optional
        Number of clusters. Default is 3.
    m : float, optional
        Fuzziness parameter (m > 1). Default is 2.0.
    max_iter : int, optional
        Maximum number of iterations. Default is 100.
    tol : float, optional
        Tolerance for stopping criterion. Default is 1e-4.
    distance_metric : Union[str, Callable], optional
        Distance metric to use. Can be 'euclidean', 'manhattan', 'cosine',
        or a custom callable. Default is 'euclidean'.
    normalization : Optional[str], optional
        Normalization method. Can be 'standard', 'minmax', or None.
    random_state : Optional[int], optional
        Random seed for initialization. Default is None.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Dictionary with keys 'centers' and 'membership'
        - 'metrics': Dictionary with clustering metrics
        - 'params_used': Dictionary of parameters used
        - 'warnings': List of warnings

    Examples
    --------
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    >>> result = fuzzy_c_means_fit(X)
    """
    # Validate inputs
    _validate_inputs(X, n_clusters, m)

    # Initialize random state if provided
    rng = np.random.RandomState(random_state)

    # Normalize data if specified
    X_norm, norm_params = _normalize_data(X, normalization)

    # Initialize centers randomly
    centers = _initialize_centers(X_norm, n_clusters, rng)

    # Initialize membership matrix
    U = _initialize_membership(X_norm, centers, n_clusters, m)

    # Main iteration loop
    for _ in range(max_iter):
        prev_centers = centers.copy()

        # Update centers
        centers = _update_centers(X_norm, U, m)

        # Update membership matrix
        U = _update_membership(X_norm, centers, distance_metric, m)

        # Check for convergence
        if np.linalg.norm(centers - prev_centers) < tol:
            break

    # Calculate metrics
    metrics = _calculate_metrics(X_norm, centers, U, distance_metric)

    # Prepare output
    result = {
        'centers': centers,
        'membership': U
    }

    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'n_clusters': n_clusters,
            'm': m,
            'max_iter': max_iter,
            'tol': tol,
            'distance_metric': distance_metric,
            'normalization': normalization
        },
        'warnings': []
    }

    return output

def _validate_inputs(X: np.ndarray, n_clusters: int, m: float) -> None:
    """Validate input parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1")
    if m <= 1:
        raise ValueError("m must be greater than 1")

def _normalize_data(X: np.ndarray, method: Optional[str]) -> tuple:
    """Normalize the input data."""
    norm_params = {}
    X_norm = X.copy()

    if method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_norm = (X - mean) / std
        norm_params['mean'] = mean
        norm_params['std'] = std
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
        norm_params['min'] = min_val
        norm_params['max'] = max_val

    return X_norm, norm_params

def _initialize_centers(X: np.ndarray, n_clusters: int, rng: np.random.RandomState) -> np.ndarray:
    """Initialize cluster centers randomly."""
    n_samples = X.shape[0]
    indices = rng.choice(n_samples, size=n_clusters, replace=False)
    return X[indices]

def _initialize_membership(X: np.ndarray, centers: np.ndarray, n_clusters: int, m: float) -> np.ndarray:
    """Initialize membership matrix."""
    n_samples = X.shape[0]
    U = np.zeros((n_samples, n_clusters))
    for i in range(n_samples):
        distances = np.linalg.norm(X[i] - centers, axis=1)
        U[i] = (distances ** (-2 / (m - 1))) / np.sum(distances ** (-2 / (m - 1)))
    return U

def _update_centers(X: np.ndarray, U: np.ndarray, m: float) -> np.ndarray:
    """Update cluster centers."""
    n_clusters = U.shape[1]
    numerator = np.dot(U ** m, X)
    denominator = np.sum(U ** m, axis=0, keepdims=True).T
    return numerator / denominator

def _update_membership(
    X: np.ndarray,
    centers: np.ndarray,
    distance_metric: Union[str, Callable],
    m: float
) -> np.ndarray:
    """Update membership matrix."""
    n_samples = X.shape[0]
    n_clusters = centers.shape[0]

    if isinstance(distance_metric, str):
        distance_func = _get_distance_function(distance_metric)
    else:
        distance_func = distance_metric

    distances = np.zeros((n_samples, n_clusters))
    for i in range(n_samples):
        for j in range(n_clusters):
            distances[i, j] = distance_func(X[i], centers[j])

    U = (distances ** (-2 / (m - 1))) / np.sum(distances ** (-2 / (m - 1)), axis=1, keepdims=True)
    return U

def _get_distance_function(metric: str) -> Callable:
    """Get distance function based on metric name."""
    if metric == 'euclidean':
        return lambda x, y: np.linalg.norm(x - y)
    elif metric == 'manhattan':
        return lambda x, y: np.sum(np.abs(x - y))
    elif metric == 'cosine':
        return lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _calculate_metrics(
    X: np.ndarray,
    centers: np.ndarray,
    U: np.ndarray,
    distance_metric: Union[str, Callable]
) -> Dict:
    """Calculate clustering metrics."""
    if isinstance(distance_metric, str):
        distance_func = _get_distance_function(distance_metric)
    else:
        distance_func = distance_metric

    distances = np.zeros((X.shape[0], centers.shape[0]))
    for i in range(X.shape[0]):
        for j in range(centers.shape[0]):
            distances[i, j] = distance_func(X[i], centers[j])

    objective = np.sum(U ** 2 * distances)
    return {
        'objective': objective,
        'partition_coefficient': np.sum(U ** 2) / X.shape[0]
    }

################################################################################
# hierarchical_clustering
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def hierarchical_clustering_fit(
    data: np.ndarray,
    linkage_method: str = 'ward',
    metric: Union[str, Callable] = 'euclidean',
    normalization: Optional[str] = None,
    threshold: Optional[float] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Perform hierarchical clustering on the input data.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    linkage_method : str, optional
        Linkage method to use. Options: 'ward', 'complete', 'average', 'single'.
    metric : str or callable, optional
        Distance metric to use. Options: 'euclidean', 'manhattan', 'cosine'.
    normalization : str, optional
        Normalization method. Options: None, 'standard', 'minmax', 'robust'.
    threshold : float, optional
        Threshold for cutting the dendrogram.
    custom_distance : callable, optional
        Custom distance function.

    Returns:
    --------
    dict
        Dictionary containing clustering results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    validate_inputs(data)

    # Normalize data if specified
    normalized_data = apply_normalization(data, normalization)

    # Compute distance matrix
    distance_matrix = compute_distance_matrix(
        normalized_data,
        metric=metric,
        custom_distance=custom_distance
    )

    # Perform hierarchical clustering
    linkage_matrix = perform_hierarchical_clustering(
        distance_matrix,
        linkage_method=linkage_method
    )

    # Cut the dendrogram if threshold is specified
    labels = cut_dendrogram(linkage_matrix, threshold) if threshold is not None else None

    # Compute metrics
    metrics = compute_metrics(data, labels)

    return {
        'result': {'linkage_matrix': linkage_matrix, 'labels': labels},
        'metrics': metrics,
        'params_used': {
            'linkage_method': linkage_method,
            'metric': metric,
            'normalization': normalization,
            'threshold': threshold
        },
        'warnings': []
    }

def validate_inputs(data: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array.")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values.")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values.")

def apply_normalization(
    data: np.ndarray,
    normalization: Optional[str]
) -> np.ndarray:
    """Apply specified normalization to the data."""
    if normalization is None:
        return data
    elif normalization == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif normalization == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif normalization == 'robust':
        return (data - np.median(data, axis=0)) / (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def compute_distance_matrix(
    data: np.ndarray,
    metric: Union[str, Callable],
    custom_distance: Optional[Callable]
) -> np.ndarray:
    """Compute the distance matrix using specified metric or custom function."""
    n_samples = data.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    if custom_distance is not None:
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance_matrix[i, j] = custom_distance(data[i], data[j])
                distance_matrix[j, i] = distance_matrix[i, j]
    elif metric == 'euclidean':
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance_matrix[i, j] = np.linalg.norm(data[i] - data[j])
                distance_matrix[j, i] = distance_matrix[i, j]
    elif metric == 'manhattan':
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance_matrix[i, j] = np.sum(np.abs(data[i] - data[j]))
                distance_matrix[j, i] = distance_matrix[i, j]
    elif metric == 'cosine':
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance_matrix[i, j] = 1 - np.dot(data[i], data[j]) / (np.linalg.norm(data[i]) * np.linalg.norm(data[j]))
                distance_matrix[j, i] = distance_matrix[i, j]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return distance_matrix

def perform_hierarchical_clustering(
    distance_matrix: np.ndarray,
    linkage_method: str
) -> np.ndarray:
    """Perform hierarchical clustering using specified linkage method."""
    if linkage_method == 'ward':
        return _ward_linkage(distance_matrix)
    elif linkage_method == 'complete':
        return _complete_linkage(distance_matrix)
    elif linkage_method == 'average':
        return _average_linkage(distance_matrix)
    elif linkage_method == 'single':
        return _single_linkage(distance_matrix)
    else:
        raise ValueError(f"Unknown linkage method: {linkage_method}")

def _ward_linkage(distance_matrix: np.ndarray) -> np.ndarray:
    """Ward linkage method."""
    # Placeholder for actual implementation
    return np.zeros((1, 4))

def _complete_linkage(distance_matrix: np.ndarray) -> np.ndarray:
    """Complete linkage method."""
    # Placeholder for actual implementation
    return np.zeros((1, 4))

def _average_linkage(distance_matrix: np.ndarray) -> np.ndarray:
    """Average linkage method."""
    # Placeholder for actual implementation
    return np.zeros((1, 4))

def _single_linkage(distance_matrix: np.ndarray) -> np.ndarray:
    """Single linkage method."""
    # Placeholder for actual implementation
    return np.zeros((1, 4))

def cut_dendrogram(
    linkage_matrix: np.ndarray,
    threshold: float
) -> np.ndarray:
    """Cut the dendrogram to form flat clusters."""
    # Placeholder for actual implementation
    return np.zeros(linkage_matrix.shape[0])

def compute_metrics(
    data: np.ndarray,
    labels: Optional[np.ndarray]
) -> Dict:
    """Compute clustering metrics."""
    if labels is None:
        return {'silhouette_score': None, 'davies_bouldin_index': None}
    # Placeholder for actual implementation
    return {'silhouette_score': 0.5, 'davies_bouldin_index': 1.0}

################################################################################
# dbscan
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def dbscan_fit(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "euclidean",
    precomputed_metric: Optional[np.ndarray] = None,
    normalize: str = "none",
    algorithm: str = "auto",
    leaf_size: int = 30,
    p: Optional[float] = None
) -> Dict:
    """
    Perform DBSCAN clustering on the input data.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    eps : float, optional
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples : int, optional
        The number of samples in a neighborhood for a point to be considered as a core point.
    metric : str or callable, optional
        The distance metric to use. Can be 'euclidean', 'manhattan', 'cosine', or a custom callable.
    precomputed_metric : np.ndarray, optional
        Precomputed distance matrix of shape (n_samples, n_samples).
    normalize : str, optional
        Normalization method. Can be 'none', 'standard', 'minmax', or 'robust'.
    algorithm : str, optional
        Algorithm to use for nearest neighbors search. Can be 'auto', 'ball_tree', 'kd_tree', or 'brute'.
    leaf_size : int, optional
        Leaf size passed to BallTree or KDTree.
    p : float, optional
        Parameter for Minkowski metric.

    Returns:
    --------
    Dict containing:
        - "result": Cluster labels for each point.
        - "metrics": Dictionary of computed metrics.
        - "params_used": Parameters used for the computation.
        - "warnings": List of warnings encountered during computation.
    """
    # Validate inputs
    _validate_inputs(X, eps, min_samples, metric, precomputed_metric, normalize)

    # Normalize data if required
    X_normalized = _normalize_data(X, normalize)

    # Compute distance matrix or use precomputed
    if precomputed_metric is not None:
        dist_matrix = precomputed_metric
    else:
        dist_matrix = _compute_distance_matrix(X_normalized, metric, p)

    # Perform DBSCAN clustering
    labels = _dbscan_core(dist_matrix, eps, min_samples)

    # Compute metrics
    metrics = _compute_metrics(X_normalized, labels, dist_matrix)

    # Prepare output
    result = {
        "result": labels,
        "metrics": metrics,
        "params_used": {
            "eps": eps,
            "min_samples": min_samples,
            "metric": metric,
            "normalize": normalize,
            "algorithm": algorithm,
            "leaf_size": leaf_size
        },
        "warnings": []
    }

    return result

def _validate_inputs(
    X: np.ndarray,
    eps: float,
    min_samples: int,
    metric: Union[str, Callable],
    precomputed_metric: Optional[np.ndarray]
) -> None:
    """Validate input parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")
    if precomputed_metric is not None:
        if precomputed_metric.shape[0] != X.shape[0] or precomputed_metric.shape[1] != X.shape[0]:
            raise ValueError("precomputed_metric must have shape (n_samples, n_samples)")

def _normalize_data(
    X: np.ndarray,
    method: str
) -> np.ndarray:
    """Normalize the input data."""
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

def _compute_distance_matrix(
    X: np.ndarray,
    metric: Union[str, Callable],
    p: Optional[float]
) -> np.ndarray:
    """Compute the distance matrix."""
    n_samples = X.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))

    if callable(metric):
        for i in range(n_samples):
            for j in range(i, n_samples):
                dist_matrix[i, j] = metric(X[i], X[j])
                dist_matrix[j, i] = dist_matrix[i, j]
    elif metric == "euclidean":
        for i in range(n_samples):
            for j in range(i, n_samples):
                dist_matrix[i, j] = np.linalg.norm(X[i] - X[j])
                dist_matrix[j, i] = dist_matrix[i, j]
    elif metric == "manhattan":
        for i in range(n_samples):
            for j in range(i, n_samples):
                dist_matrix[i, j] = np.sum(np.abs(X[i] - X[j]))
                dist_matrix[j, i] = dist_matrix[i, j]
    elif metric == "cosine":
        for i in range(n_samples):
            for j in range(i, n_samples):
                dot_product = np.dot(X[i], X[j])
                norm_i = np.linalg.norm(X[i])
                norm_j = np.linalg.norm(X[j])
                dist_matrix[i, j] = 1 - (dot_product / (norm_i * norm_j + 1e-8))
                dist_matrix[j, i] = dist_matrix[i, j]
    elif metric == "minkowski":
        if p is None:
            raise ValueError("p must be specified for Minkowski distance")
        for i in range(n_samples):
            for j in range(i, n_samples):
                dist_matrix[i, j] = np.sum(np.abs(X[i] - X[j]) ** p) ** (1 / p)
                dist_matrix[j, i] = dist_matrix[i, j]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return dist_matrix

def _dbscan_core(
    dist_matrix: np.ndarray,
    eps: float,
    min_samples: int
) -> np.ndarray:
    """Core DBSCAN algorithm."""
    n_samples = dist_matrix.shape[0]
    labels = np.full(n_samples, -1)  # -1 means unvisited
    cluster_id = 0

    for i in range(n_samples):
        if labels[i] != -1:
            continue  # Already processed

        neighbors = _region_query(dist_matrix, i, eps)
        if len(neighbors) < min_samples:
            labels[i] = -1  # Noise point
        else:
            _expand_cluster(dist_matrix, i, neighbors, labels, cluster_id, eps, min_samples)
            cluster_id += 1

    return labels

def _region_query(
    dist_matrix: np.ndarray,
    point_idx: int,
    eps: float
) -> np.ndarray:
    """Find all points within eps distance of the given point."""
    return np.where(dist_matrix[point_idx] <= eps)[0]

def _expand_cluster(
    dist_matrix: np.ndarray,
    point_idx: int,
    neighbors: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
    eps: float,
    min_samples: int
) -> None:
    """Expand the cluster from the given point."""
    labels[point_idx] = cluster_id

    for neighbor in neighbors:
        if labels[neighbor] == -1:  # Noise point
            labels[neighbor] = cluster_id
        elif labels[neighbor] == -1:  # Unvisited point
            continue

        if labels[neighbor] != cluster_id:
            continue  # Already in another cluster

        new_neighbors = _region_query(dist_matrix, neighbor, eps)
        if len(new_neighbors) >= min_samples:
            neighbors = np.concatenate([neighbors, new_neighbors])

def _compute_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    dist_matrix: np.ndarray
) -> Dict:
    """Compute clustering metrics."""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    metrics = {
        "n_clusters": n_clusters,
        "n_noise_points": np.sum(labels == -1),
    }

    return metrics

################################################################################
# spectral_clustering
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def spectral_clustering_fit(
    X: np.ndarray,
    n_clusters: int = 8,
    affinity: str = 'rbf',
    gamma: Optional[float] = None,
    n_neighbors: int = 10,
    eigen_solver: str = 'arpack',
    n_components: Optional[int] = None,
    random_state: Optional[int] = None,
    n_init: int = 10,
    eigen_tol: float = 0.0,
    assign_labels: str = 'kmeans',
    degree: float = 1.0,
    coef0: float = 1.0,
    kernel_params: Optional[Dict] = None,
    affinity_callable: Optional[Callable] = None
) -> Dict:
    """
    Perform spectral clustering from features or similarity matrix.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    n_clusters : int, default=8
        The number of clusters to form.
    affinity : {'nearest_neighbors', 'rbf', 'precomputed', 'precompute'}, default='rbf'
        The affinity metric to use.
    gamma : float, default=None
        Kernel coefficient for rbf kernel. If None is given, it uses 1 / (2 * median_distance).
    n_neighbors : int, default=10
        Number of neighbors to use when constructing the affinity matrix from a nearest neighbors graph.
    eigen_solver : {'arpack', 'lobpcg', 'amg'}, default='arpack'
        The eigenvalue solver to use.
    n_components : int, default=None
        Number of eigenvectors to use for the spectral embedding.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization.
    n_init : int, default=10
        Number of times the k-means algorithm will be run with different centroid seeds.
    eigen_tol : float, default=0.0
        Tolerance for eigenvalue solver.
    assign_labels : {'kmeans', 'discretize'}, default='kmeans'
        The algorithm to assign labels.
    degree : float, default=1.0
        Degree of the polynomial kernel.
    coef0 : float, default=1.0
        Zero coefficient for polynomial and sigmoid kernels.
    kernel_params : dict, default=None
        Parameters for the custom affinity kernel.
    affinity_callable : callable, default=None
        A custom affinity function.

    Returns
    -------
    result : dict
        Dictionary containing the clustering results.
    """
    # Validate inputs
    _validate_inputs(X, n_clusters, affinity, gamma, n_neighbors,
                     eigen_solver, n_components, random_state,
                     n_init, eigen_tol, assign_labels)

    # Compute affinity matrix
    if affinity_callable is not None:
        affinity_matrix = _compute_custom_affinity(X, affinity_callable)
    else:
        if affinity == 'rbf':
            affinity_matrix = _compute_rbf_affinity(X, gamma=gamma)
        elif affinity == 'nearest_neighbors':
            affinity_matrix = _compute_nearest_neighbors_affinity(X, n_neighbors=n_neighbors)
        elif affinity == 'precomputed':
            if not np.allclose(X, X.T):
                raise ValueError("X must be a symmetric matrix when affinity='precomputed'")
            affinity_matrix = X.copy()
        else:
            raise ValueError(f"Unknown affinity: {affinity}")

    # Normalize the affinity matrix
    affinity_matrix = _normalize_affinity(affinity_matrix, degree=degree)

    # Compute the Laplacian matrix
    laplacian = _compute_laplacian(affinity_matrix)

    # Compute eigenvectors
    eigenvectors = _compute_eigenvectors(laplacian, n_components=n_components,
                                        eigen_solver=eigen_solver, eigen_tol=eigen_tol)

    # Perform clustering
    labels = _perform_clustering(eigenvectors, n_clusters=n_clusters,
                                 assign_labels=assign_labels, random_state=random_state)

    # Compute metrics
    metrics = _compute_metrics(X, labels)

    return {
        'result': {'labels': labels},
        'metrics': metrics,
        'params_used': {
            'n_clusters': n_clusters,
            'affinity': affinity,
            'gamma': gamma,
            'n_neighbors': n_neighbors,
            'eigen_solver': eigen_solver,
            'n_components': n_components,
            'random_state': random_state,
            'n_init': n_init,
            'eigen_tol': eigen_tol,
            'assign_labels': assign_labels
        },
        'warnings': []
    }

def _validate_inputs(
    X: np.ndarray,
    n_clusters: int,
    affinity: str,
    gamma: Optional[float],
    n_neighbors: int,
    eigen_solver: str,
    n_components: Optional[int],
    random_state: Optional[int],
    n_init: int,
    eigen_tol: float,
    assign_labels: str
) -> None:
    """Validate the inputs for spectral clustering."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if n_clusters <= 0:
        raise ValueError("n_clusters must be a positive integer")
    if affinity not in ['nearest_neighbors', 'rbf', 'precomputed']:
        raise ValueError(f"Unknown affinity: {affinity}")
    if gamma is not None and gamma <= 0:
        raise ValueError("gamma must be positive or None")
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be a positive integer")
    if eigen_solver not in ['arpack', 'lobpcg', 'amg']:
        raise ValueError(f"Unknown eigen_solver: {eigen_solver}")
    if n_components is not None and (n_components <= 0 or n_components > X.shape[0]):
        raise ValueError("n_components must be None, positive and <= n_samples")
    if random_state is not None and (not isinstance(random_state, int) or random_state <= 0):
        raise ValueError("random_state must be None or a positive integer")
    if n_init <= 0:
        raise ValueError("n_init must be a positive integer")
    if eigen_tol < 0:
        raise ValueError("eigen_tol must be non-negative")
    if assign_labels not in ['kmeans', 'discretize']:
        raise ValueError(f"Unknown assign_labels: {assign_labels}")

def _compute_rbf_affinity(X: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
    """Compute the RBF affinity matrix."""
    if gamma is None:
        distances = np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
        median_distance = np.median(distances[distances > 0])
        gamma = 1.0 / (2 * median_distance ** 2)
    pairwise_dists = np.sum((X[:, np.newaxis] - X) ** 2, axis=2)
    affinity_matrix = np.exp(-gamma * pairwise_dists)
    return affinity_matrix

def _compute_nearest_neighbors_affinity(X: np.ndarray, n_neighbors: int = 10) -> np.ndarray:
    """Compute the nearest neighbors affinity matrix."""
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    distances, indices = nbrs.kneighbors(X)
    affinity_matrix = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        affinity_matrix[i, indices[i]] = 1.0 / (distances[i] + 1e-6)
    return affinity_matrix

def _compute_custom_affinity(X: np.ndarray, affinity_callable: Callable) -> np.ndarray:
    """Compute the custom affinity matrix."""
    return affinity_callable(X)

def _normalize_affinity(affinity_matrix: np.ndarray, degree: float = 1.0) -> np.ndarray:
    """Normalize the affinity matrix."""
    row_sums = np.sum(affinity_matrix, axis=1)
    normalized_affinity = affinity_matrix / (row_sums[:, np.newaxis] ** degree)
    return normalized_affinity

def _compute_laplacian(affinity_matrix: np.ndarray) -> np.ndarray:
    """Compute the Laplacian matrix."""
    degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))
    laplacian = degree_matrix - affinity_matrix
    return laplacian

def _compute_eigenvectors(
    laplacian: np.ndarray,
    n_components: Optional[int] = None,
    eigen_solver: str = 'arpack',
    eigen_tol: float = 0.0
) -> np.ndarray:
    """Compute the eigenvectors of the Laplacian matrix."""
    if n_components is None:
        n_components = min(laplacian.shape[0] - 2, 10)
    if eigen_solver == 'arpack':
        from scipy.sparse.linalg import eigs
        eigenvalues, eigenvectors = eigs(laplacian, k=n_components + 1, tol=eigen_tol)
    else:
        raise ValueError(f"Unknown eigen_solver: {eigen_solver}")
    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Return the eigenvectors corresponding to the smallest eigenvalues
    return eigenvectors[:, 1:n_components + 1]

def _perform_clustering(
    eigenvectors: np.ndarray,
    n_clusters: int,
    assign_labels: str = 'kmeans',
    random_state: Optional[int] = None
) -> np.ndarray:
    """Perform clustering on the eigenvectors."""
    if assign_labels == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(eigenvectors)
    else:
        raise ValueError(f"Unknown assign_labels: {assign_labels}")
    return labels

def _compute_metrics(X: np.ndarray, labels: np.ndarray) -> Dict:
    """Compute clustering metrics."""
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    metrics = {
        'silhouette': silhouette_score(X, labels),
        'calinski_harabasz': calinski_harabasz_score(X, labels)
    }
    return metrics

################################################################################
# gaussian_mixture_models
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def gaussian_mixture_models_fit(
    X: np.ndarray,
    n_components: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    normalization: str = 'standard',
    distance_metric: Union[str, Callable] = 'euclidean',
    solver: str = 'em',
    regularization: Optional[str] = None,
    metric: Union[str, Callable] = 'loglikelihood',
    custom_metric: Optional[Callable] = None,
    weights_init: Optional[np.ndarray] = None,
    means_init: Optional[np.ndarray] = None,
    covars_init: Optional[np.ndarray] = None
) -> Dict:
    """
    Fit Gaussian Mixture Models to the data.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    n_components : int
        Number of mixture components
    max_iter : int, optional
        Maximum number of iterations (default: 100)
    tol : float, optional
        Tolerance for convergence (default: 1e-4)
    random_state : Optional[int], optional
        Random seed for reproducibility (default: None)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard')
    distance_metric : Union[str, Callable], optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable (default: 'euclidean')
    solver : str, optional
        Solver method ('em', 'gradient_descent') (default: 'em')
    regularization : Optional[str], optional
        Regularization method ('none', 'l1', 'l2') (default: None)
    metric : Union[str, Callable], optional
        Metric for evaluation ('loglikelihood', 'bic', 'aic') or custom callable (default: 'loglikelihood')
    custom_metric : Optional[Callable], optional
        Custom metric function (default: None)
    weights_init : Optional[np.ndarray], optional
        Initial weights for components (default: None)
    means_init : Optional[np.ndarray], optional
        Initial means for components (default: None)
    covars_init : Optional[np.ndarray], optional
        Initial covariances for components (default: None)

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used and warnings.
    """
    # Validate inputs
    _validate_inputs(X, n_components, weights_init, means_init, covars_init)

    # Normalize data
    X_norm = _normalize_data(X, normalization)

    # Initialize parameters
    weights, means, covars = _initialize_parameters(
        X_norm, n_components, random_state, weights_init, means_init, covars_init
    )

    # Main optimization loop
    for iteration in range(max_iter):
        # Expectation step
        responsibilities = _expectation_step(X_norm, weights, means, covars, distance_metric)

        # Maximization step
        new_weights, new_means, new_covars = _maximization_step(
            X_norm, responsibilities, weights, means, covars, regularization
        )

        # Check convergence
        if _check_convergence(weights, means, covars, new_weights, new_means, new_covars, tol):
            break

        weights, means, covars = new_weights, new_means, new_covars

    # Calculate metrics
    metrics = _calculate_metrics(X_norm, weights, means, covars, metric, custom_metric)

    # Prepare output
    result = {
        'weights': weights,
        'means': means,
        'covariances': covars,
        'responsibilities': responsibilities
    }

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'max_iter': max_iter,
            'tol': tol,
            'normalization': normalization,
            'distance_metric': distance_metric,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

def _validate_inputs(
    X: np.ndarray,
    n_components: int,
    weights_init: Optional[np.ndarray],
    means_init: Optional[np.ndarray],
    covars_init: Optional[np.ndarray]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if weights_init is not None and len(weights_init) != n_components:
        raise ValueError("weights_init must have length equal to n_components")
    if means_init is not None and means_init.shape[0] != n_components:
        raise ValueError("means_init must have first dimension equal to n_components")
    if covars_init is not None and covars_init.shape[0] != n_components:
        raise ValueError("covars_init must have first dimension equal to n_components")

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize the input data."""
    if method == 'none':
        return X
    elif method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _initialize_parameters(
    X: np.ndarray,
    n_components: int,
    random_state: Optional[int],
    weights_init: Optional[np.ndarray],
    means_init: Optional[np.ndarray],
    covars_init: Optional[np.ndarray]
) -> tuple:
    """Initialize the parameters for GMM."""
    np.random.seed(random_state)

    if weights_init is None:
        weights = np.ones(n_components) / n_components
    else:
        weights = weights_init

    if means_init is None:
        random_indices = np.random.choice(X.shape[0], n_components, replace=False)
        means = X[random_indices]
    else:
        means = means_init

    if covars_init is None:
        covars = np.array([np.cov(X.T) for _ in range(n_components)])
    else:
        covars = covars_init

    return weights, means, covars

def _expectation_step(
    X: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    distance_metric: Union[str, Callable]
) -> np.ndarray:
    """Perform the expectation step of EM algorithm."""
    n_samples = X.shape[0]
    n_components = weights.shape[0]

    responsibilities = np.zeros((n_samples, n_components))

    for k in range(n_components):
        if distance_metric == 'euclidean':
            dist = np.sum((X - means[k])**2, axis=1)
        elif distance_metric == 'manhattan':
            dist = np.sum(np.abs(X - means[k]), axis=1)
        elif distance_metric == 'cosine':
            dist = 1 - np.dot(X, means[k]) / (np.linalg.norm(X, axis=1) * np.linalg.norm(means[k]))
        elif distance_metric == 'minkowski':
            dist = np.sum(np.abs(X - means[k])**2, axis=1)**(1/3)
        elif callable(distance_metric):
            dist = distance_metric(X, means[k])
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        exponent = -0.5 * dist
        if np.linalg.det(covars[k]) == 0:
            responsibilities[:, k] = weights[k]
        else:
            responsibilities[:, k] = weights[k] * np.exp(exponent) / np.sqrt(np.linalg.det(2 * np.pi * covars[k]))

    responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

    return responsibilities

def _maximization_step(
    X: np.ndarray,
    responsibilities: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    regularization: Optional[str]
) -> tuple:
    """Perform the maximization step of EM algorithm."""
    n_samples = X.shape[0]
    n_components = weights.shape[0]

    # Update weights
    new_weights = np.sum(responsibilities, axis=0) / n_samples

    # Update means
    new_means = np.zeros_like(means)
    for k in range(n_components):
        weighted_X = responsibilities[:, k, np.newaxis] * X
        new_means[k] = np.sum(weighted_X, axis=0) / np.sum(responsibilities[:, k])

    # Update covariances
    new_covars = np.zeros_like(covars)
    for k in range(n_components):
        diff = X - new_means[k]
        weighted_diff = responsibilities[:, k, np.newaxis] * diff
        new_covars[k] = np.dot(weighted_diff.T, diff) / np.sum(responsibilities[:, k])

    # Apply regularization if needed
    if regularization == 'l1':
        new_covars += 1e-6
    elif regularization == 'l2':
        new_covars += np.eye(new_covars.shape[1]) * 1e-6

    return new_weights, new_means, new_covars

def _check_convergence(
    weights_old: np.ndarray,
    means_old: np.ndarray,
    covars_old: np.ndarray,
    weights_new: np.ndarray,
    means_new: np.ndarray,
    covars_new: np.ndarray,
    tol: float
) -> bool:
    """Check if the algorithm has converged."""
    weights_diff = np.linalg.norm(weights_new - weights_old)
    means_diff = np.linalg.norm(means_new - means_old)
    covars_diff = np.linalg.norm(covars_new - covars_old)

    return (weights_diff < tol) and (means_diff < tol) and (covars_diff < tol)

def _calculate_metrics(
    X: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Calculate the metrics for the GMM."""
    metrics = {}

    if metric == 'loglikelihood':
        loglikelihood = _compute_loglikelihood(X, weights, means, covars)
        metrics['loglikelihood'] = loglikelihood
    elif metric == 'bic':
        bic = _compute_bic(X, weights, means, covars)
        metrics['bic'] = bic
    elif metric == 'aic':
        aic = _compute_aic(X, weights, means, covars)
        metrics['aic'] = aic
    elif callable(metric):
        metrics['custom_metric'] = metric(X, weights, means, covars)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if custom_metric is not None:
        metrics['custom_metric'] = custom_metric(X, weights, means, covars)

    return metrics

def _compute_loglikelihood(
    X: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray
) -> float:
    """Compute the loglikelihood of the data."""
    n_samples = X.shape[0]
    loglikelihood = 0.0

    for k in range(weights.shape[0]):
        exponent = -0.5 * np.sum((X - means[k])**2 / covars[k], axis=1)
        log_det = np.log(np.linalg.det(covars[k]))
        loglikelihood += weights[k] * np.sum(np.exp(exponent) / np.sqrt((2 * np.pi)**X.shape[1] * np.exp(log_det)))

    return loglikelihood / n_samples

def _compute_bic(
    X: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray
) -> float:
    """Compute the Bayesian Information Criterion."""
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_components = weights.shape[0]

    loglikelihood = _compute_loglikelihood(X, weights, means, covars)
    n_params = n_components * (n_features + n_features**2 + 1)

    return -2 * loglikelihood * n_samples + n_params * np.log(n_samples)

def _compute_aic(
    X: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray
) -> float:
    """Compute the Akaike Information Criterion."""
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_components = weights.shape[0]

    loglikelihood = _compute_loglikelihood(X, weights, means, covars)
    n_params = n_components * (n_features + n_features**2 + 1)

    return -2 * loglikelihood * n_samples + 2 * n_params

################################################################################
# mean_shift
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def mean_shift_fit(
    X: np.ndarray,
    bandwidth: float = 1.0,
    max_iter: int = 300,
    cluster_all: bool = False,
    n_jobs: Optional[int] = None,
    distance_metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    normalize: Optional[str] = None,
    seed: Optional[int] = None
) -> Dict:
    """
    Perform mean shift clustering on the input data.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    bandwidth : float, optional
        Bandwidth for mean shift.
    max_iter : int, optional
        Maximum number of iterations.
    cluster_all : bool, optional
        If True, clusters all points (including noise).
    n_jobs : int or None, optional
        Number of parallel jobs to run.
    distance_metric : str or callable, optional
        Distance metric to use. Can be 'euclidean', 'manhattan', 'cosine',
        or a custom callable.
    normalize : str or None, optional
        Normalization method. Can be 'standard', 'minmax', 'robust', or None.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns:
    --------
    dict
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, bandwidth, max_iter)

    # Normalize data if specified
    X_normalized = _normalize_data(X, normalize) if normalize else X

    # Initialize centers
    centers = _initialize_centers(X_normalized, seed)

    # Perform mean shift iterations
    centers = _mean_shift_iterations(
        X_normalized,
        centers,
        bandwidth,
        max_iter,
        distance_metric
    )

    # Assign points to clusters
    labels = _assign_labels(X_normalized, centers, distance_metric)

    # Compute metrics
    metrics = _compute_metrics(X_normalized, centers, labels, distance_metric)

    # Prepare output
    result = {
        'result': {
            'centers': centers,
            'labels': labels
        },
        'metrics': metrics,
        'params_used': {
            'bandwidth': bandwidth,
            'max_iter': max_iter,
            'cluster_all': cluster_all,
            'distance_metric': distance_metric,
            'normalize': normalize
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, bandwidth: float, max_iter: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if bandwidth <= 0:
        raise ValueError("Bandwidth must be positive.")
    if max_iter <= 0:
        raise ValueError("Max iterations must be positive.")

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize the input data."""
    if method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _initialize_centers(X: np.ndarray, seed: Optional[int]) -> np.ndarray:
    """Initialize cluster centers."""
    if seed is not None:
        np.random.seed(seed)
    return X[np.random.choice(X.shape[0], size=min(10, X.shape[0]), replace=False)]

def _mean_shift_iterations(
    X: np.ndarray,
    centers: np.ndarray,
    bandwidth: float,
    max_iter: int,
    distance_metric: Union[str, Callable]
) -> np.ndarray:
    """Perform mean shift iterations."""
    for _ in range(max_iter):
        new_centers = np.zeros_like(centers)
        for i, center in enumerate(centers):
            # Compute weights
            distances = _compute_distance(X, center, distance_metric)
            weights = np.exp(-distances ** 2 / (2 * bandwidth ** 2))
            # Update center
            new_centers[i] = np.sum(weights[:, np.newaxis] * X, axis=0) / np.sum(weights)
        # Check for convergence
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return centers

def _compute_distance(
    X: np.ndarray,
    center: np.ndarray,
    metric: Union[str, Callable]
) -> np.ndarray:
    """Compute distances between points and center."""
    if metric == 'euclidean':
        return np.linalg.norm(X - center, axis=1)
    elif metric == 'manhattan':
        return np.sum(np.abs(X - center), axis=1)
    elif metric == 'cosine':
        return 1 - np.dot(X, center) / (np.linalg.norm(X, axis=1) * np.linalg.norm(center))
    elif callable(metric):
        return np.array([metric(x, center) for x in X])
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _assign_labels(
    X: np.ndarray,
    centers: np.ndarray,
    distance_metric: Union[str, Callable]
) -> np.ndarray:
    """Assign points to the nearest cluster center."""
    labels = np.zeros(X.shape[0], dtype=int)
    for i, x in enumerate(X):
        distances = np.array([_compute_distance(np.array([x]), center, distance_metric)[0] for center in centers])
        labels[i] = np.argmin(distances)
    return labels

def _compute_metrics(
    X: np.ndarray,
    centers: np.ndarray,
    labels: np.ndarray,
    distance_metric: Union[str, Callable]
) -> Dict:
    """Compute clustering metrics."""
    metrics = {}
    for i, center in enumerate(centers):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            distances = np.array([_compute_distance(np.array([x]), center, distance_metric)[0] for x in cluster_points])
            metrics[f'cluster_{i}_mean_distance'] = np.mean(distances)
    return metrics

################################################################################
# affinity_propagation
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_affinity_propagation_inputs(
    X: np.ndarray,
    preference: Optional[Union[np.ndarray, float]] = None,
    metric: str = 'euclidean',
    max_iter: int = 200,
    convergence_iter: int = 15,
    damping: float = 0.5,
    copy: bool = True
) -> Dict[str, Any]:
    """
    Validate inputs for affinity propagation clustering.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix, shape (n_samples, n_features)
    preference : Optional[Union[np.ndarray, float]]
        Preferences for samples. If None, median of input similarities is used.
    metric : str
        Distance metric to use. Options: 'euclidean', 'manhattan', 'cosine'.
    max_iter : int
        Maximum number of iterations.
    damping : float
        Damping factor between 0.5 and 1.
    copy : bool
        Whether to make a copy of input data.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing validated parameters and warnings.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if np.any(np.isnan(X)):
        raise ValueError("Input contains NaN values")

    params = {
        'X': X.copy() if copy else X,
        'preference': preference,
        'metric': metric,
        'max_iter': max_iter,
        'convergence_iter': convergence_iter,
        'damping': damping
    }

    warnings = []
    if preference is not None:
        if isinstance(preference, (int, float)):
            params['preference'] = np.full(X.shape[0], preference)
        elif isinstance(preference, np.ndarray):
            if len(preference) != X.shape[0]:
                raise ValueError("preference must have same length as number of samples")
        else:
            raise TypeError("preference must be a scalar or numpy array")

    if not 0.5 <= damping < 1:
        warnings.append("damping should be between 0.5 and 1")

    return {'params': params, 'warnings': warnings}

def compute_similarities(
    X: np.ndarray,
    metric: str = 'euclidean',
    custom_metric: Optional[Callable] = None
) -> np.ndarray:
    """
    Compute similarity matrix using specified metric.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix, shape (n_samples, n_features)
    metric : str
        Distance metric to use.
    custom_metric : Optional[Callable]
        Custom distance function if needed.

    Returns:
    --------
    np.ndarray
        Similarity matrix, shape (n_samples, n_samples)
    """
    if custom_metric is not None:
        return -custom_metric(X)

    n_samples = X.shape[0]
    similarities = np.zeros((n_samples, n_samples))

    if metric == 'euclidean':
        for i in range(n_samples):
            similarities[i] = -np.sqrt(np.sum((X[i] - X) ** 2, axis=1))
    elif metric == 'manhattan':
        for i in range(n_samples):
            similarities[i] = -np.sum(np.abs(X[i] - X), axis=1)
    elif metric == 'cosine':
        for i in range(n_samples):
            similarities[i] = -np.dot(X[i], X.T) / (
                np.linalg.norm(X[i]) * np.linalg.norm(X, axis=1)
            )
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return similarities

def affinity_propagation_fit(
    X: np.ndarray,
    preference: Optional[Union[np.ndarray, float]] = None,
    metric: str = 'euclidean',
    max_iter: int = 200,
    convergence_iter: int = 15,
    damping: float = 0.5,
    copy: bool = True
) -> Dict[str, Any]:
    """
    Perform affinity propagation clustering.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix, shape (n_samples, n_features)
    preference : Optional[Union[np.ndarray, float]]
        Preferences for samples.
    metric : str
        Distance metric to use.
    max_iter : int
        Maximum number of iterations.
    convergence_iter : int
        Number of iterations with no change to consider as converged.
    damping : float
        Damping factor between 0.5 and 1.
    copy : bool
        Whether to make a copy of input data.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings.
    """
    # Validate inputs
    validation = validate_affinity_propagation_inputs(
        X, preference, metric, max_iter, convergence_iter, damping, copy
    )
    params = validation['params']
    warnings = validation['warnings']

    # Compute similarities
    S = compute_similarities(params['X'], params['metric'])

    if params['preference'] is None:
        params['preference'] = np.median(S)

    # Initialize availability and responsibility matrices
    A = np.zeros_like(S)
    R = np.zeros_like(S)

    # Main iteration loop
    for _ in range(params['max_iter']):
        # Update responsibility matrix
        R_old = R.copy()
        for i in range(S.shape[0]):
            for k in range(S.shape[0]):
                if i == k:
                    R[i, k] = params['preference'][i] - np.max(A[i] + S[i])
                else:
                    candidate = S[i, k] - np.max(A[i] + S[i])
                    R[i, k] = candidate if candidate > 0 else 0

        # Update availability matrix
        A_old = A.copy()
        for k in range(S.shape[0]):
            positive_resp = R[:, k].copy()
            positive_resp[k] = -np.inf
            A[:, k] = np.sum(np.minimum(0, R[:, k] + np.vstack([positive_resp] * S.shape[0])), axis=0)

        # Damping
        R = params['damping'] * R + (1 - params['damping']) * R_old
        A = params['damping'] * A + (1 - params['damping']) * A_old

        # Check for convergence
        if np.allclose(R, R_old) and np.allclose(A, A_old):
            break

    # Identify exemplars
    I = np.argmax(R + A, axis=1) == np.arange(S.shape[0])

    # Assign clusters
    labels = np.argmax(R + A, axis=1)

    return {
        'result': {
            'labels': labels,
            'exemplars': I,
            'affinity_matrix': R + A
        },
        'metrics': {
            'n_clusters': np.sum(I),
            'silhouette_score': None  # Could be added with additional computation
        },
        'params_used': params,
        'warnings': warnings
    }

################################################################################
# birch
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def birch_fit(
    data: np.ndarray,
    threshold: float = 0.5,
    branching_factor: int = 50,
    max_leaf_nodes: Optional[int] = None,
    distance_metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "euclidean",
    normalization: Optional[str] = None,
    copy: bool = True
) -> Dict:
    """
    Perform BIRCH clustering on the input data.

    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (n_samples, n_features)
    threshold : float, default=0.5
        The radius of the subcluster. If a new point's distance to an existing subcluster is
        greater than this threshold, it will form a new subcluster.
    branching_factor : int, default=50
        Maximum number of entries in the subclusters at the leaf node.
    max_leaf_nodes : int, optional
        Maximum number of leaf nodes. If None, there is no limit.
    distance_metric : str or callable, default="euclidean"
        Distance metric to use. Can be "euclidean", "manhattan", "cosine" or a custom callable.
    normalization : str, optional
        Normalization method. Can be "standard", "minmax" or None.
    copy : bool, default=True
        Whether to make a copy of the input data.

    Returns:
    --------
    dict
        Dictionary containing:
        - "result": Clustering results (subclusters)
        - "metrics": Computed metrics
        - "params_used": Parameters used for the computation
        - "warnings": Any warnings generated during computation

    Example:
    --------
    >>> data = np.random.rand(100, 5)
    >>> result = birch_fit(data, threshold=0.7, branching_factor=30)
    """
    # Validate inputs
    data = _validate_input(data, copy)

    if normalization is not None:
        data = _apply_normalization(data, normalization)

    # Initialize CF tree
    cf_tree = _CFTree(threshold, branching_factor, max_leaf_nodes)

    # Insert data points into CF tree
    for point in data:
        cf_tree.insert(point, distance_metric)

    # Build the CF tree
    cf_tree.build()

    # Get subclusters
    subclusters = cf_tree.get_subclusters()

    # Calculate metrics
    metrics = _calculate_metrics(subclusters, data)

    return {
        "result": subclusters,
        "metrics": metrics,
        "params_used": {
            "threshold": threshold,
            "branching_factor": branching_factor,
            "max_leaf_nodes": max_leaf_nodes,
            "distance_metric": distance_metric,
            "normalization": normalization
        },
        "warnings": []
    }

def _validate_input(data: np.ndarray, copy: bool) -> np.ndarray:
    """Validate input data."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Input data must be 2-dimensional")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values")
    return data.copy() if copy else data

def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to the data."""
    if method == "standard":
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == "minmax":
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _calculate_metrics(subclusters: list, data: np.ndarray) -> Dict:
    """Calculate clustering metrics."""
    # Calculate silhouette score
    silhouette = _silhouette_score(subclusters, data)

    return {
        "silhouette_score": silhouette
    }

def _silhouette_score(subclusters: list, data: np.ndarray) -> float:
    """Calculate silhouette score for the clustering."""
    # Implementation of silhouette score
    pass

class _CFTree:
    """CF Tree implementation for BIRCH clustering."""

    def __init__(self, threshold: float, branching_factor: int, max_leaf_nodes: Optional[int]):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.max_leaf_nodes = max_leaf_nodes
        self.root = None

    def insert(self, point: np.ndarray, distance_metric: Union[str, Callable]):
        """Insert a data point into the CF tree."""
        pass

    def build(self):
        """Build the CF tree."""
        pass

    def get_subclusters(self) -> list:
        """Get subclusters from the CF tree."""
        pass

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Manhattan distance between two points."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine distance between two points."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _get_distance_metric(metric: Union[str, Callable]) -> Callable:
    """Get distance metric function."""
    if callable(metric):
        return metric
    elif metric == "euclidean":
        return _euclidean_distance
    elif metric == "manhattan":
        return _manhattan_distance
    elif metric == "cosine":
        return _cosine_distance
    else:
        raise ValueError(f"Unknown distance metric: {metric}")
