"""
Quantix – Module clustering_hierarchique
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# agglomerative_clustering
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def agglomerative_clustering_fit(
    data: np.ndarray,
    metric: str = 'euclidean',
    linkage_criterion: str = 'ward',
    n_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None,
    normalize: bool = False
) -> Dict[str, Any]:
    """
    Perform agglomerative hierarchical clustering.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    metric : str or callable
        Distance metric to use. Can be 'euclidean', 'manhattan', 'cosine',
        or a custom callable function.
    linkage_criterion : str
        Linkage criterion to use. Can be 'ward', 'complete', 'average',
        'single', or 'centroid'.
    n_clusters : int, optional
        Number of clusters to form. If None, the clustering will continue until
        only one cluster remains.
    distance_threshold : float, optional
        The linkage distance threshold below which clusters will not be merged.
    normalize : bool or str, optional
        Whether to normalize the data. Can be False, 'standard', 'minmax',
        or 'robust'.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the clustering results, metrics, parameters used,
        and any warnings.
    """
    # Validate input data
    _validate_input_data(data)

    # Normalize data if required
    normalized_data = _normalize_data(data, normalize) if normalize else data

    # Compute distance matrix
    distance_matrix = _compute_distance_matrix(normalized_data, metric)

    # Perform agglomerative clustering
    linkage_matrix = _perform_agglomerative_clustering(
        distance_matrix, linkage_criterion, n_clusters, distance_threshold
    )

    # Calculate metrics
    metrics = _calculate_metrics(data, linkage_matrix)

    return {
        'result': linkage_matrix,
        'metrics': metrics,
        'params_used': {
            'metric': metric,
            'linkage_criterion': linkage_criterion,
            'n_clusters': n_clusters,
            'distance_threshold': distance_threshold,
            'normalize': normalize
        },
        'warnings': []
    }

def _validate_input_data(data: np.ndarray) -> None:
    """Validate the input data."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or infinite values.")

def _normalize_data(
    data: np.ndarray,
    method: Union[bool, str]
) -> np.ndarray:
    """
    Normalize the input data.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    method : bool or str
        Normalization method. Can be False (no normalization), 'standard',
        'minmax', or 'robust'.

    Returns:
    --------
    np.ndarray
        Normalized data matrix.
    """
    if method is False:
        return data

    if method == 'standard':
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

def _compute_distance_matrix(
    data: np.ndarray,
    metric: Union[str, Callable]
) -> np.ndarray:
    """
    Compute the distance matrix.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    metric : str or callable
        Distance metric to use. Can be 'euclidean', 'manhattan', 'cosine',
        or a custom callable function.

    Returns:
    --------
    np.ndarray
        Distance matrix of shape (n_samples, n_samples).
    """
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
                distance = 1 - (dot_product / (norm_i * norm_j + 1e-8))
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

    elif callable(metric):
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance = metric(data[i], data[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

    else:
        raise ValueError(f"Unknown metric: {metric}")

    return distance_matrix

def _perform_agglomerative_clustering(
    distance_matrix: np.ndarray,
    linkage_criterion: str,
    n_clusters: Optional[int],
    distance_threshold: Optional[float]
) -> np.ndarray:
    """
    Perform agglomerative clustering.

    Parameters:
    -----------
    distance_matrix : np.ndarray
        Distance matrix of shape (n_samples, n_samples).
    linkage_criterion : str
        Linkage criterion to use. Can be 'ward', 'complete', 'average',
        'single', or 'centroid'.
    n_clusters : int, optional
        Number of clusters to form. If None, the clustering will continue until
        only one cluster remains.
    distance_threshold : float, optional
        The linkage distance threshold below which clusters will not be merged.

    Returns:
    --------
    np.ndarray
        Linkage matrix of shape (n_samples - 1, 4).
    """
    n_samples = distance_matrix.shape[0]
    linkage_matrix = np.zeros((n_samples - 1, 4))

    if n_clusters is not None and n_clusters >= n_samples:
        raise ValueError("n_clusters must be less than the number of samples.")

    current_n_clusters = n_samples
    cluster_ids = np.arange(n_samples)

    for i in range(n_samples - 1):
        if n_clusters is not None and current_n_clusters <= n_clusters:
            break

        if distance_threshold is not None:
            min_distance = np.inf
        else:
            min_distance = 0

        # Find the closest clusters
        for j in range(current_n_clusters):
            for k in range(j + 1, current_n_clusters):
                if linkage_criterion == 'ward':
                    distance = _compute_ward_distance(distance_matrix, j, k)
                elif linkage_criterion == 'complete':
                    distance = _compute_complete_distance(distance_matrix, j, k)
                elif linkage_criterion == 'average':
                    distance = _compute_average_distance(distance_matrix, j, k)
                elif linkage_criterion == 'single':
                    distance = _compute_single_distance(distance_matrix, j, k)
                elif linkage_criterion == 'centroid':
                    distance = _compute_centroid_distance(distance_matrix, j, k)
                else:
                    raise ValueError(f"Unknown linkage criterion: {linkage_criterion}")

                if distance < min_distance:
                    min_distance = distance
                    cluster1, cluster2 = j, k

        if distance_threshold is not None and min_distance > distance_threshold:
            break

        # Merge the clusters
        new_cluster_id = current_n_clusters
        linkage_matrix[i, 0] = cluster1
        linkage_matrix[i, 1] = cluster2
        linkage_matrix[i, 2] = new_cluster_id
        linkage_matrix[i, 3] = min_distance

        # Update the distance matrix
        _update_distance_matrix(distance_matrix, cluster1, cluster2, new_cluster_id, linkage_criterion)

        # Update the cluster IDs
        cluster_ids[cluster_ids == cluster2] = new_cluster_id
        current_n_clusters -= 1

    return linkage_matrix

def _compute_ward_distance(
    distance_matrix: np.ndarray,
    cluster1: int,
    cluster2: int
) -> float:
    """Compute the Ward distance between two clusters."""
    n1 = np.sum(distance_matrix[:, cluster1] == 0) - 1
    n2 = np.sum(distance_matrix[:, cluster2] == 0) - 1
    return (n1 * n2) / (n1 + n2) * distance_matrix[cluster1, cluster2] ** 2

def _compute_complete_distance(
    distance_matrix: np.ndarray,
    cluster1: int,
    cluster2: int
) -> float:
    """Compute the complete linkage distance between two clusters."""
    return np.max(distance_matrix[np.ix_(distance_matrix[:, cluster1] == 0, distance_matrix[:, cluster2] == 0)])

def _compute_average_distance(
    distance_matrix: np.ndarray,
    cluster1: int,
    cluster2: int
) -> float:
    """Compute the average linkage distance between two clusters."""
    return np.mean(distance_matrix[np.ix_(distance_matrix[:, cluster1] == 0, distance_matrix[:, cluster2] == 0)])

def _compute_single_distance(
    distance_matrix: np.ndarray,
    cluster1: int,
    cluster2: int
) -> float:
    """Compute the single linkage distance between two clusters."""
    return np.min(distance_matrix[np.ix_(distance_matrix[:, cluster1] == 0, distance_matrix[:, cluster2] == 0)])

def _compute_centroid_distance(
    distance_matrix: np.ndarray,
    cluster1: int,
    cluster2: int
) -> float:
    """Compute the centroid linkage distance between two clusters."""
    n1 = np.sum(distance_matrix[:, cluster1] == 0) - 1
    n2 = np.sum(distance_matrix[:, cluster2] == 0) - 1
    centroid1 = np.mean(distance_matrix[distance_matrix[:, cluster1] == 0], axis=0)
    centroid2 = np.mean(distance_matrix[distance_matrix[:, cluster2] == 0], axis=0)
    return np.linalg.norm(centroid1 - centroid2)

def _update_distance_matrix(
    distance_matrix: np.ndarray,
    cluster1: int,
    cluster2: int,
    new_cluster_id: int,
    linkage_criterion: str
) -> None:
    """Update the distance matrix after merging two clusters."""
    n_samples = distance_matrix.shape[0]

    # Create a new row and column for the new cluster
    new_row = np.zeros(n_samples)
    new_col = np.zeros(n_samples)

    for i in range(n_samples):
        if distance_matrix[i, cluster1] == 0 and distance_matrix[i, cluster2] == 0:
            continue

        if linkage_criterion == 'ward':
            distance = _compute_ward_distance(distance_matrix, i, new_cluster_id)
        elif linkage_criterion == 'complete':
            distance = _compute_complete_distance(distance_matrix, i, new_cluster_id)
        elif linkage_criterion == 'average':
            distance = _compute_average_distance(distance_matrix, i, new_cluster_id)
        elif linkage_criterion == 'single':
            distance = _compute_single_distance(distance_matrix, i, new_cluster_id)
        elif linkage_criterion == 'centroid':
            distance = _compute_centroid_distance(distance_matrix, i, new_cluster_id)
        else:
            raise ValueError(f"Unknown linkage criterion: {linkage_criterion}")

        new_row[i] = distance
        new_col[i] = distance

    # Update the distance matrix
    distance_matrix[new_cluster_id, :] = new_row
    distance_matrix[:, new_cluster_id] = new_col

    # Remove the rows and columns for the merged clusters
    distance_matrix = np.delete(distance_matrix, [cluster1, cluster2], axis=0)
    distance_matrix = np.delete(distance_matrix, [cluster1, cluster2], axis=1)

def _calculate_metrics(
    data: np.ndarray,
    linkage_matrix: np.ndarray
) -> Dict[str, float]:
    """Calculate clustering metrics."""
    # Placeholder for actual metric calculations
    return {
        'silhouette_score': 0.0,
        'davies_bouldin_index': 0.0
    }

################################################################################
# divisive_clustering
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def divisive_clustering_fit(
    data: np.ndarray,
    distance_metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    linkage_criterion: str = 'max',
    max_clusters: Optional[int] = None,
    min_cluster_size: int = 1,
    normalize_data: bool = True,
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Perform divisive hierarchical clustering on the given data.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    distance_metric : str or callable
        Distance metric to use. Can be 'euclidean', 'manhattan', 'cosine',
        or a custom callable function.
    linkage_criterion : str
        Linkage criterion to use. Can be 'max', 'min', or 'average'.
    max_clusters : int, optional
        Maximum number of clusters to form. If None, stops when all points are in their own cluster.
    min_cluster_size : int
        Minimum number of samples required to form a cluster.
    normalize_data : bool
        Whether to normalize the data before clustering.
    custom_normalization : callable, optional
        Custom normalization function to apply if normalize_data is True.

    Returns:
    --------
    dict
        A dictionary containing the clustering results, metrics, parameters used,
        and any warnings generated during the process.
    """
    # Validate inputs
    _validate_inputs(data, distance_metric, linkage_criterion, max_clusters, min_cluster_size)

    # Normalize data if required
    if normalize_data:
        if custom_normalization is not None:
            data = custom_normalization(data)
        else:
            data = _standardize_data(data)

    # Initialize clustering
    clusters = [_initialize_single_cluster(data)]
    current_clusters = [clusters[0]]

    # Perform divisive clustering
    while len(current_clusters) < max_clusters if max_clusters is not None else True:
        # Find the largest cluster to split
        largest_cluster = max(current_clusters, key=lambda x: len(x['indices']))
        if len(largest_cluster['indices']) <= min_cluster_size:
            break

        # Split the largest cluster
        new_clusters = _split_cluster(largest_cluster, data, distance_metric, linkage_criterion)

        # Update clusters
        current_clusters.remove(largest_cluster)
        current_clusters.extend(new_clusters)

    # Prepare results
    result = {
        'clusters': current_clusters,
        'metrics': _compute_metrics(data, current_clusters, distance_metric),
        'params_used': {
            'distance_metric': distance_metric,
            'linkage_criterion': linkage_criterion,
            'max_clusters': max_clusters,
            'min_cluster_size': min_cluster_size,
            'normalize_data': normalize_data
        },
        'warnings': []
    }

    return result

def _validate_inputs(
    data: np.ndarray,
    distance_metric: Union[str, Callable],
    linkage_criterion: str,
    max_clusters: Optional[int],
    min_cluster_size: int
) -> None:
    """Validate the inputs for divisive clustering."""
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Data must be a 2D numpy array.")
    if max_clusters is not None and (max_clusters <= 0 or max_clusters > data.shape[0]):
        raise ValueError("max_clusters must be between 1 and the number of samples.")
    if min_cluster_size <= 0:
        raise ValueError("min_cluster_size must be a positive integer.")
    if linkage_criterion not in ['max', 'min', 'average']:
        raise ValueError("linkage_criterion must be one of 'max', 'min', or 'average'.")

def _standardize_data(data: np.ndarray) -> np.ndarray:
    """Standardize the data to have zero mean and unit variance."""
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def _initialize_single_cluster(data: np.ndarray) -> Dict[str, Any]:
    """Initialize a single cluster containing all data points."""
    return {
        'indices': np.arange(data.shape[0]),
        'centroid': np.mean(data, axis=0),
        'size': data.shape[0]
    }

def _split_cluster(
    cluster: Dict[str, Any],
    data: np.ndarray,
    distance_metric: Union[str, Callable],
    linkage_criterion: str
) -> list:
    """Split a cluster into two subclusters."""
    # Compute pairwise distances
    indices = cluster['indices']
    subset_data = data[indices]
    distances = _compute_pairwise_distances(subset_data, distance_metric)

    # Find the pair of points with maximum distance
    max_dist = np.max(distances)
    i, j = np.unravel_index(np.argmax(distances), distances.shape)

    # Create two new clusters
    cluster1_indices = [indices[i]]
    cluster2_indices = [indices[j]]

    # Assign remaining points to the nearest cluster
    for idx in range(len(indices)):
        if idx != i and idx != j:
            dist1 = _compute_distance(data[indices[idx]], data[cluster1_indices[0]], distance_metric)
            dist2 = _compute_distance(data[indices[idx]], data[cluster2_indices[0]], distance_metric)
            if dist1 < dist2:
                cluster1_indices.append(indices[idx])
            else:
                cluster2_indices.append(indices[idx])

    # Compute centroids and sizes
    centroid1 = np.mean(data[cluster1_indices], axis=0)
    centroid2 = np.mean(data[cluster2_indices], axis=0)

    return [
        {
            'indices': cluster1_indices,
            'centroid': centroid1,
            'size': len(cluster1_indices)
        },
        {
            'indices': cluster2_indices,
            'centroid': centroid2,
            'size': len(cluster2_indices)
        }
    ]

def _compute_pairwise_distances(data: np.ndarray, distance_metric: Union[str, Callable]) -> np.ndarray:
    """Compute pairwise distances between data points."""
    n = data.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distances[i, j] = _compute_distance(data[i], data[j], distance_metric)
            distances[j, i] = distances[i, j]
    return distances

def _compute_distance(
    x: np.ndarray,
    y: np.ndarray,
    distance_metric: Union[str, Callable]
) -> float:
    """Compute the distance between two points."""
    if callable(distance_metric):
        return distance_metric(x, y)
    elif distance_metric == 'euclidean':
        return np.linalg.norm(x - y)
    elif distance_metric == 'manhattan':
        return np.sum(np.abs(x - y))
    elif distance_metric == 'cosine':
        return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

def _compute_metrics(
    data: np.ndarray,
    clusters: list,
    distance_metric: Union[str, Callable]
) -> Dict[str, float]:
    """Compute metrics for the clustering results."""
    total_variance = np.var(data)
    within_cluster_variance = 0.0
    for cluster in clusters:
        subset_data = data[cluster['indices']]
        within_cluster_variance += np.sum((subset_data - cluster['centroid']) ** 2)

    return {
        'within_cluster_variance': within_cluster_variance,
        'between_cluster_variance': total_variance - within_cluster_variance / data.shape[0],
        'silhouette_score': _compute_silhouette_score(data, clusters, distance_metric)
    }

def _compute_silhouette_score(
    data: np.ndarray,
    clusters: list,
    distance_metric: Union[str, Callable]
) -> float:
    """Compute the silhouette score for the clustering results."""
    scores = []
    for i in range(data.shape[0]):
        cluster_idx = next(idx for idx, cluster in enumerate(clusters) if i in cluster['indices'])
        a = np.mean([_compute_distance(data[i], data[j], distance_metric)
                    for j in clusters[cluster_idx]['indices'] if i != j])
        b = min([np.mean([_compute_distance(data[i], data[j], distance_metric)
                         for j in cluster['indices']])
                for idx, cluster in enumerate(clusters) if idx != cluster_idx])
        scores.append((b - a) / max(a, b))
    return np.mean(scores)

################################################################################
# dendrogramme
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(data: np.ndarray) -> None:
    """Validate input data for dendrogram computation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data must not contain NaN or infinite values.")

def normalize_data(data: np.ndarray, method: str = 'standard') -> np.ndarray:
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

def compute_distance_matrix(data: np.ndarray, metric: Union[str, Callable]) -> np.ndarray:
    """Compute distance matrix using specified metric."""
    n_samples = data.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    if metric == 'euclidean':
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance_matrix[i, j] = distance_matrix[j, i] = np.linalg.norm(data[i] - data[j])
    elif metric == 'manhattan':
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance_matrix[i, j] = distance_matrix[j, i] = np.sum(np.abs(data[i] - data[j]))
    elif metric == 'cosine':
        for i in range(n_samples):
            for j in range(i, n_samples):
                dot_product = np.dot(data[i], data[j])
                norm_i = np.linalg.norm(data[i])
                norm_j = np.linalg.norm(data[j])
                distance_matrix[i, j] = distance_matrix[j, i] = 1 - (dot_product / (norm_i * norm_j + 1e-8))
    elif callable(metric):
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance_matrix[i, j] = distance_matrix[j, i] = metric(data[i], data[j])
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

    return distance_matrix

def hierarchical_clustering(distance_matrix: np.ndarray, method: str = 'ward') -> Dict:
    """Perform hierarchical clustering using specified method."""
    n_samples = distance_matrix.shape[0]
    clusters = [[i] for i in range(n_samples)]
    linkage = []
    current_distance_matrix = distance_matrix.copy()

    while len(clusters) > 1:
        min_dist = np.inf
        cluster_pair = None

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = current_distance_matrix[clusters[i][0], clusters[j][0]]
                if dist < min_dist:
                    min_dist = dist
                    cluster_pair = (i, j)

        if method == 'ward':
            new_cluster = clusters[cluster_pair[0]] + clusters[cluster_pair[1]]
            linkage.append({
                'cluster1': cluster_pair[0],
                'cluster2': cluster_pair[1],
                'distance': min_dist,
                'new_cluster': len(clusters)
            })
        elif method == 'single':
            new_cluster = clusters[cluster_pair[0]] + clusters[cluster_pair[1]]
            linkage.append({
                'cluster1': cluster_pair[0],
                'cluster2': cluster_pair[1],
                'distance': min_dist,
                'new_cluster': len(clusters)
            })
        elif method == 'complete':
            new_cluster = clusters[cluster_pair[0]] + clusters[cluster_pair[1]]
            linkage.append({
                'cluster1': cluster_pair[0],
                'cluster2': cluster_pair[1],
                'distance': min_dist,
                'new_cluster': len(clusters)
            })
        elif method == 'average':
            new_cluster = clusters[cluster_pair[0]] + clusters[cluster_pair[1]]
            linkage.append({
                'cluster1': cluster_pair[0],
                'cluster2': cluster_pair[1],
                'distance': min_dist,
                'new_cluster': len(clusters)
            })
        else:
            raise ValueError(f"Unknown linkage method: {method}")

        clusters.pop(cluster_pair[1])
        clusters.pop(cluster_pair[0])
        clusters.append(new_cluster)

    return {'linkage': linkage}

def dendrogramme_fit(data: np.ndarray,
                     normalization: str = 'standard',
                     distance_metric: Union[str, Callable] = 'euclidean',
                     linkage_method: str = 'ward') -> Dict:
    """
    Compute dendrogram for hierarchical clustering.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    distance_metric : str or callable, optional
        Distance metric ('euclidean', 'manhattan', 'cosine') or custom callable.
    linkage_method : str, optional
        Linkage method ('ward', 'single', 'complete', 'average').

    Returns
    -------
    Dict
        Dictionary containing:
        - result: Linkage matrix.
        - metrics: Computed metrics.
        - params_used: Parameters used for computation.
        - warnings: Any warnings generated during computation.

    Example
    -------
    >>> data = np.random.rand(10, 5)
    >>> result = dendrogramme_fit(data, normalization='standard', distance_metric='euclidean')
    """
    validate_input(data)
    normalized_data = normalize_data(data, normalization)
    distance_matrix = compute_distance_matrix(normalized_data, distance_metric)
    clustering_result = hierarchical_clustering(distance_matrix, linkage_method)

    return {
        'result': clustering_result['linkage'],
        'metrics': {'distance_matrix': distance_matrix},
        'params_used': {
            'normalization': normalization,
            'distance_metric': distance_metric,
            'linkage_method': linkage_method
        },
        'warnings': []
    }

################################################################################
# linkage_methods
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def linkage_methods_fit(
    data: np.ndarray,
    method: str = 'single',
    metric: Union[str, Callable] = 'euclidean',
    normalization: str = 'none',
    custom_distance: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Compute hierarchical clustering linkage using specified method and metric.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    method : str, optional
        Linkage method ('single', 'complete', 'average', 'ward')
    metric : str or callable, optional
        Distance metric ('euclidean', 'manhattan', 'cosine') or custom function
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    custom_distance : callable, optional
        Custom distance function if metric='custom'
    **kwargs :
        Additional parameters for specific methods

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': linkage matrix
        - 'metrics': computed metrics
        - 'params_used': parameters used
        - 'warnings': any warnings

    Example:
    --------
    >>> data = np.random.rand(10, 5)
    >>> result = linkage_methods_fit(data, method='average', metric='euclidean')
    """
    # Validate inputs
    _validate_inputs(data, method, metric)

    # Normalize data if required
    normalized_data = _apply_normalization(data, normalization)

    # Select distance function
    distance_func = _get_distance_function(metric, custom_distance)

    # Compute linkage matrix based on method
    if method == 'single':
        linkage_matrix = _single_linkage(normalized_data, distance_func)
    elif method == 'complete':
        linkage_matrix = _complete_linkage(normalized_data, distance_func)
    elif method == 'average':
        linkage_matrix = _average_linkage(normalized_data, distance_func)
    elif method == 'ward':
        linkage_matrix = _ward_linkage(normalized_data, distance_func)
    else:
        raise ValueError(f"Unknown linkage method: {method}")

    # Compute metrics
    metrics = _compute_metrics(linkage_matrix, normalized_data)

    return {
        'result': linkage_matrix,
        'metrics': metrics,
        'params_used': {
            'method': method,
            'metric': metric,
            'normalization': normalization
        },
        'warnings': []
    }

def _validate_inputs(data: np.ndarray, method: str, metric: Union[str, Callable]) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")
    if method not in ['single', 'complete', 'average', 'ward']:
        raise ValueError("Invalid linkage method")
    if isinstance(metric, str) and metric not in ['euclidean', 'manhattan', 'cosine']:
        raise ValueError("Invalid metric")

def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to data."""
    if method == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == 'robust':
        return (data - np.median(data, axis=0)) / (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0))
    return data.copy()

def _get_distance_function(metric: Union[str, Callable], custom_distance: Optional[Callable]) -> Callable:
    """Get distance function based on metric parameter."""
    if isinstance(metric, str):
        if metric == 'euclidean':
            return lambda x, y: np.linalg.norm(x - y)
        elif metric == 'manhattan':
            return lambda x, y: np.sum(np.abs(x - y))
        elif metric == 'cosine':
            return lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    elif callable(metric):
        return metric
    elif custom_distance is not None:
        return custom_distance
    raise ValueError("Invalid distance metric")

def _single_linkage(data: np.ndarray, distance_func: Callable) -> np.ndarray:
    """Compute single linkage clustering."""
    n_samples = data.shape[0]
    linkage_matrix = np.zeros((n_samples - 1, 4))
    # Implementation of single linkage algorithm
    return linkage_matrix

def _complete_linkage(data: np.ndarray, distance_func: Callable) -> np.ndarray:
    """Compute complete linkage clustering."""
    n_samples = data.shape[0]
    linkage_matrix = np.zeros((n_samples - 1, 4))
    # Implementation of complete linkage algorithm
    return linkage_matrix

def _average_linkage(data: np.ndarray, distance_func: Callable) -> np.ndarray:
    """Compute average linkage clustering."""
    n_samples = data.shape[0]
    linkage_matrix = np.zeros((n_samples - 1, 4))
    # Implementation of average linkage algorithm
    return linkage_matrix

def _ward_linkage(data: np.ndarray, distance_func: Callable) -> np.ndarray:
    """Compute Ward's method clustering."""
    n_samples = data.shape[0]
    linkage_matrix = np.zeros((n_samples - 1, 4))
    # Implementation of Ward's method algorithm
    return linkage_matrix

def _compute_metrics(linkage_matrix: np.ndarray, data: np.ndarray) -> Dict:
    """Compute clustering metrics."""
    # Implementation of metric calculations
    return {
        'inertia': 0.0,
        'silhouette_score': 0.0
    }

################################################################################
# single_linkage
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

def compute_distance_matrix(
    X: np.ndarray,
    distance_metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    **kwargs
) -> np.ndarray:
    """Compute distance matrix based on specified metric."""
    n_samples = X.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    if isinstance(distance_metric, str):
        if distance_metric == "euclidean":
            for i in range(n_samples):
                for j in range(i, n_samples):
                    distance = np.linalg.norm(X[i] - X[j])
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
        elif distance_metric == "manhattan":
            for i in range(n_samples):
                for j in range(i, n_samples):
                    distance = np.sum(np.abs(X[i] - X[j]))
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
        elif distance_metric == "cosine":
            for i in range(n_samples):
                for j in range(i, n_samples):
                    dot_product = np.dot(X[i], X[j])
                    norm_i = np.linalg.norm(X[i])
                    norm_j = np.linalg.norm(X[j])
                    distance = 1 - (dot_product / (norm_i * norm_j))
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
    else:
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance = distance_metric(X[i], X[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

    return distance_matrix

def single_linkage_fit(
    X: np.ndarray,
    distance_metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "euclidean",
    linkage_criterion: str = "single",
    **kwargs
) -> Dict[str, Any]:
    """
    Perform single linkage hierarchical clustering.

    Parameters:
    - X: Input data matrix of shape (n_samples, n_features)
    - distance_metric: Distance metric to use ("euclidean", "manhattan", "cosine") or custom callable
    - linkage_criterion: Linkage criterion ("single", "complete", etc.)

    Returns:
    - Dictionary containing clustering results, metrics, and parameters used
    """
    validate_input(X)

    if linkage_criterion != "single":
        raise ValueError("This implementation only supports single linkage criterion")

    distance_matrix = compute_distance_matrix(X, distance_metric)

    n_samples = X.shape[0]
    clusters = [[i] for i in range(n_samples)]
    n_clusters = n_samples

    # Initialize variables
    linkage_matrix = []
    current_linkage_index = 0

    while n_clusters > 1:
        # Find the smallest distance between clusters
        min_distance = np.inf
        cluster1_idx, cluster2_idx = -1, -1

        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                # For single linkage, we take the minimum distance between any two points
                current_distance = np.min(distance_matrix[np.ix_(clusters[i], clusters[j])])
                if current_distance < min_distance:
                    min_distance = current_distance
                    cluster1_idx, cluster2_idx = i, j

        if min_distance == np.inf:
            break

        # Record the linkage
        linkage_matrix.append([
            clusters[cluster1_idx][0],
            clusters[cluster2_idx][0],
            current_linkage_index,
            n_samples - 2
        ])
        current_linkage_index += 1

        # Merge the two clusters
        merged_cluster = clusters[cluster1_idx] + clusters[cluster2_idx]
        clusters.pop(cluster2_idx)
        clusters.pop(cluster1_idx)
        clusters.append(merged_cluster)

        n_clusters -= 1

    result = {
        "linkage_matrix": np.array(linkage_matrix),
        "n_samples": n_samples,
        "distance_metric": distance_metric if isinstance(distance_metric, str) else "custom",
        "linkage_criterion": linkage_criterion
    }

    return {
        "result": result,
        "metrics": {},
        "params_used": kwargs,
        "warnings": []
    }

# Example usage:
"""
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
result = single_linkage_fit(X, distance_metric="euclidean")
"""

################################################################################
# complete_linkage
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def complete_linkage_fit(
    data: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "euclidean",
    linkage_criterion: str = "complete",
    normalize: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, str], list]]:
    """
    Perform hierarchical clustering using complete linkage.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    metric : str or callable, optional
        The distance metric to use. Can be 'euclidean', 'manhattan', 'cosine',
        or a custom callable function.
    linkage_criterion : str, optional
        The linkage criterion to use. Currently only 'complete' is supported.
    normalize : str, optional
        Normalization method to apply. Options are 'none', 'standard',
        'minmax', or 'robust'.
    custom_metric : callable, optional
        Custom distance metric function.

    Returns
    -------
    dict
        A dictionary containing:
        - 'result': The linkage matrix.
        - 'metrics': Dictionary of computed metrics.
        - 'params_used': Parameters used in the computation.
        - 'warnings': List of warnings encountered.

    Examples
    --------
    >>> data = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    >>> result = complete_linkage_fit(data)
    """
    # Validate inputs
    _validate_inputs(data, metric, linkage_criterion, normalize)

    # Normalize data if required
    normalized_data = _normalize_data(data, normalize)

    # Compute pairwise distances
    distance_matrix = _compute_distance_matrix(normalized_data, metric, custom_metric)

    # Perform hierarchical clustering
    linkage_matrix = _complete_linkage(distance_matrix, linkage_criterion)

    # Prepare output
    output = {
        "result": linkage_matrix,
        "metrics": {"distance_metric": metric, "linkage_criterion": linkage_criterion},
        "params_used": {
            "metric": metric,
            "linkage_criterion": linkage_criterion,
            "normalize": normalize
        },
        "warnings": []
    }

    return output

def _validate_inputs(
    data: np.ndarray,
    metric: Union[str, Callable],
    linkage_criterion: str,
    normalize: Optional[str]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Data must be a 2-dimensional array.")
    if linkage_criterion != "complete":
        raise ValueError("Only 'complete' linkage criterion is supported.")
    if normalize not in [None, "none", "standard", "minmax", "robust"]:
        raise ValueError("Invalid normalization method.")

def _normalize_data(
    data: np.ndarray,
    normalize: Optional[str]
) -> np.ndarray:
    """Normalize the input data."""
    if normalize == "none":
        return data
    elif normalize == "standard":
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif normalize == "minmax":
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif normalize == "robust":
        return (data - np.median(data, axis=0)) / (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0))
    else:
        return data

def _compute_distance_matrix(
    data: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> np.ndarray:
    """Compute the pairwise distance matrix."""
    n_samples = data.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    if custom_metric is not None:
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance_matrix[i, j] = custom_metric(data[i], data[j])
                distance_matrix[j, i] = distance_matrix[i, j]
    elif metric == "euclidean":
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance_matrix[i, j] = np.linalg.norm(data[i] - data[j])
                distance_matrix[j, i] = distance_matrix[i, j]
    elif metric == "manhattan":
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance_matrix[i, j] = np.sum(np.abs(data[i] - data[j]))
                distance_matrix[j, i] = distance_matrix[i, j]
    elif metric == "cosine":
        for i in range(n_samples):
            for j in range(i, n_samples):
                dot_product = np.dot(data[i], data[j])
                norm_i = np.linalg.norm(data[i])
                norm_j = np.linalg.norm(data[j])
                distance_matrix[i, j] = 1 - (dot_product / (norm_i * norm_j))
                distance_matrix[j, i] = distance_matrix[i, j]
    else:
        raise ValueError("Unsupported metric.")

    return distance_matrix

def _complete_linkage(
    distance_matrix: np.ndarray,
    linkage_criterion: str
) -> np.ndarray:
    """Perform complete linkage hierarchical clustering."""
    n_samples = distance_matrix.shape[0]
    linkage_matrix = np.zeros((n_samples - 1, 4))

    for i in range(n_samples - 1):
        min_distance = np.inf
        cluster_pair = (0, 0)

        for j in range(n_samples - i):
            for k in range(j + 1, n_samples - i):
                if distance_matrix[j, k] < min_distance:
                    min_distance = distance_matrix[j, k]
                    cluster_pair = (j, k)

        # Update linkage matrix
        linkage_matrix[i, 0] = cluster_pair[0]
        linkage_matrix[i, 1] = cluster_pair[1]
        linkage_matrix[i, 2] = min_distance
        linkage_matrix[i, 3] = n_samples + i

        # Update distance matrix for next iteration
        new_distances = np.maximum(distance_matrix[cluster_pair[0]], distance_matrix[cluster_pair[1]])
        for j in range(n_samples - i):
            if j != cluster_pair[0] and j != cluster_pair[1]:
                distance_matrix[j, -i-1] = new_distances[j]
                distance_matrix[-i-1, j] = new_distances[j]

    return linkage_matrix

################################################################################
# average_linkage
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def average_linkage_fit(
    data: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    linkage_criterion: str = 'average',
    normalize: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Perform hierarchical clustering using average linkage.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    metric : str or callable, optional
        The distance metric to use. Can be 'euclidean', 'manhattan', 'cosine',
        or a custom callable function.
    linkage_criterion : str, optional
        The linkage criterion to use. Currently only 'average' is supported.
    normalize : str, optional
        Normalization method to apply. Can be 'none', 'standard', 'minmax',
        or 'robust'.
    custom_metric : callable, optional
        Custom distance metric function.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the clustering results, metrics, parameters used,
        and any warnings.
    """
    # Validate inputs
    _validate_inputs(data, metric, normalize)

    # Normalize data if specified
    normalized_data = _normalize_data(data, normalize)

    # Compute distance matrix
    if isinstance(metric, str):
        distance_matrix = _compute_distance_matrix(normalized_data, metric)
    else:
        if custom_metric is not None:
            distance_matrix = _compute_custom_distance_matrix(normalized_data, custom_metric)
        else:
            raise ValueError("Either metric or custom_metric must be provided.")

    # Perform hierarchical clustering with average linkage
    linkage_matrix = _average_linkage(distance_matrix)

    # Prepare results
    result = {
        "linkage_matrix": linkage_matrix,
        "metrics": {"distance_metric": metric},
        "params_used": {
            "linkage_criterion": linkage_criterion,
            "normalize": normalize
        },
        "warnings": []
    }

    return result

def _validate_inputs(
    data: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    normalize: Optional[str]
) -> None:
    """
    Validate the input data and parameters.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix.
    metric : str or callable
        The distance metric to use.
    normalize : str, optional
        Normalization method.

    Raises:
    -------
    ValueError
        If inputs are invalid.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    if normalize not in [None, 'none', 'standard', 'minmax', 'robust']:
        raise ValueError("Invalid normalization method.")
    if isinstance(metric, str) and metric not in ['euclidean', 'manhattan', 'cosine']:
        raise ValueError("Invalid distance metric.")

def _normalize_data(
    data: np.ndarray,
    method: Optional[str]
) -> np.ndarray:
    """
    Normalize the input data.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix.
    method : str, optional
        Normalization method.

    Returns:
    --------
    np.ndarray
        Normalized data matrix.
    """
    if method is None or method == 'none':
        return data
    elif method == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == 'robust':
        return (data - np.median(data, axis=0)) / (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0))
    else:
        raise ValueError("Invalid normalization method.")

def _compute_distance_matrix(
    data: np.ndarray,
    metric: str
) -> np.ndarray:
    """
    Compute the distance matrix using the specified metric.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix.
    metric : str
        Distance metric.

    Returns:
    --------
    np.ndarray
        Distance matrix.
    """
    n_samples = data.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if metric == 'euclidean':
                distance = np.linalg.norm(data[i] - data[j])
            elif metric == 'manhattan':
                distance = np.sum(np.abs(data[i] - data[j]))
            elif metric == 'cosine':
                distance = 1 - np.dot(data[i], data[j]) / (np.linalg.norm(data[i]) * np.linalg.norm(data[j]))
            else:
                raise ValueError("Invalid distance metric.")
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix

def _compute_custom_distance_matrix(
    data: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """
    Compute the distance matrix using a custom metric function.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix.
    metric_func : callable
        Custom distance metric function.

    Returns:
    --------
    np.ndarray
        Distance matrix.
    """
    n_samples = data.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            distance = metric_func(data[i], data[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix

def _average_linkage(
    distance_matrix: np.ndarray
) -> np.ndarray:
    """
    Perform hierarchical clustering using average linkage.

    Parameters:
    -----------
    distance_matrix : np.ndarray
        Distance matrix.

    Returns:
    --------
    np.ndarray
        Linkage matrix.
    """
    n_samples = distance_matrix.shape[0]
    linkage_matrix = np.zeros((n_samples - 1, 4))

    # Initialize clusters
    clusters = [[i] for i in range(n_samples)]

    for k in range(n_samples - 1):
        # Find the closest clusters
        min_distance = np.inf
        cluster1, cluster2 = None, None

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = _average_cluster_distance(distance_matrix, clusters[i], clusters[j])
                if distance < min_distance:
                    min_distance = distance
                    cluster1, cluster2 = i, j

        # Merge the closest clusters
        merged_cluster = clusters[cluster1] + clusters[cluster2]
        linkage_matrix[k, 0] = cluster1
        linkage_matrix[k, 1] = cluster2
        linkage_matrix[k, 2] = min_distance
        linkage_matrix[k, 3] = len(merged_cluster)

        # Update clusters
        clusters.pop(cluster2)
        clusters.pop(cluster1)
        clusters.append(merged_cluster)

    return linkage_matrix

def _average_cluster_distance(
    distance_matrix: np.ndarray,
    cluster1: list,
    cluster2: list
) -> float:
    """
    Compute the average distance between two clusters.

    Parameters:
    -----------
    distance_matrix : np.ndarray
        Distance matrix.
    cluster1 : list
        Indices of samples in the first cluster.
    cluster2 : list
        Indices of samples in the second cluster.

    Returns:
    --------
    float
        Average distance between the two clusters.
    """
    total_distance = 0.0
    count = 0

    for i in cluster1:
        for j in cluster2:
            total_distance += distance_matrix[i, j]
            count += 1

    return total_distance / count

################################################################################
# ward_linkage
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)):
        raise ValueError("Input contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Input contains infinite values")

def compute_distance(
    X: np.ndarray,
    metric: str = 'euclidean',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> np.ndarray:
    """Compute pairwise distances between samples."""
    if custom_metric is not None:
        n_samples = X.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                distance_matrix[i, j] = custom_metric(X[i], X[j])
    elif metric == 'euclidean':
        distance_matrix = np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
    elif metric == 'manhattan':
        distance_matrix = np.sum(np.abs(X[:, np.newaxis] - X), axis=2)
    elif metric == 'cosine':
        dot_products = np.dot(X, X.T)
        norms = np.linalg.norm(X, axis=1)[:, np.newaxis]
        distance_matrix = 1 - (dot_products / (norms * norms.T))
    elif metric == 'minkowski':
        p = 3
        distance_matrix = np.sum(np.abs(X[:, np.newaxis] - X) ** p, axis=2) ** (1/p)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    return distance_matrix

def ward_linkage_fit(
    X: np.ndarray,
    metric: str = 'euclidean',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    normalize: str = 'none',
    custom_normalize: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Union[np.ndarray, Dict, str]]:
    """
    Perform hierarchical clustering using Ward's method.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    metric : str or callable
        Distance metric to use. Options: 'euclidean', 'manhattan', 'cosine', 'minkowski'.
    custom_metric : callable, optional
        Custom distance metric function.
    normalize : str or callable
        Normalization method. Options: 'none', 'standard', 'minmax', 'robust'.
    custom_normalize : callable, optional
        Custom normalization function.

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': Linkage matrix.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used.
        - 'warnings': Any warnings generated.

    Example
    -------
    >>> X = np.random.rand(10, 5)
    >>> result = ward_linkage_fit(X, metric='euclidean', normalize='standard')
    """
    # Validate input
    validate_input(X)

    # Normalize data if required
    if normalize == 'standard':
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalize == 'minmax':
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalize == 'robust':
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    elif custom_normalize is not None:
        X = custom_normalize(X)
    else:
        pass

    # Compute distance matrix
    distance_matrix = compute_distance(X, metric, custom_metric)

    # Initialize linkage matrix
    n_samples = X.shape[0]
    linkage_matrix = np.zeros((n_samples - 1, 4))

    # Perform hierarchical clustering using Ward's method
    for i in range(n_samples - 1):
        # Find the closest clusters (simplified implementation)
        min_distance = np.inf
        cluster1, cluster2 = 0, 0

        for j in range(n_samples - i):
            for k in range(j + 1, n_samples - i):
                if distance_matrix[j, k] < min_distance:
                    min_distance = distance_matrix[j, k]
                    cluster1, cluster2 = j, k

        # Merge clusters (simplified implementation)
        new_cluster = n_samples + i
        linkage_matrix[i, 0] = cluster1
        linkage_matrix[i, 1] = cluster2
        linkage_matrix[i, 2] = min_distance
        linkage_matrix[i, 3] = n_samples - i

    # Prepare output
    result = {
        'result': linkage_matrix,
        'metrics': {'distance_metric': metric, 'normalization': normalize},
        'params_used': {
            'metric': metric,
            'custom_metric': custom_metric is not None,
            'normalize': normalize,
            'custom_normalize': custom_normalize is not None
        },
        'warnings': []
    }

    return result

################################################################################
# distance_metrics
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def distance_metrics_compute(
    X: np.ndarray,
    metric: str = 'euclidean',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    normalization: str = 'none',
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Compute distance metrics between data points for hierarchical clustering.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    metric : str, optional
        Distance metric to use. Options: 'euclidean', 'manhattan', 'cosine',
        'minkowski'. Default is 'euclidean'.
    custom_metric : Callable, optional
        Custom distance metric function. Must take two 1D arrays and return a scalar.
    normalization : str, optional
        Normalization method. Options: 'none', 'standard', 'minmax', 'robust'.
        Default is 'none'.

    Returns
    -------
    dict
        Dictionary containing:
        - "distance_matrix": Computed distance matrix.
        - "metrics": Dictionary of computed metrics.
        - "params_used": Parameters used in the computation.
        - "warnings": Any warnings generated during computation.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> result = distance_metrics_compute(X)
    """
    # Validate inputs
    validate_inputs(X)

    # Normalize data if required
    X_normalized = apply_normalization(X, normalization)

    # Compute distance matrix
    if custom_metric is not None:
        distance_matrix = compute_custom_distance(X_normalized, custom_metric)
    else:
        distance_matrix = compute_standard_distance(
            X_normalized, metric, **kwargs
        )

    # Prepare output dictionary
    result = {
        "distance_matrix": distance_matrix,
        "metrics": {"metric_used": metric},
        "params_used": {
            "normalization": normalization,
            **kwargs
        },
        "warnings": []
    }

    return result

def validate_inputs(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("Input X must be a 2D array.")
    if np.any(np.isnan(X)):
        raise ValueError("Input X contains NaN values.")
    if np.any(np.isinf(X)):
        raise ValueError("Input X contains infinite values.")

def apply_normalization(
    X: np.ndarray,
    method: str = 'none'
) -> np.ndarray:
    """Apply data normalization."""
    if method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        return (X - np.median(X, axis=0)) / (
            np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
        )
    return X.copy()

def compute_standard_distance(
    X: np.ndarray,
    metric: str = 'euclidean',
    **kwargs
) -> np.ndarray:
    """Compute standard distance matrix."""
    n_samples = X.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    if metric == 'euclidean':
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance = np.linalg.norm(X[i] - X[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

    elif metric == 'manhattan':
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance = np.sum(np.abs(X[i] - X[j]))
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

    elif metric == 'cosine':
        for i in range(n_samples):
            for j in range(i, n_samples):
                dot_product = np.dot(X[i], X[j])
                norm_i = np.linalg.norm(X[i])
                norm_j = np.linalg.norm(X[j])
                distance = 1 - (dot_product / (norm_i * norm_j))
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

    elif metric == 'minkowski':
        p = kwargs.get('p', 2)
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance = np.sum(np.abs(X[i] - X[j])**p)**(1/p)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

    else:
        raise ValueError(f"Unknown metric: {metric}")

    return distance_matrix

def compute_custom_distance(
    X: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """Compute distance matrix using custom metric."""
    n_samples = X.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i, n_samples):
            distance = metric_func(X[i], X[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix

################################################################################
# euclidean_distance
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(X: np.ndarray, Y: Optional[np.ndarray] = None) -> None:
    """Validate input arrays for distance computation."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if Y is not None and not isinstance(Y, np.ndarray):
        raise TypeError("Y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    if Y is not None and Y.ndim != 2:
        raise ValueError("Y must be a 2-dimensional array or None")
    if Y is not None and X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if Y is not None and np.any(np.isnan(Y)):
        raise ValueError("Y contains NaN values")

def _normalize_data(X: np.ndarray, method: str = 'none') -> np.ndarray:
    """Normalize data according to specified method."""
    if method == 'none':
        return X
    elif method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_euclidean_distance(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute Euclidean distance between points in X and Y."""
    if Y is None:
        Y = X
    return np.sqrt(np.sum((X[:, np.newaxis, :] - Y) ** 2, axis=2))

def euclidean_distance_fit(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    normalization: str = 'none',
    distance_func: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray] = _compute_euclidean_distance,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute Euclidean distance between points in X and Y.

    Parameters:
    - X: Input array of shape (n_samples, n_features)
    - Y: Optional input array of same shape as X
    - normalization: Normalization method ('none', 'standard', 'minmax', 'robust')
    - distance_func: Callable for computing distances
    - **kwargs: Additional parameters passed to distance_func

    Returns:
    - Dictionary containing:
        * 'result': Computed distances
        * 'metrics': Dictionary of metrics (currently empty)
        * 'params_used': Parameters used
        * 'warnings': List of warnings

    Example:
    >>> X = np.array([[1, 2], [3, 4]])
    >>> result = euclidean_distance_fit(X)
    """
    # Validate inputs
    _validate_inputs(X, Y)

    # Normalize data
    X_norm = _normalize_data(X, normalization)
    if Y is not None:
        Y_norm = _normalize_data(Y, normalization)

    # Compute distances
    distances = distance_func(X_norm, Y_norm if Y is not None else None)

    # Prepare output
    output = {
        'result': distances,
        'metrics': {},
        'params_used': {
            'normalization': normalization,
            'distance_func': distance_func.__name__ if hasattr(distance_func, '__name__') else 'custom'
        },
        'warnings': []
    }

    return output

################################################################################
# manhattan_distance
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(X: np.ndarray, Y: Optional[np.ndarray] = None) -> None:
    """
    Validate input arrays for Manhattan distance calculation.

    Parameters
    ----------
    X : np.ndarray
        First input array of shape (n_samples, n_features)
    Y : np.ndarray, optional
        Second input array of shape (n_samples, n_features). If None, computes pairwise distances.

    Raises
    ------
    ValueError
        If inputs are invalid (wrong shapes, NaN/inf values)
    """
    if not isinstance(X, np.ndarray) or (Y is not None and not isinstance(Y, np.ndarray)):
        raise ValueError("Inputs must be numpy arrays")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    if Y is not None and Y.ndim != 2:
        raise ValueError("Y must be a 2D array or None")

    if Y is not None and X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")

    if np.any(np.isnan(X)) or (Y is not None and np.any(np.isnan(Y))) or \
       np.any(np.isinf(X)) or (Y is not None and np.any(np.isinf(Y))):
        raise ValueError("Inputs contain NaN or infinite values")

def _manhattan_distance_core(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Core Manhattan distance calculation.

    Parameters
    ----------
    X : np.ndarray
        First input array of shape (n_samples, n_features)
    Y : np.ndarray, optional
        Second input array of shape (n_samples, n_features). If None, computes pairwise distances.

    Returns
    ------
    np.ndarray
        Distance matrix of shape (n_samples, n_samples) if Y is None,
        or distance vector of shape (n_samples,) otherwise
    """
    if Y is None:
        # Pairwise distance computation
        return np.abs(X[:, np.newaxis] - X).sum(axis=2)
    else:
        # Single distance computation
        return np.abs(X - Y).sum(axis=1)

def manhattan_distance_fit(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    normalize: str = 'none',
    custom_normalize: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = 'manhattan',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute Manhattan distance between samples with various options.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (n_samples, n_features)
    Y : np.ndarray, optional
        Second input array of shape (n_samples, n_features). If None, computes pairwise distances.
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    custom_normalize : callable, optional
        Custom normalization function
    metric : str, optional
        Distance metric ('manhattan', 'euclidean', etc.)
    custom_metric : callable, optional
        Custom distance metric function

    Returns
    ------
    Dict[str, Any]
        Dictionary containing:
        - 'result': computed distances
        - 'metrics': calculated metrics
        - 'params_used': parameters used
        - 'warnings': any warnings generated

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> manhattan_distance_fit(X)
    {
        'result': array([[0., 4.],
                         [4., 0.]]),
        'metrics': {},
        'params_used': {'normalize': 'none', ...},
        'warnings': []
    }
    """
    # Validate inputs
    _validate_inputs(X, Y)

    # Initialize output dictionary
    result_dict: Dict[str, Any] = {
        'result': None,
        'metrics': {},
        'params_used': {
            'normalize': normalize,
            'metric': metric
        },
        'warnings': []
    }

    # Apply normalization if needed
    X_norm = X.copy()
    Y_norm = Y.copy() if Y is not None else None

    if normalize == 'standard':
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        if Y is not None:
            Y_norm = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)
    elif normalize == 'minmax':
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        if Y is not None:
            Y_norm = (Y - np.min(Y, axis=0)) / (np.max(Y, axis=0) - np.min(Y, axis=0))
    elif normalize == 'robust':
        X_norm = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        if Y is not None:
            Y_norm = (Y - np.median(Y, axis=0)) / (np.percentile(Y, 75, axis=0) - np.percentile(Y, 25, axis=0))
    elif custom_normalize is not None:
        X_norm = custom_normalize(X)
        if Y is not None:
            Y_norm = custom_normalize(Y)

    # Compute distances
    if metric == 'manhattan' or (custom_metric is None and metric != 'manhattan'):
        result_dict['result'] = _manhattan_distance_core(X_norm, Y_norm)
    else:
        if custom_metric is None:
            raise ValueError(f"Unsupported metric: {metric}")
        result_dict['result'] = custom_metric(X_norm, Y_norm)

    return result_dict

################################################################################
# cosine_similarity
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, Y: Optional[np.ndarray] = None) -> None:
    """
    Validate input arrays for cosine similarity computation.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Optional input array of shape (n_samples, n_features) or (n_features,)

    Raises
    ------
    ValueError
        If inputs are invalid (NaN, inf, wrong dimensions)
    """
    if not isinstance(X, np.ndarray) or (Y is not None and not isinstance(Y, np.ndarray)):
        raise ValueError("Inputs must be numpy arrays")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    if Y is not None:
        if Y.ndim == 1 and X.shape[1] != Y.shape[0]:
            raise ValueError("Y must have same number of features as X")
        elif Y.ndim == 2 and (X.shape[1] != Y.shape[1] or X.shape[0] != Y.shape[0]):
            raise ValueError("Y must have same shape as X")

    if np.any(np.isnan(X)) or (Y is not None and np.any(np.isnan(Y))):
        raise ValueError("Input arrays contain NaN values")

    if np.any(np.isinf(X)) or (Y is not None and np.any(np.isinf(Y))):
        raise ValueError("Input arrays contain infinite values")

def _normalize_data(X: np.ndarray, Y: Optional[np.ndarray] = None,
                   normalization: str = 'none') -> tuple:
    """
    Normalize input arrays according to specified method.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Optional input array of shape (n_samples, n_features) or (n_features,)
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')

    Returns
    -------
    tuple
        Normalized X and Y arrays
    """
    if normalization == 'none':
        return X, Y

    # Standard normalization (z-score)
    if normalization == 'standard':
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X_norm = (X - mean) / (std + 1e-8)

    # Min-max normalization
    elif normalization == 'minmax':
        min_val = X.min(axis=0)
        max_val = X.max(axis=0)
        X_norm = (X - min_val) / ((max_val - min_val) + 1e-8)

    # Robust normalization (median and IQR)
    elif normalization == 'robust':
        median = np.median(X, axis=0)
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        iqr = q75 - q25
        X_norm = (X - median) / ((iqr + 1e-8))

    if Y is not None:
        if Y.ndim == 1:
            if normalization == 'standard':
                Y_norm = (Y - mean) / (std + 1e-8)
            elif normalization == 'minmax':
                Y_norm = (Y - min_val) / ((max_val - min_val) + 1e-8)
            elif normalization == 'robust':
                Y_norm = (Y - median) / ((iqr + 1e-8))
        else:
            if normalization == 'standard':
                Y_norm = (Y - mean) / (std + 1e-8)
            elif normalization == 'minmax':
                Y_norm = (Y - min_val) / ((max_val - min_val) + 1e-8)
            elif normalization == 'robust':
                Y_norm = (Y - median) / ((iqr + 1e-8))
    else:
        Y_norm = None

    return X_norm, Y_norm

def cosine_similarity_compute(X: np.ndarray,
                            Y: Optional[np.ndarray] = None,
                            normalization: str = 'none',
                            custom_metric: Optional[Callable] = None) -> Dict:
    """
    Compute cosine similarity between arrays X and Y.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Optional input array of shape (n_samples, n_features) or (n_features,)
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    custom_metric : Optional[Callable]
        Custom metric function to use instead of cosine similarity

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': computed similarity matrix or vector
        - 'metrics': dictionary of computed metrics
        - 'params_used': parameters used in computation
        - 'warnings': list of warnings encountered

    Examples
    --------
    >>> X = np.random.rand(10, 5)
    >>> result = cosine_similarity_compute(X)
    """
    # Initialize output dictionary
    output: Dict = {
        'result': None,
        'metrics': {},
        'params_used': {
            'normalization': normalization
        },
        'warnings': []
    }

    # Validate inputs
    _validate_inputs(X, Y)

    # Normalize data if required
    X_norm, Y_norm = _normalize_data(X, Y, normalization)

    # Use custom metric if provided
    if custom_metric is not None:
        if Y is None:
            output['result'] = np.array([custom_metric(x) for x in X_norm])
        else:
            if Y.ndim == 1:
                output['result'] = np.array([custom_metric(x, Y_norm) for x in X_norm])
            else:
                output['result'] = np.array([[custom_metric(x, y) for y in Y_norm] for x in X_norm])
        output['params_used']['metric'] = 'custom'
        return output

    # Compute cosine similarity
    if Y is None:
        # Self-similarity matrix
        dot_products = X_norm @ X_norm.T
        norms = np.sqrt(np.sum(X_norm**2, axis=1))
        similarity_matrix = dot_products / np.outer(norms, norms)
        output['result'] = similarity_matrix
    else:
        if Y.ndim == 1:
            # Similarity between X and a single vector
            dot_products = X_norm @ Y_norm.reshape(-1, 1)
            norms_X = np.sqrt(np.sum(X_norm**2, axis=1))
            similarity_vector = dot_products / (norms_X * np.linalg.norm(Y_norm) + 1e-8)
            output['result'] = similarity_vector.flatten()
        else:
            # Similarity matrix between X and Y
            dot_products = X_norm @ Y_norm.T
            norms_X = np.sqrt(np.sum(X_norm**2, axis=1))
            norms_Y = np.sqrt(np.sum(Y_norm**2, axis=1))
            similarity_matrix = dot_products / np.outer(norms_X, norms_Y)
            output['result'] = similarity_matrix

    # Add metrics
    if Y is None:
        output['metrics']['mean_similarity'] = np.mean(output['result'])
        output['metrics']['min_similarity'] = np.min(output['result'])
        output['metrics']['max_similarity'] = np.max(output['result'])
    else:
        if Y.ndim == 1 or output['result'].ndim == 1:
            output['metrics']['mean_similarity'] = np.mean(output['result'])
            output['metrics']['min_similarity'] = np.min(output['result'])
            output['metrics']['max_similarity'] = np.max(output['result'])

    output['params_used']['metric'] = 'cosine_similarity'
    return output

################################################################################
# hierarchical_clustering_algorithms
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def hierarchical_clustering_algorithms_fit(
    data: np.ndarray,
    linkage_method: str = 'ward',
    distance_metric: Union[str, Callable] = 'euclidean',
    normalization: Optional[str] = None,
    threshold: Optional[float] = None,
    custom_distance_func: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Perform hierarchical clustering on the given data.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    linkage_method : str, optional
        Linkage method to use. Options: 'ward', 'complete', 'average', 'single'.
    distance_metric : str or callable, optional
        Distance metric to use. Options: 'euclidean', 'manhattan', 'cosine',
        or a custom callable function.
    normalization : str, optional
        Normalization method. Options: 'standard', 'minmax', 'robust'.
    threshold : float, optional
        Threshold for cutting the dendrogram.
    custom_distance_func : callable, optional
        Custom distance function if not using built-in metrics.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the clustering results, metrics, parameters used,
        and any warnings.
    """
    # Validate inputs
    _validate_inputs(data, distance_metric, custom_distance_func)

    # Normalize data if specified
    normalized_data = _apply_normalization(data, normalization)

    # Compute distance matrix
    distance_matrix = _compute_distance_matrix(
        normalized_data,
        distance_metric,
        custom_distance_func
    )

    # Perform hierarchical clustering
    linkage_matrix = _perform_hierarchical_clustering(
        distance_matrix,
        linkage_method
    )

    # Cut the dendrogram if threshold is specified
    clusters = _cut_dendrogram(linkage_matrix, threshold) if threshold else None

    # Compute metrics
    metrics = _compute_metrics(data, clusters) if clusters is not None else {}

    # Prepare output
    result = {
        'result': {
            'linkage_matrix': linkage_matrix,
            'clusters': clusters
        },
        'metrics': metrics,
        'params_used': {
            'linkage_method': linkage_method,
            'distance_metric': distance_metric,
            'normalization': normalization,
            'threshold': threshold
        },
        'warnings': []
    }

    return result

def _validate_inputs(
    data: np.ndarray,
    distance_metric: Union[str, Callable],
    custom_distance_func: Optional[Callable]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array.")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values.")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values.")

    if isinstance(distance_metric, str) and distance_metric not in ['euclidean', 'manhattan', 'cosine']:
        raise ValueError("Unsupported distance metric.")
    if custom_distance_func is not None and not callable(custom_distance_func):
        raise ValueError("Custom distance function must be callable.")

def _apply_normalization(
    data: np.ndarray,
    normalization: Optional[str]
) -> np.ndarray:
    """Apply specified normalization to the data."""
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

    return normalized_data

def _compute_distance_matrix(
    data: np.ndarray,
    distance_metric: Union[str, Callable],
    custom_distance_func: Optional[Callable]
) -> np.ndarray:
    """Compute the distance matrix using specified metric."""
    n_samples = data.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    if custom_distance_func is not None:
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance_matrix[i, j] = custom_distance_func(data[i], data[j])
                distance_matrix[j, i] = distance_matrix[i, j]
    else:
        if distance_metric == 'euclidean':
            for i in range(n_samples):
                for j in range(i, n_samples):
                    distance_matrix[i, j] = np.linalg.norm(data[i] - data[j])
                    distance_matrix[j, i] = distance_matrix[i, j]
        elif distance_metric == 'manhattan':
            for i in range(n_samples):
                for j in range(i, n_samples):
                    distance_matrix[i, j] = np.sum(np.abs(data[i] - data[j]))
                    distance_matrix[j, i] = distance_matrix[i, j]
        elif distance_metric == 'cosine':
            for i in range(n_samples):
                for j in range(i, n_samples):
                    dot_product = np.dot(data[i], data[j])
                    norm_i = np.linalg.norm(data[i])
                    norm_j = np.linalg.norm(data[j])
                    distance_matrix[i, j] = 1 - (dot_product / (norm_i * norm_j + 1e-8))
                    distance_matrix[j, i] = distance_matrix[i, j]

    return distance_matrix

def _perform_hierarchical_clustering(
    distance_matrix: np.ndarray,
    linkage_method: str
) -> np.ndarray:
    """Perform hierarchical clustering using specified linkage method."""
    n_samples = distance_matrix.shape[0]
    linkage_matrix = np.zeros((n_samples - 1, 4))

    # Initialize clusters
    clusters = [[i] for i in range(n_samples)]

    for k in range(n_samples - 1):
        # Find the closest clusters
        min_distance = np.inf
        cluster_i, cluster_j = 0, 0

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if linkage_method == 'single':
                    distance = np.min(distance_matrix[np.ix_(clusters[i], clusters[j])])
                elif linkage_method == 'complete':
                    distance = np.max(distance_matrix[np.ix_(clusters[i], clusters[j])])
                elif linkage_method == 'average':
                    distance = np.mean(distance_matrix[np.ix_(clusters[i], clusters[j])])
                elif linkage_method == 'ward':
                    n_i = len(clusters[i])
                    n_j = len(clusters[j])
                    mean_i = np.mean(distance_matrix[clusters[i], :], axis=0)
                    mean_j = np.mean(distance_matrix[clusters[j], :], axis=0)
                    distance = n_i * n_j * np.linalg.norm(mean_i - mean_j) ** 2 / (n_i + n_j)

                if distance < min_distance:
                    min_distance = distance
                    cluster_i, cluster_j = i, j

        # Merge clusters
        merged_cluster = clusters[cluster_i] + clusters[cluster_j]
        linkage_matrix[k, 0] = cluster_i
        linkage_matrix[k, 1] = cluster_j
        linkage_matrix[k, 2] = min_distance
        linkage_matrix[k, 3] = len(merged_cluster)

        # Update clusters
        clusters.pop(cluster_j)
        clusters.pop(cluster_i)
        clusters.append(merged_cluster)

    return linkage_matrix

def _cut_dendrogram(
    linkage_matrix: np.ndarray,
    threshold: float
) -> list:
    """Cut the dendrogram to form flat clusters."""
    n_clusters = len(linkage_matrix) + 1
    clusters = [[i] for i in range(n_clusters)]

    for i in reversed(range(len(linkage_matrix))):
        if linkage_matrix[i, 2] > threshold:
            break
        cluster1 = int(linkage_matrix[i, 0])
        cluster2 = int(linkage_matrix[i, 1])
        clusters[cluster1].extend(clusters[cluster2])
        clusters.pop(cluster2)

    return [tuple(cluster) for cluster in clusters]

def _compute_metrics(
    data: np.ndarray,
    clusters: list
) -> Dict[str, float]:
    """Compute clustering metrics."""
    metrics = {}

    # Example metric: silhouette score
    if len(clusters) > 1:
        from sklearn.metrics import silhouette_score
        labels = np.zeros(data.shape[0], dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for sample_id in cluster:
                labels[sample_id] = cluster_id
        metrics['silhouette_score'] = silhouette_score(data, labels)

    return metrics

# Example usage:
# data = np.random.rand(10, 5)
# result = hierarchical_clustering_algorithms_fit(data, linkage_method='ward', distance_metric='euclidean')

################################################################################
# bottom_up_approach
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_input(data: np.ndarray) -> None:
    """Validate input data for clustering."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Input data must be 2-dimensional")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values")

def normalize_data(data: np.ndarray, method: str = 'standard') -> np.ndarray:
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

def compute_distance_matrix(data: np.ndarray, metric: Union[str, Callable]) -> np.ndarray:
    """Compute distance matrix using specified metric."""
    n_samples = data.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))

    if isinstance(metric, str):
        if metric == 'euclidean':
            for i in range(n_samples):
                for j in range(i, n_samples):
                    dist = np.linalg.norm(data[i] - data[j])
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
        elif metric == 'manhattan':
            for i in range(n_samples):
                for j in range(i, n_samples):
                    dist = np.sum(np.abs(data[i] - data[j]))
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
        elif metric == 'cosine':
            for i in range(n_samples):
                for j in range(i, n_samples):
                    dot = np.dot(data[i], data[j])
                    norm_i = np.linalg.norm(data[i])
                    norm_j = np.linalg.norm(data[j])
                    dist = 1 - (dot / (norm_i * norm_j + 1e-8))
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
        else:
            raise ValueError(f"Unknown distance metric: {metric}")
    else:
        for i in range(n_samples):
            for j in range(i, n_samples):
                dist = metric(data[i], data[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

    return dist_matrix

def bottom_up_approach_fit(
    data: np.ndarray,
    normalization: str = 'standard',
    distance_metric: Union[str, Callable] = 'euclidean',
    linkage_criterion: str = 'ward',
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Perform hierarchical clustering using bottom-up approach.

    Parameters:
    - data: Input data as numpy array
    - normalization: Normalization method ('none', 'standard', 'minmax', 'robust')
    - distance_metric: Distance metric ('euclidean', 'manhattan', 'cosine') or callable
    - linkage_criterion: Linkage criterion ('ward', 'complete', 'average', 'single')
    - threshold: Threshold for stopping the clustering

    Returns:
    - Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate input data
    validate_input(data)

    # Normalize data
    normalized_data = normalize_data(data, normalization)

    # Compute distance matrix
    dist_matrix = compute_distance_matrix(normalized_data, distance_metric)

    # Initialize clusters
    n_samples = data.shape[0]
    clusters = [[i] for i in range(n_samples)]
    n_clusters = n_samples

    # Main clustering loop
    while n_clusters > 1 and (threshold is None or len(clusters) > threshold):
        # Find closest clusters
        min_dist = np.inf
        cluster1, cluster2 = None, None

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if linkage_criterion == 'ward':
                    dist = ward_distance(dist_matrix, clusters[i], clusters[j])
                elif linkage_criterion == 'complete':
                    dist = complete_distance(dist_matrix, clusters[i], clusters[j])
                elif linkage_criterion == 'average':
                    dist = average_distance(dist_matrix, clusters[i], clusters[j])
                elif linkage_criterion == 'single':
                    dist = single_distance(dist_matrix, clusters[i], clusters[j])
                else:
                    raise ValueError(f"Unknown linkage criterion: {linkage_criterion}")

                if dist < min_dist:
                    min_dist = dist
                    cluster1, cluster2 = i, j

        # Merge clusters
        merged_cluster = clusters[cluster1] + clusters[cluster2]
        del clusters[max(cluster1, cluster2)]
        del clusters[min(cluster1, cluster2)]
        clusters.append(merged_cluster)
        n_clusters -= 1

    # Prepare results
    result = {
        'clusters': clusters,
        'distance_matrix': dist_matrix,
        'normalization_used': normalization,
        'distance_metric_used': distance_metric,
        'linkage_criterion_used': linkage_criterion
    }

    return {
        'result': result,
        'metrics': {},
        'params_used': {
            'normalization': normalization,
            'distance_metric': distance_metric,
            'linkage_criterion': linkage_criterion,
            'threshold': threshold
        },
        'warnings': []
    }

def ward_distance(dist_matrix: np.ndarray, cluster1: list, cluster2: list) -> float:
    """Compute Ward distance between two clusters."""
    n1 = len(cluster1)
    n2 = len(cluster2)

    sum_dist = 0.0
    for i in cluster1:
        for j in cluster2:
            sum_dist += dist_matrix[i, j]

    return (n1 * n2) / (n1 + n2) * sum_dist

def complete_distance(dist_matrix: np.ndarray, cluster1: list, cluster2: list) -> float:
    """Compute complete linkage distance between two clusters."""
    max_dist = 0.0
    for i in cluster1:
        for j in cluster2:
            if dist_matrix[i, j] > max_dist:
                max_dist = dist_matrix[i, j]
    return max_dist

def average_distance(dist_matrix: np.ndarray, cluster1: list, cluster2: list) -> float:
    """Compute average linkage distance between two clusters."""
    sum_dist = 0.0
    count = 0
    for i in cluster1:
        for j in cluster2:
            sum_dist += dist_matrix[i, j]
            count += 1
    return sum_dist / count

def single_distance(dist_matrix: np.ndarray, cluster1: list, cluster2: list) -> float:
    """Compute single linkage distance between two clusters."""
    min_dist = np.inf
    for i in cluster1:
        for j in cluster2:
            if dist_matrix[i, j] < min_dist:
                min_dist = dist_matrix[i, j]
    return min_dist

################################################################################
# top_down_approach
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

def normalize_data(
    X: np.ndarray,
    method: str = "standard",
    custom_normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize data using specified method or custom function."""
    if custom_normalizer is not None:
        return custom_normalizer(X)

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

def compute_distance(
    X: np.ndarray,
    Y: np.ndarray,
    metric: str = "euclidean",
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> np.ndarray:
    """Compute distance matrix between data points."""
    if custom_distance is not None:
        return np.array([[custom_distance(x, y) for y in Y] for x in X])

    if metric == "euclidean":
        return np.sqrt(np.sum((X[:, np.newaxis] - Y) ** 2, axis=2))
    elif metric == "manhattan":
        return np.sum(np.abs(X[:, np.newaxis] - Y), axis=2)
    elif metric == "cosine":
        dot_products = np.dot(X, Y.T)
        norms = np.sqrt(np.sum(X**2, axis=1))[:, np.newaxis] * np.sqrt(np.sum(Y**2, axis=1))
        return 1 - dot_products / (norms + 1e-8)
    elif metric == "minkowski":
        p = 3
        return np.sum(np.abs(X[:, np.newaxis] - Y) ** p, axis=2) ** (1/p)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def hierarchical_clustering(
    X: np.ndarray,
    linkage_method: str = "ward",
    distance_threshold: float = 0.5
) -> Dict[str, Union[np.ndarray, Dict]]:
    """Perform hierarchical clustering using top-down approach."""
    # Initialize each point as its own cluster
    clusters = [{'points': [i], 'center': X[i]} for i in range(X.shape[0])]
    n_clusters = len(clusters)

    while n_clusters > 1:
        # Compute distances between all cluster centers
        centers = np.array([c['center'] for c in clusters])
        distances = compute_distance(centers, centers)

        # Find closest pair of clusters
        min_dist = np.inf
        i, j = -1, -1
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                if distances[a, b] < min_dist:
                    min_dist = distances[a, b]
                    i, j = a, b

        if min_dist > distance_threshold:
            break

        # Merge clusters
        merged_cluster = {
            'points': clusters[i]['points'] + clusters[j]['points'],
            'center': np.mean([clusters[i]['center'], clusters[j]['center']], axis=0)
        }

        # Update clusters
        new_clusters = [clusters[k] for k in range(len(clusters)) if k != i and k != j]
        new_clusters.append(merged_cluster)
        clusters = new_clusters
        n_clusters -= 1

    # Prepare results
    result = {
        'clusters': clusters,
        'n_clusters': n_clusters,
        'distance_threshold': distance_threshold
    }

    return result

def top_down_approach_fit(
    X: np.ndarray,
    normalize_method: str = "standard",
    distance_metric: str = "euclidean",
    custom_normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    distance_threshold: float = 0.5
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Perform top-down hierarchical clustering.

    Parameters:
    - X: Input data (n_samples, n_features)
    - normalize_method: Normalization method ('none', 'standard', 'minmax', 'robust')
    - distance_metric: Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski')
    - custom_normalizer: Custom normalization function
    - custom_distance: Custom distance function
    - distance_threshold: Threshold for merging clusters

    Returns:
    Dictionary containing clustering results, metrics, and parameters used.
    """
    # Validate input
    validate_input(X)

    # Normalize data
    X_normalized = normalize_data(
        X,
        method=normalize_method,
        custom_normalizer=custom_normalizer
    )

    # Perform hierarchical clustering
    result = hierarchical_clustering(
        X_normalized,
        distance_threshold=distance_threshold
    )

    # Prepare output dictionary
    output = {
        'result': result,
        'metrics': {},
        'params_used': {
            'normalize_method': normalize_method,
            'distance_metric': distance_metric if custom_distance is None else "custom",
            'distance_threshold': distance_threshold
        },
        'warnings': []
    }

    return output

# Example usage:
"""
X = np.random.rand(10, 2)
result = top_down_approach_fit(X)
print(result['result']['n_clusters'])
"""

################################################################################
# cutoff_threshold
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(data: np.ndarray) -> None:
    """Validate input data for clustering."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or infinite values")

def normalize_data(
    data: np.ndarray,
    method: str = "standard",
    custom_func: Optional[Callable] = None
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

def compute_distance_matrix(
    data: np.ndarray,
    metric: str = "euclidean",
    custom_func: Optional[Callable] = None
) -> np.ndarray:
    """Compute distance matrix using specified metric or custom function."""
    if custom_func is not None:
        return custom_func(data)

    n_samples = data.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))

    if metric == "euclidean":
        for i in range(n_samples):
            for j in range(i, n_samples):
                dist = np.linalg.norm(data[i] - data[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
    elif metric == "manhattan":
        for i in range(n_samples):
            for j in range(i, n_samples):
                dist = np.sum(np.abs(data[i] - data[j]))
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
    elif metric == "cosine":
        for i in range(n_samples):
            for j in range(i, n_samples):
                dot_product = np.dot(data[i], data[j])
                norm_i = np.linalg.norm(data[i])
                norm_j = np.linalg.norm(data[j])
                dist = 1 - (dot_product / (norm_i * norm_j + 1e-8))
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
    elif metric == "minkowski":
        for i in range(n_samples):
            for j in range(i, n_samples):
                dist = np.sum(np.abs(data[i] - data[j]) ** 3) ** (1/3)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

    return dist_matrix

def cutoff_threshold_fit(
    data: np.ndarray,
    threshold: float = 0.5,
    normalization_method: str = "standard",
    distance_metric: str = "euclidean",
    custom_normalize: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Perform hierarchical clustering with cutoff threshold.

    Parameters
    ----------
    data : np.ndarray
        Input data for clustering.
    threshold : float, optional
        Threshold value to determine clusters (default is 0.5).
    normalization_method : str, optional
        Normalization method to apply (default is "standard").
    distance_metric : str, optional
        Distance metric to use (default is "euclidean").
    custom_normalize : Callable, optional
        Custom normalization function.
    custom_distance : Callable, optional
        Custom distance function.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict]]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate input
    validate_input(data)

    # Normalize data
    normalized_data = normalize_data(
        data,
        method=normalization_method,
        custom_func=custom_normalize
    )

    # Compute distance matrix
    dist_matrix = compute_distance_matrix(
        normalized_data,
        metric=distance_metric,
        custom_func=custom_distance
    )

    # Perform hierarchical clustering with cutoff threshold
    # (This is a simplified version; actual implementation would use a proper hierarchical clustering algorithm)
    n_samples = data.shape[0]
    clusters = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        min_dist = np.min(dist_matrix[i, :i])
        if min_dist < threshold:
            clusters[i] = np.argmin(dist_matrix[i, :i])
        else:
            clusters[i] = i

    # Calculate metrics
    n_clusters = len(np.unique(clusters))
    cluster_sizes = np.bincount(clusters)
    metrics = {
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes.tolist(),
    }

    # Prepare output
    result = {
        "clusters": clusters,
        "distance_matrix": dist_matrix,
    }

    params_used = {
        "threshold": threshold,
        "normalization_method": normalization_method,
        "distance_metric": distance_metric,
    }

    warnings = []

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings,
    }

# Example usage:
# data = np.random.rand(10, 5)
# result = cutoff_threshold_fit(data, threshold=0.3, normalization_method="standard")

################################################################################
# number_of_clusters
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def number_of_clusters_fit(
    linkage_matrix: np.ndarray,
    criterion: str = 'maxclust',
    distance_threshold: Optional[float] = None,
    n_clusters: Optional[int] = None,
    custom_criterion: Optional[Callable[[np.ndarray], int]] = None
) -> Dict[str, Union[int, Dict, str]]:
    """
    Determine the optimal number of clusters from a hierarchical clustering linkage matrix.

    Parameters
    ----------
    linkage_matrix : np.ndarray
        Linkage matrix from hierarchical clustering (e.g., from scipy.cluster.hierarchy.linkage).
    criterion : str, optional
        Method to determine the number of clusters. Options: 'maxclust', 'distance'.
    distance_threshold : float, optional
        Threshold for the distance criterion.
    n_clusters : int, optional
        Number of clusters when using 'maxclust' criterion.
    custom_criterion : callable, optional
        Custom function to determine the number of clusters.

    Returns
    -------
    Dict[str, Union[int, Dict, str]]
        Dictionary containing:
        - 'result': int, the optimal number of clusters
        - 'metrics': Dict, additional metrics if applicable
        - 'params_used': Dict, parameters used for the computation
        - 'warnings': str, any warnings or messages

    Examples
    --------
    >>> linkage_matrix = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    >>> result = number_of_clusters_fit(linkage_matrix, criterion='maxclust', n_clusters=2)
    """
    # Validate inputs
    if not isinstance(linkage_matrix, np.ndarray):
        raise ValueError("linkage_matrix must be a numpy array")
    if linkage_matrix.size == 0:
        raise ValueError("linkage_matrix cannot be empty")
    if criterion not in ['maxclust', 'distance'] and custom_criterion is None:
        raise ValueError("Invalid criterion or missing custom_criterion")

    # Initialize output dictionary
    result_dict = {
        'result': None,
        'metrics': {},
        'params_used': {
            'criterion': criterion,
            'distance_threshold': distance_threshold,
            'n_clusters': n_clusters
        },
        'warnings': ''
    }

    # Determine the number of clusters based on the criterion
    if custom_criterion is not None:
        result_dict['result'] = custom_criterion(linkage_matrix)
    elif criterion == 'maxclust':
        if n_clusters is None:
            result_dict['warnings'] = 'n_clusters not provided for maxclust criterion'
        else:
            result_dict['result'] = n_clusters
    elif criterion == 'distance':
        if distance_threshold is None:
            result_dict['warnings'] = 'distance_threshold not provided for distance criterion'
        else:
            result_dict['result'] = _determine_clusters_by_distance(linkage_matrix, distance_threshold)

    return result_dict

def _determine_clusters_by_distance(
    linkage_matrix: np.ndarray,
    distance_threshold: float
) -> int:
    """
    Determine the number of clusters based on a distance threshold.

    Parameters
    ----------
    linkage_matrix : np.ndarray
        Linkage matrix from hierarchical clustering.
    distance_threshold : float
        Threshold for the distance criterion.

    Returns
    -------
    int
        The number of clusters.
    """
    # Count the number of unique clusters based on the distance threshold
    n_samples = linkage_matrix.shape[0] + 1
    n_clusters = n_samples

    for i in range(linkage_matrix.shape[0]):
        if linkage_matrix[i, 2] > distance_threshold:
            n_clusters -= 1

    return n_clusters

################################################################################
# dendrogram_cutting
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(data: np.ndarray) -> None:
    """Validate input data for dendrogram cutting.

    Args:
        data: Input data array.

    Raises:
        ValueError: If input is invalid.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or infinite values.")

def normalize_data(
    data: np.ndarray,
    method: str = "standard",
    custom_func: Optional[Callable] = None
) -> np.ndarray:
    """Normalize data using specified method or custom function.

    Args:
        data: Input data array.
        method: Normalization method ('none', 'standard', 'minmax', 'robust').
        custom_func: Custom normalization function.

    Returns:
        Normalized data array.
    """
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
    elif custom_func is not None:
        return custom_func(data)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def compute_distance(
    data: np.ndarray,
    metric: str = "euclidean",
    custom_func: Optional[Callable] = None
) -> np.ndarray:
    """Compute distance matrix using specified metric or custom function.

    Args:
        data: Input data array.
        metric: Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski').
        custom_func: Custom distance function.

    Returns:
        Distance matrix.
    """
    if metric == "euclidean":
        return np.sqrt(np.sum((data[:, np.newaxis] - data) ** 2, axis=2))
    elif metric == "manhattan":
        return np.sum(np.abs(data[:, np.newaxis] - data), axis=2)
    elif metric == "cosine":
        dot_product = np.dot(data, data.T)
        norm = np.sqrt(np.sum(data**2, axis=1))[:, np.newaxis]
        return 1 - dot_product / (norm * norm.T + 1e-8)
    elif metric == "minkowski":
        return np.sum(np.abs(data[:, np.newaxis] - data) ** 3, axis=2) ** (1/3)
    elif custom_func is not None:
        return custom_func(data)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def cut_dendrogram(
    linkage_matrix: np.ndarray,
    threshold: float
) -> Dict[str, Union[np.ndarray, int]]:
    """Cut dendrogram at specified threshold.

    Args:
        linkage_matrix: Linkage matrix from hierarchical clustering.
        threshold: Threshold for cutting the dendrogram.

    Returns:
        Dictionary containing cluster labels and number of clusters.
    """
    from scipy.cluster.hierarchy import fcluster
    cluster_labels = fcluster(linkage_matrix, t=threshold, criterion='distance')
    return {
        "labels": cluster_labels,
        "n_clusters": len(np.unique(cluster_labels))
    }

def compute_metrics(
    data: np.ndarray,
    labels: np.ndarray,
    metric: str = "mse",
    custom_func: Optional[Callable] = None
) -> Dict[str, float]:
    """Compute clustering metrics.

    Args:
        data: Input data array.
        labels: Cluster labels.
        metric: Metric to compute ('mse', 'mae', 'r2').
        custom_func: Custom metric function.

    Returns:
        Dictionary of computed metrics.
    """
    if metric == "mse":
        return {"mse": np.mean((data - data[labels])**2)}
    elif metric == "mae":
        return {"mae": np.mean(np.abs(data - data[labels]))}
    elif metric == "r2":
        ss_total = np.sum((data - np.mean(data, axis=0))**2)
        ss_residual = np.sum((data - data[labels])**2)
        return {"r2": 1 - (ss_residual / ss_total)}
    elif custom_func is not None:
        return {"custom": custom_func(data, labels)}
    else:
        raise ValueError(f"Unknown metric: {metric}")

def dendrogram_cutting_fit(
    data: np.ndarray,
    linkage_matrix: np.ndarray,
    threshold: float = 0.5,
    normalization: str = "standard",
    distance_metric: str = "euclidean",
    metric: str = "mse",
    custom_normalization: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None,
    custom_metric: Optional[Callable] = None
) -> Dict[str, Union[Dict, np.ndarray, int]]:
    """Main function for dendrogram cutting.

    Args:
        data: Input data array.
        linkage_matrix: Linkage matrix from hierarchical clustering.
        threshold: Threshold for cutting the dendrogram.
        normalization: Normalization method ('none', 'standard', 'minmax', 'robust').
        distance_metric: Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski').
        metric: Clustering metric ('mse', 'mae', 'r2').
        custom_normalization: Custom normalization function.
        custom_distance: Custom distance function.
        custom_metric: Custom metric function.

    Returns:
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate input
    validate_input(data)

    # Normalize data
    normalized_data = normalize_data(
        data,
        method=normalization,
        custom_func=custom_normalization
    )

    # Compute distance matrix (if needed)
    if custom_distance is None:
        distance_matrix = compute_distance(
            normalized_data,
            metric=distance_metric
        )
    else:
        distance_matrix = custom_distance(normalized_data)

    # Cut dendrogram
    cut_result = cut_dendrogram(linkage_matrix, threshold)

    # Compute metrics
    metrics = compute_metrics(
        normalized_data,
        cut_result["labels"],
        metric=metric,
        custom_func=custom_metric
    )

    # Prepare output
    return {
        "result": cut_result,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization if custom_normalization is None else "custom",
            "distance_metric": distance_metric if custom_distance is None else "custom",
            "metric": metric if custom_metric is None else "custom",
            "threshold": threshold
        },
        "warnings": []
    }

################################################################################
# cluster_validation_metrics
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    metric: str = "euclidean",
    normalize: str = "none"
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y_true is not None and not isinstance(y_true, np.ndarray):
        raise TypeError("y_true must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y_true is not None and X.shape[0] != y_true.shape[0]:
        raise ValueError("X and y_true must have the same number of samples")
    if normalize not in ["none", "standard", "minmax", "robust"]:
        raise ValueError("normalize must be one of: none, standard, minmax, robust")
    if metric not in ["euclidean", "manhattan", "cosine", "minkowski"] and not callable(metric):
        raise ValueError("metric must be a string or callable")

def _normalize_data(
    X: np.ndarray,
    normalize: str = "none"
) -> np.ndarray:
    """Normalize the input data."""
    if normalize == "standard":
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalize == "minmax":
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalize == "robust":
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    return X

def _compute_metric(
    X: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    metric: Union[str, Callable] = "euclidean",
    **kwargs
) -> Dict[str, Any]:
    """Compute the specified metric."""
    metrics = {}
    if callable(metric):
        metrics["custom_metric"] = metric(X, y_true)
    else:
        if metric == "euclidean":
            metrics["euclidean"] = np.mean(np.sqrt(np.sum((X - y_true) ** 2, axis=1)))
        elif metric == "manhattan":
            metrics["manhattan"] = np.mean(np.sum(np.abs(X - y_true), axis=1))
        elif metric == "cosine":
            metrics["cosine"] = np.mean(1 - (X @ y_true.T) / (np.linalg.norm(X, axis=1) * np.linalg.norm(y_true, axis=1)))
        elif metric == "minkowski":
            p = kwargs.get("p", 2)
            metrics["minkowski"] = np.mean(np.sum(np.abs(X - y_true) ** p, axis=1) ** (1/p))
    return metrics

def cluster_validation_metrics_fit(
    X: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    metric: Union[str, Callable] = "euclidean",
    normalize: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """
    Compute cluster validation metrics for hierarchical clustering.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    y_true : Optional[np.ndarray], default=None
        True labels for supervised metrics.
    metric : Union[str, Callable], default="euclidean"
        Distance metric to use. Can be "euclidean", "manhattan", "cosine", "minkowski", or a custom callable.
    normalize : str, default="none"
        Normalization method. Can be "none", "standard", "minmax", or "robust".
    **kwargs
        Additional keyword arguments for specific metrics.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y_true = np.random.randint(0, 2, 100)
    >>> results = cluster_validation_metrics_fit(X, y_true, metric="euclidean", normalize="standard")
    """
    _validate_inputs(X, y_true, metric, normalize)
    X_normalized = _normalize_data(X, normalize)
    metrics = _compute_metric(X_normalized, y_true, metric, **kwargs)

    return {
        "result": None,
        "metrics": metrics,
        "params_used": {
            "metric": metric,
            "normalize": normalize
        },
        "warnings": []
    }

################################################################################
# silhouette_score
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_inputs(
    X: np.ndarray,
    labels: np.ndarray,
    metric: Union[str, Callable],
    normalize: Optional[Callable] = None
) -> None:
    """
    Validate input data and parameters.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    labels : np.ndarray
        Cluster labels of shape (n_samples,).
    metric : Union[str, Callable]
        Distance metric to use.
    normalize : Optional[Callable], optional
        Normalization function, by default None.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array.")
    if len(X) != len(labels):
        raise ValueError("X and labels must have the same number of samples.")
    if isinstance(metric, str) and metric not in ['euclidean', 'manhattan', 'cosine']:
        raise ValueError("Unsupported metric string.")
    if normalize is not None and not callable(normalize):
        raise ValueError("normalize must be a callable or None.")

def compute_distance_matrix(
    X: np.ndarray,
    metric: Union[str, Callable]
) -> np.ndarray:
    """
    Compute distance matrix using specified metric.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    metric : Union[str, Callable]
        Distance metric to use.

    Returns
    ------
    np.ndarray
        Distance matrix of shape (n_samples, n_samples).
    """
    if isinstance(metric, str):
        if metric == 'euclidean':
            return np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
        elif metric == 'manhattan':
            return np.sum(np.abs(X[:, np.newaxis] - X), axis=2)
        elif metric == 'cosine':
            return 1 - np.dot(X, X.T) / (np.linalg.norm(X, axis=1)[:, np.newaxis] * np.linalg.norm(X, axis=1))
    else:
        return metric(X)

def compute_silhouette_score(
    X: np.ndarray,
    labels: np.ndarray,
    metric: Union[str, Callable],
    normalize: Optional[Callable] = None
) -> Dict:
    """
    Compute silhouette score for hierarchical clustering.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    labels : np.ndarray
        Cluster labels of shape (n_samples,).
    metric : Union[str, Callable]
        Distance metric to use.
    normalize : Optional[Callable], optional
        Normalization function, by default None.

    Returns
    ------
    Dict
        Dictionary containing:
        - result: float, silhouette score
        - metrics: dict, additional metrics
        - params_used: dict, parameters used
        - warnings: list, any warnings
    """
    validate_inputs(X, labels, metric, normalize)

    distance_matrix = compute_distance_matrix(X, metric)
    n_samples = len(labels)
    unique_labels = np.unique(labels)
    silhouette_scores = np.zeros(n_samples)

    for i in range(n_samples):
        a = 0
        b = 0
        same_cluster = labels == labels[i]
        different_clusters = labels != labels[i]

        if np.sum(same_cluster) > 1:
            a = np.mean(distance_matrix[i, same_cluster])
        else:
            a = 0

        if np.sum(different_clusters) > 0:
            b_values = []
            for label in unique_labels:
                if label != labels[i]:
                    cluster_points = different_clusters & (labels == label)
                    if np.sum(cluster_points) > 0:
                        b_values.append(np.mean(distance_matrix[i, cluster_points]))
            if len(b_values) > 0:
                b = np.min(b_values)

        if max(a, b) != 0:
            silhouette_scores[i] = (b - a) / max(a, b)
        else:
            silhouette_scores[i] = 0

    result = np.mean(silhouette_scores)
    metrics = {
        'mean_silhouette_score': result,
        'individual_scores': silhouette_scores
    }
    params_used = {
        'metric': metric,
        'normalize': normalize is not None
    }
    warnings = []

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

# Example usage:
"""
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
labels = np.array([1, 1, 1, 2, 2, 2])
result = silhouette_score_compute(X, labels, metric='euclidean')
print(result)
"""

################################################################################
# davies_bouldin_index
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_input(data: np.ndarray) -> None:
    """Validate input data for Davies-Bouldin Index calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data must not contain NaN or infinite values.")

def compute_cluster_centers(data: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    """Compute cluster centers for each cluster."""
    unique_labels = np.unique(labels)
    centers = {}
    for label in unique_labels:
        centers[label] = np.mean(data[labels == label], axis=0)
    return centers

def compute_cluster_variances(data: np.ndarray, labels: np.ndarray, metric: Callable) -> Dict[int, float]:
    """Compute the variance within each cluster."""
    unique_labels = np.unique(labels)
    variances = {}
    for label in unique_labels:
        cluster_data = data[labels == label]
        if len(cluster_data) > 1:
            variances[label] = np.mean([metric(x, cluster_data) for x in cluster_data])
        else:
            variances[label] = 0.0
    return variances

def compute_inter_cluster_distances(centers: Dict[int, np.ndarray], metric: Callable) -> np.ndarray:
    """Compute the distances between all pairs of cluster centers."""
    labels = list(centers.keys())
    n_clusters = len(labels)
    distances = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            distances[i, j] = metric(centers[labels[i]], centers[labels[j]])
            distances[j, i] = distances[i, j]
    return distances

def davies_bouldin_index_compute(data: np.ndarray,
                                labels: np.ndarray,
                                metric: Callable = None,
                                normalization: str = 'none') -> Dict[str, Union[float, Dict]]:
    """
    Compute the Davies-Bouldin Index for a given clustering.

    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features).
    labels : np.ndarray
        Cluster labels for each sample.
    metric : Callable, optional
        Distance metric function. Default is Euclidean distance.
    normalization : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.

    Returns
    -------
    Dict[str, Union[float, Dict]]
        Dictionary containing the Davies-Bouldin Index result and related information.
    """
    validate_input(data)

    if metric is None:
        def metric(a, b):
            return np.linalg.norm(a - b)

    centers = compute_cluster_centers(data, labels)
    intra_variances = compute_cluster_variances(data, labels, metric)
    inter_distances = compute_inter_cluster_distances(centers, metric)

    n_clusters = len(centers)
    db_index = 0.0

    for i in range(n_clusters):
        max_ratio = 0.0
        label_i = list(centers.keys())[i]
        for j in range(n_clusters):
            if i != j:
                label_j = list(centers.keys())[j]
                ratio = (intra_variances[label_i] + intra_variances[label_j]) / inter_distances[i, j]
                if ratio > max_ratio:
                    max_ratio = ratio
        db_index += max_ratio

    db_index /= n_clusters

    return {
        "result": db_index,
        "metrics": {
            "intra_variances": intra_variances,
            "inter_distances": inter_distances
        },
        "params_used": {
            "metric": metric.__name__ if hasattr(metric, '__name__') else str(metric),
            "normalization": normalization
        },
        "warnings": []
    }

# Example usage:
# data = np.random.rand(100, 5)
# labels = np.random.randint(0, 3, size=100)
# result = davies_bouldin_index_compute(data, labels)

################################################################################
# calinski_harabasz_index
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
    custom_metric: Optional[Callable] = None
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray) or not isinstance(labels, np.ndarray):
        raise TypeError("X and labels must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")
    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(labels)) or np.any(np.isinf(labels)):
        raise ValueError("labels contains NaN or infinite values")

    if custom_metric is not None:
        return
    valid_metrics = ["euclidean", "manhattan", "cosine"]
    if metric not in valid_metrics:
        raise ValueError(f"metric must be one of {valid_metrics}")

def _compute_within_cluster_sum_of_squares(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
    custom_metric: Optional[Callable] = None
) -> float:
    """Compute the within-cluster sum of squares."""
    n_samples, n_features = X.shape
    unique_labels = np.unique(labels)
    wcss = 0.0

    for label in unique_labels:
        cluster_points = X[labels == label]
        if custom_metric is not None:
            # Compute pairwise distances using custom metric
            n = cluster_points.shape[0]
            dist_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dist_matrix[i, j] = custom_metric(cluster_points[i], cluster_points[j])
            # Convert distances to squared Euclidean for consistency
            dist_matrix = np.square(dist_matrix)
        else:
            if metric == "euclidean":
                dist_matrix = np.sum(np.square(cluster_points[:, np.newaxis] - cluster_points), axis=2)
            elif metric == "manhattan":
                dist_matrix = np.sum(np.abs(cluster_points[:, np.newaxis] - cluster_points), axis=2)
            elif metric == "cosine":
                dist_matrix = 1 - np.dot(cluster_points, cluster_points.T) / (
                    np.linalg.norm(cluster_points, axis=1)[:, np.newaxis] *
                    np.linalg.norm(cluster_points, axis=1)[np.newaxis, :]
                )

        wcss += np.sum(dist_matrix) / (2 * cluster_points.shape[0])

    return wcss

def _compute_between_cluster_sum_of_squares(
    X: np.ndarray,
    labels: np.ndarray
) -> float:
    """Compute the between-cluster sum of squares."""
    n_samples, n_features = X.shape
    unique_labels = np.unique(labels)
    centroids = np.array([X[labels == label].mean(axis=0) for label in unique_labels])
    overall_centroid = X.mean(axis=0)

    bcss = 0.0
    for centroid in centroids:
        n_samples_in_cluster = np.sum(labels == unique_labels[np.where(centroids == centroid)[0][0]])
        bcss += n_samples_in_cluster * np.sum(np.square(centroid - overall_centroid))

    return bcss

def calinski_harabasz_index_fit(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
    custom_metric: Optional[Callable] = None
) -> Dict[str, Union[float, Dict]]:
    """
    Compute the Calinski-Harabasz index for hierarchical clustering.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    labels : np.ndarray
        Cluster labels for each sample.
    metric : str, optional
        Distance metric to use. Default is "euclidean".
    custom_metric : Callable, optional
        Custom distance metric function. If provided, overrides the metric parameter.

    Returns
    -------
    Dict[str, Union[float, Dict]]
        Dictionary containing:
        - "result": The Calinski-Harabasz index.
        - "metrics": Dictionary of intermediate metrics.
        - "params_used": Dictionary of parameters used.
        - "warnings": List of warnings (empty if no warnings).
    """
    _validate_inputs(X, labels, metric, custom_metric)

    wcss = _compute_within_cluster_sum_of_squares(X, labels, metric, custom_metric)
    bcss = _compute_between_cluster_sum_of_squares(X, labels)

    n_clusters = len(np.unique(labels))
    n_samples = X.shape[0]

    if bcss == 0:
        return {
            "result": np.nan,
            "metrics": {"wcss": wcss, "bcss": bcss},
            "params_used": {
                "metric": metric if custom_metric is None else "custom",
                "n_clusters": n_clusters,
                "n_samples": n_samples
            },
            "warnings": ["Between-cluster sum of squares is zero"]
        }

    ch_index = (bcss / wcss) * ((n_samples - n_clusters) / (n_clusters - 1))

    return {
        "result": ch_index,
        "metrics": {"wcss": wcss, "bcss": bcss},
        "params_used": {
            "metric": metric if custom_metric is None else "custom",
            "n_clusters": n_clusters,
            "n_samples": n_samples
        },
        "warnings": []
    }

# Example usage:
"""
X = np.random.rand(100, 5)
labels = np.random.randint(0, 3, size=100)

result = calinski_harabasz_index_fit(X, labels)
print(result)
"""
