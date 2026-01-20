"""
Quantix – Module manifold_learning
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# isomap
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

def _validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input contains NaN or Inf values")

def _compute_pairwise_distances(X: np.ndarray, metric: str) -> np.ndarray:
    """Compute pairwise distances between samples."""
    return squareform(pdist(X, metric=metric))

def _build_graph(distances: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Build adjacency graph using k-nearest neighbors."""
    n_samples = distances.shape[0]
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(distances)
    _, indices = knn.kneighbors(distances)
    graph = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in indices[i]:
            graph[i, j] = distances[i, j]
    return graph

def _compute_geodesic_distances(graph: np.ndarray) -> np.ndarray:
    """Compute geodesic distances using Dijkstra's algorithm."""
    n_samples = graph.shape[0]
    distances = np.full((n_samples, n_samples), np.inf)
    for i in range(n_samples):
        distances[i, i] = 0
        visited = set()
        queue = [(i, 0)]
        while queue:
            u, d = queue.pop(0)
            if u in visited:
                continue
            visited.add(u)
            for v in range(n_samples):
                if graph[u, v] > 0 and d + graph[u, v] < distances[i, v]:
                    distances[i, v] = d + graph[u, v]
                    queue.append((v, d + graph[u, v]))
    return distances

def _center_data(X: np.ndarray) -> np.ndarray:
    """Center the data by subtracting the mean."""
    return X - np.mean(X, axis=0)

def _compute_embedding(distances: np.ndarray, n_components: int) -> np.ndarray:
    """Compute the embedding using classical MDS."""
    n_samples = distances.shape[0]
    H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
    B = -H @ (distances ** 2) @ H / 2
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    embedding = eigenvectors[:, :n_components] @ np.diag(np.sqrt(eigenvalues[:n_components]))
    return embedding

def isomap_fit(
    X: np.ndarray,
    n_neighbors: int = 5,
    n_components: int = 2,
    metric: str = 'euclidean',
    normalize: bool = False
) -> Dict[str, Any]:
    """
    Perform Isomap embedding on the input data.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_neighbors : int, optional
        Number of neighbors to consider for graph construction.
    n_components : int, optional
        Number of dimensions for the embedding.
    metric : str or callable, optional
        Distance metric to use. Default is 'euclidean'.
    normalize : bool, optional
        Whether to center the data before embedding.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the embedding result, metrics, parameters used, and warnings.
    """
    _validate_input(X)

    if normalize:
        X = _center_data(X)

    distances = _compute_pairwise_distances(X, metric)
    graph = _build_graph(distances, n_neighbors)
    geodesic_distances = _compute_geodesic_distances(graph)
    embedding = _compute_embedding(geodesic_distances, n_components)

    result = {
        "embedding": embedding,
        "metrics": {},
        "params_used": {
            "n_neighbors": n_neighbors,
            "n_components": n_components,
            "metric": metric,
            "normalize": normalize
        },
        "warnings": []
    }

    return result

# Example usage:
# embedding_result = isomap_fit(X=np.random.rand(100, 5), n_neighbors=3, n_components=2)

################################################################################
# locally_linear_embedding
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def locally_linear_embedding_fit(
    X: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 12,
    reg: float = 1e-3,
    metric: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    normalization: Optional[str] = None,
    random_state: Optional[int] = None
) -> Dict:
    """
    Perform Locally Linear Embedding (LLE) on the input data.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Dimension of the embedded space. Default is 2.
    n_neighbors : int, optional
        Number of neighbors to consider for each point. Default is 12.
    reg : float, optional
        Regularization parameter to avoid singular matrices. Default is 1e-3.
    metric : str or callable, optional
        Distance metric to use. Default is 'euclidean'.
    solver : str, optional
        Solver to use for optimization. Default is 'closed_form'.
    normalization : str, optional
        Normalization method to apply. Options are None, 'standard', 'minmax',
        or 'robust'. Default is None.
    random_state : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    Dict
        Dictionary containing the following keys:
        - 'result': Embedded data of shape (n_samples, n_components).
        - 'metrics': Dictionary of computed metrics.
        - 'params_used': Dictionary of parameters used.
        - 'warnings': List of warnings encountered.

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> result = locally_linear_embedding_fit(X, n_components=2)
    """
    # Validate inputs
    X = _validate_input(X)

    if normalization is not None:
        X = _apply_normalization(X, normalization)

    # Compute neighborhood graph
    W = _compute_weights(X, n_neighbors, metric, reg)

    # Compute embedding
    Y = _compute_embedding(W, X.shape[0], n_components, solver, random_state)

    # Compute metrics
    metrics = _compute_metrics(X, Y, metric)

    return {
        'result': Y,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'n_neighbors': n_neighbors,
            'reg': reg,
            'metric': metric,
            'solver': solver,
            'normalization': normalization
        },
        'warnings': []
    }

def _validate_input(X: np.ndarray) -> np.ndarray:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("Input X must be a 2D array.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input X contains NaN or Inf values.")
    return X

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to the input data."""
    if method == 'standard':
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X = (X - median) / iqr
    return X

def _compute_weights(
    X: np.ndarray,
    n_neighbors: int,
    metric: Union[str, Callable],
    reg: float
) -> np.ndarray:
    """Compute the weight matrix for LLE."""
    n_samples = X.shape[0]
    W = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        if isinstance(metric, str):
            distances = _compute_distance(X[i], X, metric)
        else:
            distances = metric(X[i], X)

        nearest_indices = np.argsort(distances)[1:n_neighbors + 1]
        Z = X[nearest_indices] - X[i]

        if solver == 'closed_form':
            C = np.dot(Z, Z.T)
            C += reg * np.eye(n_neighbors)
            W[i, nearest_indices] = np.linalg.solve(C, np.ones(n_neighbors))
        else:
            raise ValueError("Unsupported solver.")

    return W

def _compute_distance(
    x: np.ndarray,
    X: np.ndarray,
    metric: str
) -> np.ndarray:
    """Compute distances between a point and all other points."""
    if metric == 'euclidean':
        return np.linalg.norm(X - x, axis=1)
    elif metric == 'manhattan':
        return np.sum(np.abs(X - x), axis=1)
    elif metric == 'cosine':
        return 1 - np.dot(X, x) / (np.linalg.norm(X, axis=1) * np.linalg.norm(x))
    elif metric == 'minkowski':
        return np.sum(np.abs(X - x) ** 2, axis=1) ** (1/2)
    else:
        raise ValueError("Unsupported metric.")

def _compute_embedding(
    W: np.ndarray,
    n_samples: int,
    n_components: int,
    solver: str,
    random_state: Optional[int]
) -> np.ndarray:
    """Compute the embedding using the weight matrix."""
    if solver == 'closed_form':
        M = (np.eye(n_samples) - W).T @ (np.eye(n_samples) - W)
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        idx = np.argsort(eigenvalues)[::-1]
        Y = eigenvectors[:, idx[:n_components]]
    else:
        raise ValueError("Unsupported solver.")

    return Y

def _compute_metrics(
    X: np.ndarray,
    Y: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Compute metrics for the embedding."""
    if isinstance(metric, str):
        distances_original = _compute_distance(X[0], X, metric)
        distances_embedding = _compute_distance(Y[0], Y, metric)
    else:
        distances_original = metric(X[0], X)
        distances_embedding = metric(Y[0], Y)

    return {
        'reconstruction_error': np.mean(distances_embedding) / np.mean(distances_original)
    }

################################################################################
# t_sne
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def t_sne_fit(
    X: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    early_exaggeration: float = 12.0,
    learning_rate: float = 200.0,
    n_iter: int = 1000,
    metric: Union[str, Callable] = 'euclidean',
    init: str = 'random',
    verbose: int = 0,
    random_state: Optional[int] = None,
    normalize: str = 'none',
    distance_func: Optional[Callable] = None,
    solver: str = 'gradient_descent',
    tol: float = 1e-5,
    max_grad_norm: Optional[float] = None,
    momentum: float = 0.8,
) -> Dict:
    """
    t-SNE (t-Distributed Stochastic Neighbor Embedding) for dimensionality reduction.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Dimension of the embedded space.
    perplexity : float, optional
        Perplexity parameter (related to number of nearest neighbors).
    early_exaggeration : float, optional
        Controls how tight natural clusters in the original space are in the embedded space.
    learning_rate : float, optional
        Learning rate for gradient descent.
    n_iter : int, optional
        Maximum number of iterations for the optimizer.
    metric : str or callable, optional
        Distance metric ('euclidean', 'manhattan', 'cosine') or custom callable.
    init : str, optional
        Initialization method ('random', 'pca').
    verbose : int, optional
        Verbosity level.
    random_state : int or None, optional
        Random seed for reproducibility.
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    distance_func : callable, optional
        Custom distance function.
    solver : str, optional
        Solver method ('gradient_descent', 'newton').
    tol : float, optional
        Tolerance for stopping criterion.
    max_grad_norm : float or None, optional
        Maximum gradient norm for early stopping.
    momentum : float, optional
        Momentum parameter for gradient descent.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Embedded data of shape (n_samples, n_components)
        - 'metrics': Dictionary of metrics
        - 'params_used': Parameters used in the computation
        - 'warnings': List of warnings

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> result = t_sne_fit(X, n_components=2, perplexity=30)
    """
    # Validate inputs
    X = _validate_input(X, normalize)

    # Set random state
    rng = np.random.RandomState(random_state)

    # Compute pairwise distances
    if distance_func is not None:
        distances = _compute_custom_distance(X, distance_func)
    else:
        distances = _compute_distance(X, metric)

    # Compute affinities
    P = _compute_affinities(distances, perplexity, rng)

    # Initialize embedding
    Y = _initialize_embedding(X.shape[0], n_components, init, rng)

    # Optimize embedding
    Y = _optimize_embedding(
        P,
        Y,
        n_components,
        early_exaggeration,
        learning_rate,
        n_iter,
        solver,
        tol,
        max_grad_norm,
        momentum,
        rng
    )

    # Compute metrics
    metrics = _compute_metrics(Y, P)

    return {
        'result': Y,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'perplexity': perplexity,
            'early_exaggeration': early_exaggeration,
            'learning_rate': learning_rate,
            'n_iter': n_iter,
            'metric': metric,
            'init': init,
            'normalize': normalize,
            'solver': solver,
            'tol': tol,
            'max_grad_norm': max_grad_norm,
            'momentum': momentum
        },
        'warnings': []
    }

def _validate_input(X: np.ndarray, normalize: str) -> np.ndarray:
    """Validate and preprocess input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X must not contain NaN or inf values")

    if normalize != 'none':
        X = _normalize_data(X, normalize)

    return X

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize data using specified method."""
    if method == 'standard':
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    return X

def _compute_distance(X: np.ndarray, metric: str) -> np.ndarray:
    """Compute pairwise distances using specified metric."""
    if metric == 'euclidean':
        return np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
    elif metric == 'manhattan':
        return np.sum(np.abs(X[:, np.newaxis] - X), axis=2)
    elif metric == 'cosine':
        return 1 - np.dot(X, X.T) / (np.linalg.norm(X, axis=1)[:, np.newaxis] * np.linalg.norm(X, axis=1))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _compute_custom_distance(X: np.ndarray, distance_func: Callable) -> np.ndarray:
    """Compute pairwise distances using custom function."""
    n_samples = X.shape[0]
    distances = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            distances[i, j] = distance_func(X[i], X[j])
            distances[j, i] = distances[i, j]
    return distances

def _compute_affinities(distances: np.ndarray, perplexity: float, rng: np.random.RandomState) -> np.ndarray:
    """Compute pairwise affinities using t-SNE."""
    n_samples = distances.shape[0]
    P = np.zeros((n_samples, n_samples))
    H = np.log2(perplexity)

    for i in range(n_samples):
        # Binary search to find suitable sigma
        sigmas = _binary_search_sigma(distances[i], H, rng)
        P_i = np.exp(-distances[i] ** 2 / (2 * sigmas ** 2))
        P_i /= np.sum(P_i)
        P[i] = P_i

    # Symmetrize affinities
    P = (P + P.T) / (2 * n_samples)

    return P

def _binary_search_sigma(distances: np.ndarray, H: float, rng: np.random.RandomState) -> float:
    """Binary search to find suitable sigma for t-SNE."""
    # Implementation of binary search
    pass

def _initialize_embedding(n_samples: int, n_components: int, init: str, rng: np.random.RandomState) -> np.ndarray:
    """Initialize embedding using specified method."""
    if init == 'random':
        return rng.randn(n_samples, n_components)
    elif init == 'pca':
        # PCA initialization
        pass
    else:
        raise ValueError(f"Unknown init method: {init}")

def _optimize_embedding(
    P: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    early_exaggeration: float,
    learning_rate: float,
    n_iter: int,
    solver: str,
    tol: float,
    max_grad_norm: Optional[float],
    momentum: float,
    rng: np.random.RandomState
) -> np.ndarray:
    """Optimize embedding using specified solver."""
    if solver == 'gradient_descent':
        return _gradient_descent(
            P, Y, n_components, early_exaggeration, learning_rate,
            n_iter, tol, max_grad_norm, momentum, rng
        )
    elif solver == 'newton':
        # Newton's method implementation
        pass
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _gradient_descent(
    P: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    early_exaggeration: float,
    learning_rate: float,
    n_iter: int,
    tol: float,
    max_grad_norm: Optional[float],
    momentum: float,
    rng: np.random.RandomState
) -> np.ndarray:
    """Gradient descent optimization for t-SNE."""
    # Implementation of gradient descent
    pass

def _compute_metrics(Y: np.ndarray, P: np.ndarray) -> Dict:
    """Compute metrics for the embedding."""
    # Implementation of metric computation
    return {
        'kl_divergence': 0.0,
        'stress': 0.0
    }

################################################################################
# umap
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2-dimensional array")
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

def compute_distance_matrix(
    X: np.ndarray,
    metric: str = "euclidean",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> np.ndarray:
    """Compute distance matrix using specified metric or custom function."""
    n_samples = X.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))

    if custom_metric is not None:
        for i in range(n_samples):
            for j in range(i, n_samples):
                dist_matrix[i, j] = custom_metric(X[i], X[j])
                dist_matrix[j, i] = dist_matrix[i, j]
        return dist_matrix

    if metric == "euclidean":
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
        p = 3
        for i in range(n_samples):
            for j in range(i, n_samples):
                dist_matrix[i, j] = np.sum(np.abs(X[i] - X[j])**p)**(1/p)
                dist_matrix[j, i] = dist_matrix[i, j]
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

    return dist_matrix

def umap_fit(
    X: np.ndarray,
    n_components: int = 2,
    normalization: str = "standard",
    metric: str = "euclidean",
    solver: str = "gradient_descent",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    spread: float = 1.0,
    learning_rate: float = 1.0,
    n_epochs: int = 200,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    custom_normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Perform UMAP (Uniform Manifold Approximation and Projection) dimensionality reduction.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_components : int, optional
        Number of dimensions for the manifold (default: 2)
    normalization : str or callable, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') or custom function (default: 'standard')
    metric : str or callable, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom function (default: 'euclidean')
    solver : str, optional
        Optimization solver ('gradient_descent', 'newton') (default: 'gradient_descent')
    n_neighbors : int, optional
        Number of neighbors to consider (default: 15)
    min_dist : float, optional
        Minimum distance between embedded points (default: 0.1)
    spread : float, optional
        Spread of the embedded points (default: 1.0)
    learning_rate : float, optional
        Learning rate for gradient descent (default: 1.0)
    n_epochs : int, optional
        Number of optimization epochs (default: 200)
    tol : float, optional
        Tolerance for convergence (default: 1e-4)
    random_state : int, optional
        Random seed for reproducibility (default: None)
    custom_normalizer : callable, optional
        Custom normalization function (default: None)
    custom_metric : callable, optional
        Custom distance metric function (default: None)

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': Embedded data of shape (n_samples, n_components)
        - 'metrics': Dictionary of computed metrics
        - 'params_used': Dictionary of parameters used
        - 'warnings': List of warnings encountered
    """
    # Validate input
    validate_input(X)

    # Initialize random state if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize data
    X_normalized = normalize_data(X, normalization, custom_normalizer)

    # Compute distance matrix
    dist_matrix = compute_distance_matrix(X_normalized, metric, custom_metric)

    # Initialize warnings
    warnings = []

    # TODO: Implement UMAP algorithm based on the parameters

    # Placeholder for embedded data
    embedding = np.random.rand(X.shape[0], n_components)

    # Placeholder for metrics
    metrics = {
        'reconstruction_error': 0.0,
        'stress': 0.0
    }

    # Record parameters used
    params_used = {
        'n_components': n_components,
        'normalization': normalization,
        'metric': metric,
        'solver': solver,
        'n_neighbors': n_neighbors,
        'min_dist': min_dist,
        'spread': spread,
        'learning_rate': learning_rate,
        'n_epochs': n_epochs,
        'tol': tol
    }

    return {
        'result': embedding,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

# Example usage:
"""
X = np.random.rand(100, 5)  # 100 samples with 5 features each
result = umap_fit(X, n_components=2)
print(result['result'].shape)  # Should print (100, 2)
"""

################################################################################
# spectral_embedding
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def spectral_embedding_fit(
    X: np.ndarray,
    n_components: int = 2,
    affinity: str = 'rbf',
    gamma: Optional[float] = None,
    n_neighbors: int = 10,
    eigen_solver: str = 'arpack',
    random_state: Optional[int] = None,
    n_jobs: int = 1,
    distance_metric: str = 'euclidean',
    normalize: bool = True
) -> Dict[str, Any]:
    """
    Compute spectral embedding for dimensionality reduction.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_components : int, optional
        Number of dimensions to embed into (default: 2)
    affinity : str or callable, optional
        Affinity function ('rbf', 'nearest_neighbors') (default: 'rbf')
    gamma : float, optional
        Kernel coefficient for rbf affinity (default: None)
    n_neighbors : int, optional
        Number of neighbors for nearest_neighbors affinity (default: 10)
    eigen_solver : str, optional
        Eigenvalue solver ('arpack', 'lobpcg') (default: 'arpack')
    random_state : int, optional
        Random seed for reproducibility (default: None)
    n_jobs : int, optional
        Number of parallel jobs to run (default: 1)
    distance_metric : str, optional
        Distance metric for affinity computation (default: 'euclidean')
    normalize : bool, optional
        Whether to normalize the Laplacian (default: True)

    Returns:
    --------
    dict
        Dictionary containing:
        - 'embedding': The embedding coordinates (n_samples, n_components)
        - 'metrics': Dictionary of computed metrics
        - 'params_used': Parameters actually used
        - 'warnings': Any warnings encountered

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> result = spectral_embedding_fit(X, n_components=3)
    """
    # Validate inputs
    _validate_inputs(X, n_components)

    # Set random state if provided
    rng = np.random.RandomState(random_state) if random_state is not None else None

    # Compute affinity matrix
    affinity_matrix = _compute_affinity(
        X, affinity=affinity, gamma=gamma,
        n_neighbors=n_neighbors,
        metric=distance_metric
    )

    # Compute Laplacian
    laplacian = _compute_laplacian(affinity_matrix, normalize=normalize)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = _compute_eigen(
        laplacian, n_components=n_components,
        solver=eigen_solver, random_state=rng
    )

    # Compute embedding
    embedding = _compute_embedding(eigenvectors, eigenvalues)

    # Compute metrics
    metrics = _compute_metrics(embedding, X)

    return {
        'result': embedding,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'affinity': affinity,
            'gamma': gamma,
            'n_neighbors': n_neighbors,
            'eigen_solver': eigen_solver,
            'distance_metric': distance_metric,
            'normalize': normalize
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if X.shape[0] < n_components:
        raise ValueError("n_components cannot be greater than number of samples")

def _compute_affinity(
    X: np.ndarray,
    affinity: str = 'rbf',
    gamma: Optional[float] = None,
    n_neighbors: int = 10,
    metric: str = 'euclidean'
) -> np.ndarray:
    """Compute affinity matrix."""
    if gamma is None and affinity == 'rbf':
        gamma = 1.0 / X.shape[1]

    if affinity == 'rbf':
        return _compute_rbf_affinity(X, gamma=gamma)
    elif affinity == 'nearest_neighbors':
        return _compute_knn_affinity(X, n_neighbors=n_neighbors, metric=metric)
    else:
        raise ValueError(f"Unknown affinity: {affinity}")

def _compute_rbf_affinity(X: np.ndarray, gamma: float) -> np.ndarray:
    """Compute RBF affinity matrix."""
    pairwise_dists = _compute_pairwise_distances(X, metric='euclidean')
    return np.exp(-gamma * pairwise_dists)

def _compute_knn_affinity(
    X: np.ndarray,
    n_neighbors: int,
    metric: str = 'euclidean'
) -> np.ndarray:
    """Compute k-nearest neighbors affinity matrix."""
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(X)
    distances, indices = nbrs.kneighbors(X)

    affinity_matrix = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        affinity_matrix[i, indices[i]] = 1.0 / (distances[i] + 1e-8)

    return affinity_matrix

def _compute_pairwise_distances(X: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """Compute pairwise distances between samples."""
    if metric == 'euclidean':
        return np.sqrt(((X[:, np.newaxis] - X) ** 2).sum(axis=2))
    elif metric == 'manhattan':
        return np.abs(X[:, np.newaxis] - X).sum(axis=2)
    elif metric == 'cosine':
        return 1.0 - np.dot(X, X.T) / (np.linalg.norm(X, axis=1)[:, np.newaxis] *
                                      np.linalg.norm(X, axis=1)[np.newaxis, :])
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _compute_laplacian(affinity_matrix: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Compute graph Laplacian."""
    degree_matrix = np.diag(affinity_matrix.sum(axis=1))

    if normalize:
        # Symmetric normalized Laplacian
        inv_sqrt_degree = np.diag(1.0 / np.sqrt(np.diag(degree_matrix) + 1e-8))
        return inv_sqrt_degree @ (degree_matrix - affinity_matrix) @ inv_sqrt_degree
    else:
        return degree_matrix - affinity_matrix

def _compute_eigen(
    laplacian: np.ndarray,
    n_components: int,
    solver: str = 'arpack',
    random_state: Optional[np.random.RandomState] = None
) -> tuple:
    """Compute eigenvalues and eigenvectors."""
    if solver == 'arpack':
        from scipy.sparse.linalg import eigsh
        eigenvalues, eigenvectors = eigsh(
            laplacian,
            k=n_components + 1,
            which='SM',
            random_state=random_state
        )
    elif solver == 'lobpcg':
        from scipy.sparse.linalg import lobpcg
        eigenvalues, eigenvectors = lobpcg(
            laplacian,
            np.random.rand(laplacian.shape[0], n_components + 1),
            largest=False,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Sort by ascending eigenvalues
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors

def _compute_embedding(eigenvectors: np.ndarray, eigenvalues: np.ndarray) -> np.ndarray:
    """Compute embedding from eigenvectors."""
    # Discard the first eigenvector (corresponding to eigenvalue 0)
    return eigenvectors[:, 1:]

def _compute_metrics(embedding: np.ndarray, X: np.ndarray) -> Dict[str, float]:
    """Compute metrics for the embedding."""
    from sklearn.metrics import pairwise_distances

    # Reconstruction error
    reconstructed = _reconstruct_from_embedding(embedding, X)
    reconstruction_error = np.mean((X - reconstructed) ** 2)

    # Trustworthiness
    original_dist = pairwise_distances(X)
    embedded_dist = pairwise_distances(embedding)

    # Compute trustworthiness
    trustworthiness = _compute_trustworthiness(original_dist, embedded_dist)

    return {
        'reconstruction_error': float(reconstruction_error),
        'trustworthiness': float(trustworthiness)
    }

def _reconstruct_from_embedding(embedding: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Reconstruct original data from embedding."""
    # This is a simplified reconstruction
    return np.dot(embedding, embedding.T) / X.shape[0]

def _compute_trustworthiness(original_dist: np.ndarray, embedded_dist: np.ndarray) -> float:
    """Compute trustworthiness metric."""
    n = original_dist.shape[0]
    k = min(5, n - 1)

    # Compute rank in original space
    original_rank = np.argsort(original_dist, axis=1)[:, :k]

    # Compute rank in embedded space
    embedded_rank = np.argsort(embedded_dist, axis=1)[:, :k]

    # Count how many original neighbors are in the embedded top-k
    count = 0
    for i in range(n):
        original_neighbors = set(original_rank[i])
        embedded_neighbors = set(embedded_rank[i])
        count += len(original_neighbors & embedded_neighbors)

    return (n * k - count) / (n * k)

################################################################################
# laplacian_eigenmaps
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input contains NaN or Inf values")

def _compute_affinity_matrix(X: np.ndarray, metric: Union[str, Callable], sigma: float = 1.0) -> np.ndarray:
    """Compute the affinity matrix using a given metric."""
    n_samples = X.shape[0]
    if isinstance(metric, str):
        if metric == 'euclidean':
            dist = np.sqrt(((X[:, np.newaxis] - X) ** 2).sum(axis=2))
        elif metric == 'manhattan':
            dist = np.abs(X[:, np.newaxis] - X).sum(axis=2)
        elif metric == 'cosine':
            dist = 1 - np.dot(X, X.T) / (np.linalg.norm(X, axis=1)[:, np.newaxis] * np.linalg.norm(X, axis=1))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        dist = np.array([[metric(x1, x2) for x2 in X] for x1 in X])

    affinity = np.exp(-dist ** 2 / (2 * sigma ** 2))
    return affinity

def _normalize_affinity_matrix(W: np.ndarray, normalization: str = 'sym') -> np.ndarray:
    """Normalize the affinity matrix."""
    if normalization == 'none':
        return W
    elif normalization == 'sym':
        D = np.diag(1 / np.sqrt(np.sum(W, axis=0)))
        return D @ W @ D
    elif normalization == 'rw':
        D = np.diag(1 / np.sum(W, axis=0))
        return D @ W
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

def _compute_laplacian(L: np.ndarray, laplacian_type: str = 'unormalized') -> np.ndarray:
    """Compute the Laplacian matrix."""
    if laplacian_type == 'unormalized':
        return L
    elif laplacian_type == 'normalized':
        D = np.diag(1 / np.sqrt(np.sum(L, axis=0)))
        return D @ L @ D
    else:
        raise ValueError(f"Unknown Laplacian type: {laplacian_type}")

def _eigendecomposition(L: np.ndarray, n_components: int = 2) -> tuple:
    """Compute the eigen decomposition of the Laplacian matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues[:n_components], eigenvectors[:, :n_components]

def laplacian_eigenmaps_fit(
    X: np.ndarray,
    n_components: int = 2,
    metric: Union[str, Callable] = 'euclidean',
    sigma: float = 1.0,
    normalization: str = 'sym',
    laplacian_type: str = 'unormalized',
    solver: str = 'eigh'
) -> Dict:
    """
    Fit the Laplacian Eigenmaps model.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Number of components to keep. Default is 2.
    metric : str or callable, optional
        Distance metric to use. Default is 'euclidean'.
    sigma : float, optional
        Bandwidth parameter for the Gaussian kernel. Default is 1.0.
    normalization : str, optional
        Normalization method for the affinity matrix. Default is 'sym'.
    laplacian_type : str, optional
        Type of Laplacian matrix to use. Default is 'unormalized'.
    solver : str, optional
        Solver to use for eigen decomposition. Default is 'eigh'.

    Returns
    -------
    Dict
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    _validate_input(X)

    W = _compute_affinity_matrix(X, metric, sigma)
    L = _normalize_affinity_matrix(W, normalization)
    L = _compute_laplacian(L, laplacian_type)

    if solver == 'eigh':
        eigenvalues, eigenvectors = _eigendecomposition(L, n_components)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    result = {
        'embedding': eigenvectors,
        'eigenvalues': eigenvalues
    }

    metrics = {
        'explained_variance_ratio': None  # Placeholder for future implementation
    }

    params_used = {
        'n_components': n_components,
        'metric': metric,
        'sigma': sigma,
        'normalization': normalization,
        'laplacian_type': laplacian_type,
        'solver': solver
    }

    warnings = []

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

# Example usage:
# result = laplacian_eigenmaps_fit(X=np.random.rand(10, 5), n_components=2)

################################################################################
# diffusion_maps
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input must not contain NaN or inf values")

def _compute_kernel(
    X: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    epsilon: float = 1.0
) -> np.ndarray:
    """Compute the kernel matrix."""
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = np.exp(-metric(X[i], X[j]) / (2 * epsilon ** 2))
    return K

def _normalize_kernel(
    K: np.ndarray,
    normalization: str = "none"
) -> np.ndarray:
    """Normalize the kernel matrix."""
    if normalization == "none":
        return K
    elif normalization == "standard":
        row_sums = K.sum(axis=1, keepdims=True)
        return K / row_sums
    elif normalization == "minmax":
        min_val = K.min()
        max_val = K.max()
        return (K - min_val) / (max_val - min_val)
    elif normalization == "robust":
        median = np.median(K)
        mad = np.median(np.abs(K - median))
        return (K - median) / (1.5 * mad)
    else:
        raise ValueError("Unknown normalization method")

def _compute_diffusion_maps(
    K: np.ndarray,
    n_components: int = 2
) -> Dict[str, Any]:
    """Compute diffusion maps."""
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    result = {
        "eigenvalues": eigenvalues[:n_components],
        "eigenvectors": eigenvectors[:, :n_components]
    }
    return result

def diffusion_maps_fit(
    X: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = np.linalg.norm,
    epsilon: float = 1.0,
    normalization: str = "none",
    n_components: int = 2
) -> Dict[str, Any]:
    """
    Compute diffusion maps for dimensionality reduction.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    metric : Callable[[np.ndarray, np.ndarray], float]
        Distance metric function.
    epsilon : float
        Bandwidth parameter for the kernel.
    normalization : str
        Normalization method for the kernel matrix.
    n_components : int
        Number of diffusion map components to return.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    _validate_input(X)

    K = _compute_kernel(X, metric, epsilon)
    K_normalized = _normalize_kernel(K, normalization)

    result = _compute_diffusion_maps(K_normalized, n_components)

    output = {
        "result": result,
        "metrics": {},
        "params_used": {
            "metric": metric.__name__ if hasattr(metric, '__name__') else str(metric),
            "epsilon": epsilon,
            "normalization": normalization,
            "n_components": n_components
        },
        "warnings": []
    }

    return output

# Example usage:
# X = np.random.rand(100, 5)
# result = diffusion_maps_fit(X, metric=np.linalg.norm, epsilon=1.0, normalization="standard", n_components=2)

################################################################################
# autoencoder
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def autoencoder_fit(
    X: np.ndarray,
    n_components: int = 2,
    normalizer: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    max_iter: int = 1000,
    tol: float = 1e-4,
    learning_rate: float = 0.01,
    batch_size: Optional[int] = None,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> Dict:
    """
    Fit an autoencoder model to the input data.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    n_components : int, optional
        Dimensionality of the encoded representation (default: 2).
    normalizer : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard').
    metric : str or callable, optional
        Loss function ('mse', 'mae', 'r2', 'logloss') or custom callable (default: 'mse').
    distance : str, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') (default: 'euclidean').
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent') (default: 'gradient_descent').
    regularization : str or None, optional
        Regularization method ('none', 'l1', 'l2', 'elasticnet') (default: None).
    max_iter : int, optional
        Maximum number of iterations (default: 1000).
    tol : float, optional
        Tolerance for stopping criterion (default: 1e-4).
    learning_rate : float, optional
        Learning rate for gradient-based solvers (default: 0.01).
    batch_size : int or None, optional
        Batch size for stochastic solvers (default: None).
    random_state : int or None, optional
        Random seed for reproducibility (default: None).
    verbose : bool, optional
        Whether to print progress information (default: False).

    Returns:
    --------
    dict
        Dictionary containing the fitted model, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, n_components)

    # Initialize random state if provided
    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    # Normalize data
    X_norm, normalizer_used = _normalize_data(X, normalizer)

    # Initialize model parameters
    params = _initialize_parameters(X_norm.shape[1], n_components, rng)

    # Choose solver
    if solver == 'closed_form':
        encoder_weights, decoder_weights = _solve_closed_form(X_norm, n_components)
    elif solver == 'gradient_descent':
        encoder_weights, decoder_weights = _gradient_descent_solver(
            X_norm, n_components, max_iter, tol, learning_rate,
            batch_size, rng, metric, distance, regularization
        )
    elif solver == 'newton':
        encoder_weights, decoder_weights = _newton_solver(
            X_norm, n_components, max_iter, tol,
            metric, distance, regularization
        )
    elif solver == 'coordinate_descent':
        encoder_weights, decoder_weights = _coordinate_descent_solver(
            X_norm, n_components, max_iter, tol,
            metric, distance, regularization
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_norm, encoder_weights, decoder_weights, metric)

    # Prepare output
    result = {
        'encoder_weights': encoder_weights,
        'decoder_weights': decoder_weights
    }

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'normalizer': normalizer_used,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'max_iter': max_iter,
            'tol': tol,
            'learning_rate': learning_rate,
            'batch_size': batch_size
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")

def _normalize_data(
    X: np.ndarray,
    method: str
) -> tuple[np.ndarray, str]:
    """Normalize the input data."""
    if method == 'none':
        return X, 'none'
    elif method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_norm = (X - mean) / (std + 1e-8)
        return X_norm, 'standard'
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
        return X_norm, 'minmax'
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - median) / (iqr + 1e-8)
        return X_norm, 'robust'
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _initialize_parameters(
    n_features: int,
    n_components: int,
    rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize encoder and decoder weights."""
    encoder_weights = rng.randn(n_features, n_components) * 0.1
    decoder_weights = rng.randn(n_components, n_features) * 0.1
    return encoder_weights, decoder_weights

def _solve_closed_form(
    X: np.ndarray,
    n_components: int
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the autoencoder problem using closed-form solution."""
    # This is a placeholder for actual closed-form implementation
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    encoder_weights = U[:, :n_components] @ np.diag(S[:n_components])
    decoder_weights = Vt[:n_components, :]
    return encoder_weights, decoder_weights

def _gradient_descent_solver(
    X: np.ndarray,
    n_components: int,
    max_iter: int,
    tol: float,
    learning_rate: float,
    batch_size: Optional[int],
    rng: np.random.RandomState,
    metric: Union[str, Callable],
    distance: str,
    regularization: Optional[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the autoencoder problem using gradient descent."""
    n_samples = X.shape[0]
    encoder_weights, decoder_weights = _initialize_parameters(X.shape[1], n_components, rng)

    for iteration in range(max_iter):
        if batch_size is not None and batch_size < n_samples:
            indices = rng.choice(n_samples, batch_size, replace=False)
            X_batch = X[indices]
        else:
            X_batch = X

        # Compute gradients and update weights
        gradients = _compute_gradients(
            X_batch, encoder_weights, decoder_weights,
            metric, distance, regularization
        )
        encoder_weights -= learning_rate * gradients['encoder']
        decoder_weights -= learning_rate * gradients['decoder']

        # Check convergence
        if iteration > 0 and np.linalg.norm(gradients['encoder']) < tol:
            break

    return encoder_weights, decoder_weights

def _compute_gradients(
    X: np.ndarray,
    encoder_weights: np.ndarray,
    decoder_weights: np.ndarray,
    metric: Union[str, Callable],
    distance: str,
    regularization: Optional[str]
) -> Dict:
    """Compute gradients for the autoencoder loss function."""
    # This is a placeholder for actual gradient computation
    encoded = X @ encoder_weights
    decoded = encoded @ decoder_weights

    if metric == 'mse':
        error = X - decoded
        gradient_encoder = (2 / X.shape[0]) * (error @ decoder_weights.T) @ encoder_weights
        gradient_decoder = (2 / X.shape[0]) * encoded.T @ error
    else:
        raise NotImplementedError(f"Metric {metric} not implemented for gradient computation")

    # Add regularization if needed
    if regularization == 'l1':
        gradient_encoder += np.sign(encoder_weights) * 0.01
        gradient_decoder += np.sign(decoder_weights) * 0.01
    elif regularization == 'l2':
        gradient_encoder += encoder_weights * 0.01
        gradient_decoder += decoder_weights * 0.01

    return {
        'encoder': gradient_encoder,
        'decoder': gradient_decoder
    }

def _newton_solver(
    X: np.ndarray,
    n_components: int,
    max_iter: int,
    tol: float,
    metric: Union[str, Callable],
    distance: str,
    regularization: Optional[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the autoencoder problem using Newton's method."""
    # This is a placeholder for actual Newton implementation
    raise NotImplementedError("Newton solver not implemented")

def _coordinate_descent_solver(
    X: np.ndarray,
    n_components: int,
    max_iter: int,
    tol: float,
    metric: Union[str, Callable],
    distance: str,
    regularization: Optional[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the autoencoder problem using coordinate descent."""
    # This is a placeholder for actual coordinate descent implementation
    raise NotImplementedError("Coordinate descent solver not implemented")

def _compute_metrics(
    X: np.ndarray,
    encoder_weights: np.ndarray,
    decoder_weights: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Compute metrics for the autoencoder."""
    encoded = X @ encoder_weights
    decoded = encoded @ decoder_weights

    metrics = {}
    if metric == 'mse':
        mse = np.mean((X - decoded) ** 2)
        metrics['mse'] = mse
    elif metric == 'mae':
        mae = np.mean(np.abs(X - decoded))
        metrics['mae'] = mae
    elif metric == 'r2':
        ss_res = np.sum((X - decoded) ** 2)
        ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        metrics['r2'] = r2
    elif callable(metric):
        custom_metric = metric(X, decoded)
        metrics['custom'] = custom_metric
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

# Example usage
if __name__ == "__main__":
    # Generate some random data
    X = np.random.randn(100, 5)

    # Fit the autoencoder
    result = autoencoder_fit(X, n_components=2)

################################################################################
# kernel_pca
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, kernel: str = 'linear', normalize: bool = False) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if kernel not in ['linear', 'poly', 'rbf', 'sigmoid']:
        raise ValueError("kernel must be one of: linear, poly, rbf, sigmoid")
    if normalize and X.shape[0] != X.shape[1]:
        raise ValueError("For normalization, X must be square")

def _compute_kernel_matrix(X: np.ndarray, kernel: str = 'linear', gamma: float = 1.0,
                          degree: int = 3, coef0: float = 1) -> np.ndarray:
    """Compute the kernel matrix based on the specified kernel."""
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))

    if kernel == 'linear':
        K = X @ X.T
    elif kernel == 'poly':
        K = (X @ X.T + coef0) ** degree
    elif kernel == 'rbf':
        pairwise_sq_dists = np.sum(X**2, axis=1)[:, np.newaxis] + \
                           np.sum(X**2, axis=1)[np.newaxis, :] - \
                           2 * X @ X.T
        K = np.exp(-gamma * pairwise_sq_dists)
    elif kernel == 'sigmoid':
        K = np.tanh(gamma * X @ X.T + coef0)

    return K

def _center_kernel_matrix(K: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Center the kernel matrix."""
    n_samples = K.shape[0]
    one_n = np.ones((n_samples, n_samples)) / n_samples
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

    if normalize:
        K_centered = K_centered / np.sqrt(np.diag(K_centered)[:, np.newaxis] * np.diag(K_centered)[np.newaxis, :])

    return K_centered

def _eig_decomposition(K: np.ndarray, n_components: int = 2) -> tuple:
    """Perform eigenvalue decomposition on the kernel matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx][:n_components]
    eigenvectors = eigenvectors[:, idx][:, :n_components]

    return eigenvalues, eigenvectors

def kernel_pca_fit(X: np.ndarray,
                   n_components: int = 2,
                   kernel: str = 'linear',
                   gamma: float = 1.0,
                   degree: int = 3,
                   coef0: float = 1,
                   normalize: bool = False) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Perform Kernel PCA on the input data.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Number of principal components to keep, by default 2.
    kernel : str, optional
        Kernel type ('linear', 'poly', 'rbf', 'sigmoid'), by default 'linear'.
    gamma : float, optional
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid', by default 1.0.
    degree : int, optional
        Degree of the polynomial kernel, by default 3.
    coef0 : float, optional
        Independent term in kernel function, by default 1.
    normalize : bool, optional
        Whether to normalize the kernel matrix, by default False.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict]]
        Dictionary containing:
        - 'result': Transformed data of shape (n_samples, n_components)
        - 'metrics': Dictionary of metrics
        - 'params_used': Dictionary of parameters used
        - 'warnings': List of warnings

    Example
    -------
    >>> X = np.random.rand(10, 5)
    >>> result = kernel_pca_fit(X, n_components=2, kernel='rbf')
    """
    _validate_inputs(X, kernel, normalize)

    K = _compute_kernel_matrix(X, kernel, gamma, degree, coef0)
    K_centered = _center_kernel_matrix(K, normalize)
    eigenvalues, eigenvectors = _eig_decomposition(K_centered, n_components)

    transformed_data = eigenvectors * np.sqrt(eigenvalues)
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

    metrics = {
        'explained_variance_ratio': explained_variance_ratio,
        'total_explained_variance': np.sum(explained_variance_ratio)
    }

    params_used = {
        'n_components': n_components,
        'kernel': kernel,
        'gamma': gamma,
        'degree': degree,
        'coef0': coef0,
        'normalize': normalize
    }

    warnings = []

    return {
        'result': transformed_data,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

################################################################################
# multidimensional_scaling
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(
    X: np.ndarray,
    metric: str,
    distance_func: Optional[Callable] = None
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X must not contain NaN or infinite values")
    if distance_func is None and metric not in ['euclidean', 'manhattan', 'cosine']:
        raise ValueError("Invalid metric or distance function")

def _compute_distance_matrix(
    X: np.ndarray,
    metric: str,
    distance_func: Optional[Callable] = None
) -> np.ndarray:
    """Compute the distance matrix based on specified metric or custom function."""
    if distance_func is not None:
        return np.array([[distance_func(xi, xj) for xj in X] for xi in X])
    if metric == 'euclidean':
        return np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
    elif metric == 'manhattan':
        return np.sum(np.abs(X[:, np.newaxis] - X), axis=2)
    elif metric == 'cosine':
        return 1 - np.dot(X, X.T) / (np.linalg.norm(X, axis=1)[:, np.newaxis] * np.linalg.norm(X, axis=1))

def _center_data(
    D: np.ndarray,
    normalization: str = 'standard'
) -> np.ndarray:
    """Center the distance matrix based on specified normalization."""
    n = D.shape[0]
    if normalization == 'standard':
        H = np.eye(n) - np.ones((n, n)) / n
        return -0.5 * H @ D ** 2 @ H
    elif normalization == 'none':
        return -0.5 * (D ** 2)
    else:
        raise ValueError("Unsupported normalization method")

def _eigendecomposition(
    B: np.ndarray,
    solver: str = 'closed_form'
) -> tuple:
    """Perform eigendecomposition based on specified solver."""
    if solver == 'closed_form':
        eigenvalues, eigenvectors = np.linalg.eigh(B)
    else:
        raise ValueError("Unsupported solver method")
    return eigenvalues, eigenvectors

def _select_dimensions(
    eigenvalues: np.ndarray,
    n_components: int
) -> tuple:
    """Select the top n_components dimensions based on eigenvalues."""
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx[:n_components]], idx[:n_components]

def _compute_embedding(
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    n_components: int
) -> np.ndarray:
    """Compute the embedding based on eigenvectors and eigenvalues."""
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, idx[:n_components]] * np.sqrt(eigenvalues[idx[:n_components]])

def multidimensional_scaling_fit(
    X: np.ndarray,
    n_components: int = 2,
    metric: str = 'euclidean',
    distance_func: Optional[Callable] = None,
    normalization: str = 'standard',
    solver: str = 'closed_form'
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Perform Multidimensional Scaling (MDS) on the input data.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Number of dimensions for the embedding, by default 2.
    metric : str, optional
        Distance metric to use, by default 'euclidean'.
    distance_func : Optional[Callable], optional
        Custom distance function, by default None.
    normalization : str, optional
        Normalization method for the distance matrix, by default 'standard'.
    solver : str, optional
        Solver to use for eigendecomposition, by default 'closed_form'.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict]]
        Dictionary containing the embedding, metrics, parameters used, and warnings.
    """
    _validate_input(X, metric, distance_func)
    D = _compute_distance_matrix(X, metric, distance_func)
    B = _center_data(D, normalization)
    eigenvalues, eigenvectors = _eigendecomposition(B, solver)
    embedding = _compute_embedding(eigenvectors, eigenvalues, n_components)

    return {
        "result": embedding,
        "metrics": {"stress": np.linalg.norm(D - _compute_distance_matrix(embedding, metric))},
        "params_used": {
            "n_components": n_components,
            "metric": metric,
            "normalization": normalization,
            "solver": solver
        },
        "warnings": []
    }

# Example usage:
# embedding = multidimensional_scaling_fit(X=np.random.rand(10, 5), n_components=2)
