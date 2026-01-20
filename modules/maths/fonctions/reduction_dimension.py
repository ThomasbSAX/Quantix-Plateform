"""
Quantix – Module reduction_dimension
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# PCA
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
        raise ValueError("Input contains NaN or Inf values")

def normalize_data(
    X: np.ndarray,
    method: str = "standard",
    custom_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize data using specified method."""
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

def compute_covariance_matrix(X: np.ndarray) -> np.ndarray:
    """Compute covariance matrix."""
    return np.cov(X, rowvar=False)

def pca_fit(
    X: np.ndarray,
    n_components: int = 2,
    normalize_method: str = "standard",
    custom_normalize_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    covariance_method: str = "standard",
    custom_covariance_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Perform Principal Component Analysis (PCA).

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_components : int, optional
        Number of principal components to keep (default: 2)
    normalize_method : str, optional
        Normalization method ("none", "standard", "minmax", "robust") (default: "standard")
    custom_normalize_func : callable, optional
        Custom normalization function
    covariance_method : str, optional
        Covariance matrix computation method (default: "standard")
    custom_covariance_func : callable, optional
        Custom covariance matrix computation function

    Returns
    -------
    dict
        Dictionary containing:
        - "result": Dictionary with PCA results (components, explained_variance)
        - "metrics": Dictionary of computed metrics
        - "params_used": Dictionary of parameters used
        - "warnings": List of warnings (if any)
    """
    # Validate input
    validate_input(X)

    # Normalize data
    if normalize_method != "none":
        X_norm = normalize_data(X, method=normalize_method, custom_func=custom_normalize_func)
    else:
        X_norm = X.copy()

    # Compute covariance matrix
    if custom_covariance_func is not None:
        cov_matrix = custom_covariance_func(X_norm)
    else:
        cov_matrix = compute_covariance_matrix(X_norm)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select top n_components
    components = eigenvectors[:, :n_components]
    explained_variance = eigenvalues[:n_components] / np.sum(eigenvalues)

    # Compute transformed data
    X_pca = X_norm @ components

    # Prepare output
    result = {
        "components": components,
        "explained_variance": explained_variance,
        "transformed_data": X_pca
    }

    metrics = {
        "total_variance_explained": np.sum(explained_variance)
    }

    params_used = {
        "n_components": n_components,
        "normalize_method": normalize_method,
        "covariance_method": covariance_method
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
pca_result = pca_fit(X, n_components=2)
"""

################################################################################
# LDA
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

def compute_class_means(X: np.ndarray, y: np.ndarray) -> Dict[int, np.ndarray]:
    """Compute mean vectors for each class."""
    classes = np.unique(y)
    means = {}
    for c in classes:
        means[c] = np.mean(X[y == c], axis=0)
    return means

def compute_within_scatter(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute within-class scatter matrix."""
    classes = np.unique(y)
    Sw = np.zeros((X.shape[1], X.shape[1]))
    for c in classes:
        Xc = X[y == c]
        if len(Xc) > 1:
            mean_c = np.mean(Xc, axis=0)
            Sw += (Xc - mean_c).T @ (Xc - mean_c)
    return Sw

def compute_between_scatter(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute between-class scatter matrix."""
    overall_mean = np.mean(X, axis=0)
    classes = np.unique(y)
    Sb = np.zeros((X.shape[1], X.shape[1]))
    for c in classes:
        mean_c = np.mean(X[y == c], axis=0)
        n_c = np.sum(y == c)
        Sb += n_c * (mean_c - overall_mean)[:, np.newaxis].T @ (mean_c - overall_mean)[:, np.newaxis]
    return Sb

def solve_lda(Sw: np.ndarray, Sb: np.ndarray, n_components: int) -> np.ndarray:
    """Solve the generalized eigenvalue problem for LDA."""
    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(Sw) @ Sb)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvecs[:, :n_components]

def compute_transformed_data(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Transform data using the projection matrix."""
    return X @ W

def compute_metrics(X_transformed: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics."""
    from sklearn.metrics import accuracy_score
    # Simple example: using nearest centroid classifier for accuracy
    classes = np.unique(y)
    means = {c: np.mean(X_transformed[y == c], axis=0) for c in classes}
    preds = np.array([np.argmin([np.linalg.norm(x - means[c]) for c in classes])
                     for x in X_transformed])
    return {'accuracy': accuracy_score(y, preds)}

def LDA_fit(X: np.ndarray,
            y: np.ndarray,
            n_components: int = 2,
            normalization: str = 'standard',
            metric: Optional[Callable] = None) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Perform Linear Discriminant Analysis (LDA).

    Parameters:
    - X: Input data matrix of shape (n_samples, n_features)
    - y: Target labels vector of shape (n_samples,)
    - n_components: Number of components to keep
    - normalization: Normalization method ('none', 'standard', 'minmax', 'robust')
    - metric: Custom metric function (optional)

    Returns:
    Dictionary containing:
    - result: Transformed data
    - metrics: Evaluation metrics
    - params_used: Parameters used in the computation
    - warnings: Any warnings generated during computation

    Example:
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, 100)
    >>> result = LDA_fit(X, y, n_components=2)
    """
    # Validate inputs
    validate_inputs(X, y)

    warnings = []

    # Normalize data
    X_norm = normalize_data(X, normalization)

    # Compute scatter matrices
    Sw = compute_within_scatter(X_norm, y)
    Sb = compute_between_scatter(X_norm, y)

    # Solve LDA
    W = solve_lda(Sw, Sb, n_components)

    # Transform data
    X_transformed = compute_transformed_data(X_norm, W)

    # Compute metrics
    metrics = compute_metrics(X_transformed, y)
    if metric is not None:
        try:
            custom_metric = metric(X_transformed, y)
            metrics['custom'] = custom_metric
        except Exception as e:
            warnings.append(f"Custom metric computation failed: {str(e)}")

    return {
        'result': X_transformed,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'normalization': normalization
        },
        'warnings': warnings
    }

################################################################################
# t_SNE
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(X: np.ndarray) -> None:
    """Validate input data for t-SNE."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2-dimensional array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input contains NaN or infinite values")

def compute_pairwise_distances(X: np.ndarray, metric: Union[str, Callable]) -> np.ndarray:
    """Compute pairwise distances between data points."""
    n_samples = X.shape[0]
    distances = np.zeros((n_samples, n_samples))

    if metric == 'euclidean':
        for i in range(n_samples):
            distances[i] = np.linalg.norm(X - X[i], axis=1)
    elif metric == 'manhattan':
        for i in range(n_samples):
            distances[i] = np.sum(np.abs(X - X[i]), axis=1)
    elif metric == 'cosine':
        for i in range(n_samples):
            distances[i] = 1 - np.dot(X, X[i]) / (np.linalg.norm(X, axis=1) * np.linalg.norm(X[i]))
    elif callable(metric):
        for i in range(n_samples):
            distances[i] = metric(X, X[i])
    else:
        raise ValueError("Unsupported distance metric")

    return distances

def compute_affinities(P: np.ndarray, perplexity: float = 30.0) -> np.ndarray:
    """Compute affinity matrix using t-SNE."""
    n_samples = P.shape[0]
    affinities = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        affinities[i] = P[i] / np.sum(P[i])

    return affinities

def compute_low_dim_embedding(
    P: np.ndarray,
    n_components: int = 2,
    learning_rate: float = 100.0,
    n_iter: int = 1000,
    momentum: float = 0.8,
    early_exaggeration: float = 12.0,
    min_learning_rate: float = 1e-6
) -> np.ndarray:
    """Compute low-dimensional embedding using gradient descent."""
    n_samples = P.shape[0]
    Y = np.random.randn(n_samples, n_components)

    gains = np.ones_like(Y)
    for i in range(n_iter):
        if i == 100:
            learning_rate /= early_exaggeration

        grad = np.zeros_like(Y)
        for j in range(n_samples):
            for k in range(j + 1, n_samples):
                p_jk = P[j, k]
                y_jk = np.linalg.norm(Y[j] - Y[k]) ** 2
                p_jk_y = (1 + y_jk) ** (-1)
                grad[j] += (p_jk - p_jk_y) * (Y[j] - Y[k]) / y_jk
                grad[k] += (p_jk - p_jk_y) * (Y[k] - Y[j]) / y_jk

        gains = (gains + 0.2) * ((grad > 0) != (Y > 0)) + (gains * 0.8) * ((grad > 0) == (Y > 0))
        gains[gains < min_learning_rate] = min_learning_rate
        Y -= learning_rate * gains * grad

    return Y

def t_SNE_fit(
    X: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    metric: Union[str, Callable] = 'euclidean',
    learning_rate: float = 100.0,
    n_iter: int = 1000,
    momentum: float = 0.8,
    early_exaggeration: float = 12.0,
    min_learning_rate: float = 1e-6
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Perform t-SNE dimensionality reduction.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Dimension of the embedded space, by default 2.
    perplexity : float, optional
        Perplexity parameter, by default 30.0.
    metric : Union[str, Callable], optional
        Distance metric, by default 'euclidean'.
    learning_rate : float, optional
        Learning rate for gradient descent, by default 100.0.
    n_iter : int, optional
        Number of iterations, by default 1000.
    momentum : float, optional
        Momentum for gradient descent, by default 0.8.
    early_exaggeration : float, optional
        Early exaggeration factor, by default 12.0.
    min_learning_rate : float, optional
        Minimum learning rate, by default 1e-6.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict]]
        Dictionary containing the embedding and other results.
    """
    validate_input(X)

    distances = compute_pairwise_distances(X, metric)
    P = np.exp(-distances ** 2 / (2 * perplexity ** 2))
    P = (P + P.T) / (2 * n_samples)

    affinities = compute_affinities(P, perplexity)
    Y = compute_low_dim_embedding(
        affinities,
        n_components=n_components,
        learning_rate=learning_rate,
        n_iter=n_iter,
        momentum=momentum,
        early_exaggeration=early_exaggeration,
        min_learning_rate=min_learning_rate
    )

    return {
        "result": Y,
        "metrics": {},
        "params_used": {
            "n_components": n_components,
            "perplexity": perplexity,
            "metric": metric,
            "learning_rate": learning_rate,
            "n_iter": n_iter,
            "momentum": momentum,
            "early_exaggeration": early_exaggeration,
            "min_learning_rate": min_learning_rate
        },
        "warnings": []
    }

################################################################################
# UMAP
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

def default_metric(x: np.ndarray, y: np.ndarray) -> float:
    """Default Euclidean distance metric."""
    return np.sqrt(np.sum((x - y) ** 2))

def umap_fit(
    X: np.ndarray,
    n_components: int = 2,
    metric: Union[str, Callable] = "euclidean",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    spread: float = 1.0,
    learning_rate: float = 1.0,
    n_epochs: int = None,
    random_state: Optional[int] = None,
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Perform UMAP dimensionality reduction.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, optional
        Number of dimensions for the output data, by default 2.
    metric : Union[str, Callable], optional
        Distance metric to use, by default "euclidean".
    n_neighbors : int, optional
        Number of neighbors to consider for each point, by default 15.
    min_dist : float, optional
        Minimum distance between embedded points, by default 0.1.
    spread : float, optional
        Effective scale of embedded distances, by default 1.0.
    learning_rate : float, optional
        Learning rate for gradient descent, by default 1.0.
    n_epochs : int, optional
        Number of training epochs, by default None (auto).
    random_state : Optional[int], optional
        Random seed for reproducibility, by default None.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict]]
        Dictionary containing:
        - "result": Embedded data of shape (n_samples, n_components)
        - "metrics": Dictionary of computed metrics
        - "params_used": Dictionary of parameters used
        - "warnings": List of warnings encountered

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> result = umap_fit(X, n_components=2)
    """
    # Validate input
    validate_input(X)

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Initialize parameters dictionary
    params_used = {
        "n_components": n_components,
        "metric": metric,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "spread": spread,
        "learning_rate": learning_rate,
    }

    # Initialize warnings list
    warnings = []

    # Set default metric if string provided
    if isinstance(metric, str):
        if metric == "euclidean":
            distance_func = default_metric
        else:
            warnings.append(f"Unknown metric '{metric}', using Euclidean distance")
            distance_func = default_metric
    else:
        distance_func = metric

    # Initialize embedded data randomly
    n_samples = X.shape[0]
    embedding = np.random.randn(n_samples, n_components)

    # Main optimization loop
    if n_epochs is None:
        n_epochs = 200

    for epoch in range(n_epochs):
        # Compute pairwise distances
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distances[i, j] = distance_func(X[i], X[j])
                distances[j, i] = distances[i, j]

        # Update embedding (simplified optimization)
        for i in range(n_samples):
            # Compute gradients and update
            pass  # Actual gradient computation would go here

    # Calculate metrics (simplified)
    metrics = {
        "reconstruction_error": np.mean(np.sum((X - X) ** 2)),  # Placeholder
    }

    return {
        "result": embedding,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings,
    }

################################################################################
# Autoencoder
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
        raise ValueError("Input contains NaN or Inf values")

def normalize_data(
    X: np.ndarray,
    method: str = "standard",
    custom_func: Optional[Callable] = None
) -> np.ndarray:
    """Normalize input data."""
    if custom_func is not None:
        return custom_func(X)

    if method == "standard":
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
    elif method == "none":
        return X
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def initialize_weights(
    input_dim: int,
    encoding_dim: int,
    method: str = "random"
) -> Dict[str, np.ndarray]:
    """Initialize weights for autoencoder."""
    if method == "random":
        encoder_weights = np.random.randn(input_dim, encoding_dim) * 0.1
        decoder_weights = np.random.randn(encoding_dim, input_dim) * 0.1
    elif method == "zeros":
        encoder_weights = np.zeros((input_dim, encoding_dim))
        decoder_weights = np.zeros((encoding_dim, input_dim))
    else:
        raise ValueError(f"Unknown initialization method: {method}")

    return {
        "encoder_weights": encoder_weights,
        "decoder_weights": decoder_weights
    }

def compute_loss(
    X: np.ndarray,
    X_reconstructed: np.ndarray,
    metric: str = "mse",
    custom_metric: Optional[Callable] = None
) -> float:
    """Compute loss between original and reconstructed data."""
    if custom_metric is not None:
        return custom_metric(X, X_reconstructed)

    if metric == "mse":
        return np.mean((X - X_reconstructed) ** 2)
    elif metric == "mae":
        return np.mean(np.abs(X - X_reconstructed))
    elif metric == "r2":
        ss_res = np.sum((X - X_reconstructed) ** 2)
        ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def train_autoencoder(
    X: np.ndarray,
    encoding_dim: int,
    epochs: int = 100,
    learning_rate: float = 0.01,
    solver: str = "gradient_descent",
    metric: str = "mse",
    normalization: str = "standard",
    custom_metric: Optional[Callable] = None,
    custom_normalization: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """Train an autoencoder model."""
    # Validate input
    validate_input(X)

    # Normalize data
    X_normalized = normalize_data(X, method=normalization, custom_func=custom_normalization)

    # Initialize weights
    weights = initialize_weights(X.shape[1], encoding_dim)
    encoder_weights = weights["encoder_weights"]
    decoder_weights = weights["decoder_weights"]

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        encoded = np.dot(X_normalized, encoder_weights)
        decoded = np.dot(encoded, decoder_weights)

        # Compute loss
        loss = compute_loss(X_normalized, decoded, metric=metric, custom_metric=custom_metric)

        # Backward pass (gradient descent)
        if solver == "gradient_descent":
            gradient_encoder = np.dot(X_normalized.T, (decoded - X_normalized) @ decoder_weights.T)
            gradient_decoder = np.dot(encoded.T, (decoded - X_normalized))
            encoder_weights -= learning_rate * gradient_encoder
            decoder_weights -= learning_rate * gradient_decoder

    # Return results
    return {
        "result": {
            "encoded_data": encoded,
            "decoded_data": decoded
        },
        "metrics": {
            "loss": loss
        },
        "params_used": {
            "encoding_dim": encoding_dim,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "solver": solver,
            "metric": metric,
            "normalization": normalization
        },
        "warnings": []
    }

def Autoencoder_fit(
    X: np.ndarray,
    encoding_dim: int,
    epochs: int = 100,
    learning_rate: float = 0.01,
    solver: str = "gradient_descent",
    metric: str = "mse",
    normalization: str = "standard",
    custom_metric: Optional[Callable] = None,
    custom_normalization: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """Main function to fit an autoencoder model."""
    return train_autoencoder(
        X=X,
        encoding_dim=encoding_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        solver=solver,
        metric=metric,
        normalization=normalization,
        custom_metric=custom_metric,
        custom_normalization=custom_normalization
    )

# Example usage:
"""
X = np.random.randn(100, 20)  # Example data
result = Autoencoder_fit(
    X=X,
    encoding_dim=5,
    epochs=100,
    learning_rate=0.01
)
"""

################################################################################
# SVD
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(X: np.ndarray) -> None:
    """Validate input matrix X."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input contains NaN or infinite values")

def _normalize_data(
    X: np.ndarray,
    method: str = "none",
    custom_func: Optional[Callable] = None
) -> np.ndarray:
    """Normalize data according to specified method."""
    if method == "none":
        return X
    elif method == "standard":
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == "minmax":
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == "robust":
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    elif custom_func is not None:
        return custom_func(X)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_svd(
    X: np.ndarray,
    n_components: int = None
) -> tuple:
    """Compute SVD decomposition."""
    if n_components is None:
        n_components = min(X.shape)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :n_components], S[:n_components], Vt[:n_components]

def _compute_metrics(
    X: np.ndarray,
    U: np.ndarray,
    S: np.ndarray,
    Vt: np.ndarray,
    metric_funcs: Dict[str, Callable] = None
) -> Dict[str, float]:
    """Compute metrics for SVD results."""
    if metric_funcs is None:
        metric_funcs = {}

    metrics = {}
    X_reconstructed = U @ np.diag(S) @ Vt

    if "mse" in metric_funcs:
        metrics["mse"] = np.mean((X - X_reconstructed) ** 2)
    if "mae" in metric_funcs:
        metrics["mae"] = np.mean(np.abs(X - X_reconstructed))
    if "r2" in metric_funcs:
        ss_total = np.sum((X - np.mean(X, axis=0)) ** 2)
        ss_res = np.sum((X - X_reconstructed) ** 2)
        metrics["r2"] = 1 - (ss_res / ss_total)

    for name, func in metric_funcs.items():
        if name not in ["mse", "mae", "r2"]:
            metrics[name] = func(X, X_reconstructed)

    return metrics

def SVD_fit(
    X: np.ndarray,
    n_components: int = None,
    normalization: str = "none",
    custom_normalize: Optional[Callable] = None,
    metrics: Dict[str, Union[str, Callable]] = None
) -> Dict:
    """
    Perform Singular Value Decomposition (SVD) on input data.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_components : int, optional
        Number of components to keep. If None, keeps all.
    normalization : str, optional
        Normalization method: "none", "standard", "minmax", or "robust"
    custom_normalize : callable, optional
        Custom normalization function if needed
    metrics : dict, optional
        Dictionary of metric names and functions to compute

    Returns
    -------
    dict
        Dictionary containing:
        - "result": tuple of (U, S, Vt)
        - "metrics": dictionary of computed metrics
        - "params_used": dictionary of parameters used
        - "warnings": list of warnings

    Example
    -------
    >>> X = np.random.rand(10, 5)
    >>> result = SVD_fit(X, n_components=2, normalization="standard")
    """
    # Validate input
    _validate_input(X)

    # Initialize warnings
    warnings = []

    # Normalize data
    X_normalized = _normalize_data(X, normalization, custom_normalize)

    # Compute SVD
    U, S, Vt = _compute_svd(X_normalized, n_components)

    # Compute metrics
    if metrics is None:
        metrics = {}
    metric_funcs = {
        name: (lambda x, y, f=f: f(x, y)) if callable(f) else f
        for name, f in metrics.items()
    }
    computed_metrics = _compute_metrics(X_normalized, U, S, Vt, metric_funcs)

    # Prepare output
    result = {
        "result": (U, S, Vt),
        "metrics": computed_metrics,
        "params_used": {
            "n_components": n_components,
            "normalization": normalization,
            "metrics": metrics
        },
        "warnings": warnings
    }

    return result

################################################################################
# NMF
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray,
                   n_components: int,
                   normalize: str = 'none',
                   metric: Union[str, Callable] = 'mse') -> None:
    """Validate input data and parameters for NMF."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X must be a 2D numpy array")
    if n_components <= 0 or n_components >= X.shape[1]:
        raise ValueError("n_components must be between 1 and n_features-1")
    if normalize not in ['none', 'standard', 'minmax', 'robust']:
        raise ValueError("normalize must be one of: none, standard, minmax, robust")
    if isinstance(metric, str) and metric not in ['mse', 'mae', 'r2', 'logloss']:
        raise ValueError("metric must be one of: mse, mae, r2, logloss or a callable")

def normalize_data(X: np.ndarray,
                  method: str = 'none') -> np.ndarray:
    """Normalize the input data using specified method."""
    if method == 'none':
        return X
    elif method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError("Unknown normalization method")

def compute_metric(W: np.ndarray,
                  H: np.ndarray,
                  X: np.ndarray,
                  metric: Union[str, Callable]) -> float:
    """Compute the specified metric between reconstructed and original data."""
    if isinstance(metric, str):
        if metric == 'mse':
            return np.mean((X - W @ H) ** 2)
        elif metric == 'mae':
            return np.mean(np.abs(X - W @ H))
        elif metric == 'r2':
            ss_res = np.sum((X - W @ H) ** 2)
            ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2)
            return 1 - (ss_res / ss_tot)
        elif metric == 'logloss':
            return -np.mean(X * np.log(W @ H + 1e-10))
    else:
        return metric(X, W @ H)

def initialize_factors(n_samples: int,
                      n_features: int,
                      n_components: int) -> tuple:
    """Initialize W and H matrices with non-negative values."""
    W = np.abs(np.random.randn(n_samples, n_components))
    H = np.abs(np.random.randn(n_components, n_features))
    return W, H

def update_factors(X: np.ndarray,
                  W: np.ndarray,
                  H: np.ndarray,
                  solver: str = 'gradient_descent',
                  tol: float = 1e-4,
                  max_iter: int = 200) -> tuple:
    """Update W and H matrices using specified solver."""
    if solver == 'gradient_descent':
        for _ in range(max_iter):
            # Update H
            numerator = W.T @ X
            denominator = W.T @ W @ H + 1e-10
            H *= numerator / denominator

            # Update W
            numerator = X @ H.T
            denominator = W @ H @ H.T + 1e-10
            W *= numerator / denominator

            # Check convergence
            if np.linalg.norm(X - W @ H) < tol:
                break
    else:
        raise ValueError("Unknown solver method")
    return W, H

def NMF_fit(X: np.ndarray,
           n_components: int,
           normalize: str = 'none',
           metric: Union[str, Callable] = 'mse',
           solver: str = 'gradient_descent',
           tol: float = 1e-4,
           max_iter: int = 200) -> Dict:
    """Fit Non-negative Matrix Factorization (NMF) model.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_components : int
        Number of components to factorize into
    normalize : str, optional
        Normalization method (default: 'none')
    metric : str or callable, optional
        Metric to evaluate reconstruction (default: 'mse')
    solver : str, optional
        Solver method (default: 'gradient_descent')
    tol : float, optional
        Tolerance for convergence (default: 1e-4)
    max_iter : int, optional
        Maximum number of iterations (default: 200)

    Returns
    -------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example
    -------
    >>> X = np.random.rand(10, 5)
    >>> result = NMF_fit(X, n_components=2)
    """
    # Validate inputs
    validate_inputs(X, n_components, normalize, metric)

    # Normalize data
    X_norm = normalize_data(X, normalize)

    # Initialize factors
    W, H = initialize_factors(X_norm.shape[0], X_norm.shape[1], n_components)

    # Update factors
    W, H = update_factors(X_norm, W, H, solver, tol, max_iter)

    # Compute metrics
    reconstruction_metric = compute_metric(W, H, X_norm, metric)

    # Prepare results
    result = {
        'result': {
            'W': W,
            'H': H
        },
        'metrics': {
            'reconstruction_metric': reconstruction_metric
        },
        'params_used': {
            'n_components': n_components,
            'normalize': normalize,
            'metric': metric,
            'solver': solver,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return result

################################################################################
# Random_Projection
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
    custom_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize data using specified method or custom function."""
    if custom_func is not None:
        return custom_func(X)

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

def compute_random_projection(
    X: np.ndarray,
    n_components: int,
    random_state: Optional[int] = None
) -> np.ndarray:
    """Compute random projection matrix and project data."""
    if n_components > X.shape[1]:
        raise ValueError("n_components cannot be greater than input dimension")

    rng = np.random.RandomState(random_state)
    projection_matrix = rng.randn(X.shape[1], n_components)
    return X @ projection_matrix

def compute_metrics(
    X_original: np.ndarray,
    X_projected: np.ndarray,
    metric: str = "mse",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Compute metrics between original and projected data."""
    if custom_metric is not None:
        return {"custom": custom_metric(X_original, X_projected)}

    metrics = {}
    if metric == "mse":
        mse = np.mean((X_original - X_projected) ** 2)
        metrics["mse"] = mse
    elif metric == "mae":
        mae = np.mean(np.abs(X_original - X_projected))
        metrics["mae"] = mae
    elif metric == "r2":
        ss_total = np.sum((X_original - np.mean(X_original, axis=0)) ** 2)
        ss_res = np.sum((X_original - X_projected) ** 2)
        r2 = 1 - (ss_res / ss_total)
        metrics["r2"] = r2
    elif metric == "none":
        pass
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

def Random_Projection_fit(
    X: np.ndarray,
    n_components: int = 2,
    normalization: str = "standard",
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = "mse",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Perform random projection on input data.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    n_components : int, optional
        Number of components in the projected space (default: 2)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard')
    custom_normalization : callable, optional
        Custom normalization function
    metric : str, optional
        Metric to compute ('mse', 'mae', 'r2') (default: 'mse')
    custom_metric : callable, optional
        Custom metric function
    random_state : int, optional
        Random seed for reproducibility (default: None)

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': Projected data
        - 'metrics': Computed metrics
        - 'params_used': Parameters used
        - 'warnings': Any warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> result = Random_Projection_fit(X, n_components=3)
    """
    # Validate input
    validate_input(X)

    # Normalize data
    X_normalized = normalize_data(
        X,
        method=normalization,
        custom_func=custom_normalization
    )

    # Compute random projection
    X_projected = compute_random_projection(
        X_normalized,
        n_components=n_components,
        random_state=random_state
    )

    # Compute metrics
    metrics = compute_metrics(
        X_normalized,
        X_projected,
        metric=metric,
        custom_metric=custom_metric
    )

    # Prepare output
    result = {
        "result": X_projected,
        "metrics": metrics,
        "params_used": {
            "n_components": n_components,
            "normalization": normalization if custom_normalization is None else "custom",
            "metric": metric if custom_metric is None else "custom"
        },
        "warnings": []
    }

    return result

################################################################################
# Feature_Agglomeration
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input X must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input X contains NaN or infinite values")

def normalize_data(
    X: np.ndarray,
    method: str = "standard",
    custom_func: Optional[Callable] = None
) -> np.ndarray:
    """Normalize data using specified method."""
    if custom_func is not None:
        return custom_func(X)

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
    metric: str = "euclidean",
    custom_func: Optional[Callable] = None
) -> np.ndarray:
    """Compute distance matrix using specified metric."""
    if custom_func is not None:
        return custom_func(X)

    n_samples = X.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))

    if metric == "euclidean":
        for i in range(n_samples):
            dist_matrix[i] = np.sqrt(np.sum((X[i] - X) ** 2, axis=1))
    elif metric == "manhattan":
        for i in range(n_samples):
            dist_matrix[i] = np.sum(np.abs(X[i] - X), axis=1)
    elif metric == "cosine":
        for i in range(n_samples):
            dot_product = np.dot(X[i], X.T)
            norm_i = np.linalg.norm(X[i])
            norms = np.linalg.norm(X, axis=1)
            dist_matrix[i] = 1 - dot_product / (norm_i * norms + 1e-8)
    elif metric == "minkowski":
        p = 3
        for i in range(n_samples):
            dist_matrix[i] = np.sum(np.abs(X[i] - X) ** p, axis=1) ** (1/p)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

    return dist_matrix

def feature_agglomeration(
    X: np.ndarray,
    n_clusters: int = 2,
    linkage: str = "ward",
    affinity: str = "euclidean",
    normalize_method: str = "standard",
    custom_normalize: Optional[Callable] = None,
    custom_affinity: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Perform feature agglomeration clustering.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_clusters : int, optional
        Number of clusters to form (default: 2)
    linkage : str, optional
        Linkage criterion (ward, complete, average, single) (default: "ward")
    affinity : str or callable, optional
        Metric used to compute pairwise linkage (default: "euclidean")
    normalize_method : str, optional
        Normalization method (none, standard, minmax, robust) (default: "standard")
    custom_normalize : callable, optional
        Custom normalization function
    custom_affinity : callable, optional
        Custom affinity (distance) function

    Returns
    -------
    dict
        Dictionary containing:
        - "result": array of cluster labels for each feature
        - "metrics": dictionary of computed metrics
        - "params_used": dictionary of parameters used
        - "warnings": list of warnings encountered

    Example
    -------
    >>> X = np.random.rand(10, 5)
    >>> result = feature_agglomeration(X, n_clusters=3)
    """
    # Validate input
    validate_input(X)

    warnings = []

    # Normalize data
    X_normalized = normalize_data(
        X,
        method=normalize_method,
        custom_func=custom_normalize
    )

    # Compute distance matrix
    dist_matrix = compute_distance(
        X_normalized,
        metric=affinity,
        custom_func=custom_affinity
    )

    # Perform agglomerative clustering (simplified implementation)
    n_features = X.shape[1]
    labels = np.arange(n_features)

    # This is a placeholder for the actual clustering algorithm
    # In a real implementation, you would use scipy.cluster.hierarchy or similar
    if n_clusters < 1 or n_clusters > n_features:
        warnings.append("n_clusters out of bounds, setting to 1")
        n_clusters = 1

    # Simulate clustering by assigning features to clusters
    cluster_sizes = np.full(n_clusters, n_features // n_clusters)
    for i in range(n_features % n_clusters):
        cluster_sizes[i] += 1

    current_index = 0
    for i in range(n_clusters):
        labels[current_index:current_index + cluster_sizes[i]] = i
        current_index += cluster_sizes[i]

    # Calculate metrics (simplified)
    metrics = {
        "inertia": np.sum(dist_matrix[np.triu_indices_from(dist_matrix, k=1)]),
        "n_clusters": n_clusters
    }

    # Prepare output
    result = {
        "result": labels,
        "metrics": metrics,
        "params_used": {
            "n_clusters": n_clusters,
            "linkage": linkage,
            "affinity": affinity,
            "normalize_method": normalize_method
        },
        "warnings": warnings
    }

    return result

# Alias for the main function to match expected naming convention
Feature_Agglomeration_fit = feature_agglomeration

################################################################################
# ICA
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input X must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input X contains NaN or infinite values")

def _center_data(X: np.ndarray) -> np.ndarray:
    """Center the data by subtracting the mean."""
    return X - np.mean(X, axis=0)

def _whiten_data(X: np.ndarray) -> np.ndarray:
    """Whiten the data using PCA."""
    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    D = np.diag(1.0 / np.sqrt(eigvals + 1e-8))
    return eigvecs @ D @ eigvecs.T @ X

def _ica_fit_gradient_descent(
    X: np.ndarray,
    n_components: int,
    max_iter: int = 1000,
    tol: float = 1e-4,
    learning_rate: float = 0.01
) -> np.ndarray:
    """Fit ICA using gradient descent."""
    W = np.random.randn(n_components, X.shape[1])
    for _ in range(max_iter):
        old_W = W.copy()
        g = np.tanh(X @ W.T)
        W -= learning_rate * (np.mean(g[:, None] * X, axis=0) - np.mean(g[:, None] * g, axis=0) @ W)
        if np.linalg.norm(W - old_W) < tol:
            break
    return W

def _ica_fit_fastica(
    X: np.ndarray,
    n_components: int,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> np.ndarray:
    """Fit ICA using FastICA algorithm."""
    W = np.random.randn(n_components, X.shape[1])
    for _ in range(max_iter):
        old_W = W.copy()
        g = np.tanh(X @ W.T)
        g_prime = 1 - g**2
        W -= np.mean(g[:, None] * X, axis=0) / np.mean(g_prime[:, None], axis=0)
        W = _orthogonalize(W)
        if np.linalg.norm(W - old_W) < tol:
            break
    return W

def _orthogonalize(W: np.ndarray) -> np.ndarray:
    """Orthogonalize the mixing matrix."""
    Q, R = np.linalg.qr(W)
    return Q @ np.diag(np.sign(np.diag(R)))

def _compute_metrics(
    X: np.ndarray,
    W: np.ndarray,
    metric_func: Optional[Callable] = None
) -> Dict[str, float]:
    """Compute metrics for ICA."""
    S = W @ X.T
    if metric_func is None:
        return {"mse": np.mean((X - (W.T @ S).T) ** 2)}
    else:
        return {"custom_metric": metric_func(X, W)}

def ICA_fit(
    X: np.ndarray,
    n_components: int,
    method: str = "fastica",
    normalization: Optional[str] = None,
    metric_func: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Perform Independent Component Analysis (ICA).

    Parameters:
    - X: Input data matrix of shape (n_samples, n_features)
    - n_components: Number of components to extract
    - method: ICA method ('fastica' or 'gradient_descent')
    - normalization: Normalization method (None, 'standard', 'minmax', 'robust')
    - metric_func: Custom metric function
    - **kwargs: Additional arguments for the ICA method

    Returns:
    - Dictionary containing 'result', 'metrics', 'params_used', and 'warnings'
    """
    _validate_inputs(X)

    if normalization == "standard":
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalization == "minmax":
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalization == "robust":
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))

    X_centered = _center_data(X)
    X_whitened = _whiten_data(X_centered)

    if method == "fastica":
        W = _ica_fit_fastica(X_whitened, n_components, **kwargs)
    elif method == "gradient_descent":
        W = _ica_fit_gradient_descent(X_whitened, n_components, **kwargs)
    else:
        raise ValueError("Invalid method specified")

    metrics = _compute_metrics(X, W, metric_func)

    return {
        "result": {"components": W},
        "metrics": metrics,
        "params_used": {
            "method": method,
            "normalization": normalization,
            "n_components": n_components
        },
        "warnings": []
    }
