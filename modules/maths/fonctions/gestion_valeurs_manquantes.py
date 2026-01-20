"""
Quantix – Module gestion_valeurs_manquantes
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# suppression
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def suppression_fit(
    data: np.ndarray,
    metric: str = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    normalization: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Suppression des valeurs manquantes par imputation basée sur les métriques et distances spécifiées.

    Parameters:
    -----------
    data : np.ndarray
        Matrice des données avec valeurs manquantes (np.nan).
    metric : str, optional
        Métrique pour évaluer la qualité de l'imputation (mse, mae, r2).
    distance : str, optional
        Distance pour calculer la similarité (euclidean, manhattan, cosine).
    solver : str, optional
        Solveur pour l'imputation (closed_form, gradient_descent).
    normalization : str, optional
        Normalisation des données (none, standard, minmax, robust).
    custom_metric : Callable, optional
        Fonction personnalisée pour la métrique.
    custom_distance : Callable, optional
        Fonction personnalisée pour la distance.
    tol : float, optional
        Tolérance pour la convergence.
    max_iter : int, optional
        Nombre maximal d'itérations.

    Returns:
    --------
    dict
        Dictionnaire contenant les résultats, métriques, paramètres utilisés et avertissements.
    """
    # Validation des entrées
    _validate_inputs(data)

    # Normalisation des données
    normalized_data = _normalize_data(data, normalization)

    # Calcul des distances
    distance_matrix = _compute_distance(normalized_data, distance, custom_distance)

    # Imputation des valeurs manquantes
    imputed_data = _impute_missing_values(normalized_data, distance_matrix, solver)

    # Calcul des métriques
    metrics = _compute_metrics(data, imputed_data, metric, custom_metric)

    # Retour des résultats
    return {
        "result": imputed_data,
        "metrics": metrics,
        "params_used": {
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "normalization": normalization
        },
        "warnings": _check_warnings(imputed_data)
    }

def _validate_inputs(data: np.ndarray) -> None:
    """Validation des entrées."""
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy array")
    if np.isnan(data).any():
        pass  # Valeurs manquantes autorisées
    else:
        raise ValueError("No missing values found in the data")

def _normalize_data(data: np.ndarray, method: Optional[str]) -> np.ndarray:
    """Normalisation des données."""
    if method is None or method == 'none':
        return data
    elif method == 'standard':
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        return (data - mean) / std
    elif method == 'minmax':
        min_val = np.nanmin(data, axis=0)
        max_val = np.nanmax(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.nanmedian(data, axis=0)
        iqr = np.nanpercentile(data, 75, axis=0) - np.nanpercentile(data, 25, axis=0)
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_distance(data: np.ndarray, method: str, custom_func: Optional[Callable]) -> np.ndarray:
    """Calcul de la matrice de distances."""
    if custom_func is not None:
        return np.array([[custom_func(x, y) for x in data] for y in data])
    elif method == 'euclidean':
        return np.sqrt(np.sum((data[:, np.newaxis] - data) ** 2, axis=2))
    elif method == 'manhattan':
        return np.sum(np.abs(data[:, np.newaxis] - data), axis=2)
    elif method == 'cosine':
        dot_products = np.dot(data, data.T)
        norms = np.sqrt(np.sum(data ** 2, axis=1))[:, np.newaxis]
        return 1 - dot_products / (norms * norms.T)
    elif method == 'minkowski':
        return np.sum(np.abs(data[:, np.newaxis] - data) ** 3, axis=2) ** (1/3)
    else:
        raise ValueError(f"Unknown distance method: {method}")

def _impute_missing_values(data: np.ndarray, distance_matrix: np.ndarray, solver: str) -> np.ndarray:
    """Imputation des valeurs manquantes."""
    imputed_data = data.copy()
    mask = np.isnan(data)

    if solver == 'closed_form':
        for i in range(imputed_data.shape[0]):
            if np.any(mask[i, :]):
                neighbors = distance_matrix[i]
                weights = 1 / (neighbors + 1e-8)
                weights[np.isinf(weights)] = 0
                weights /= np.sum(weights)
                imputed_data[i, mask[i, :]] = np.dot(data[~mask[i, :], :].T, weights)
    elif solver == 'gradient_descent':
        # Implémentation simplifiée de gradient descent
        pass  # À compléter selon les besoins
    else:
        raise ValueError(f"Unknown solver: {solver}")

    return imputed_data

def _compute_metrics(original: np.ndarray, imputed: np.ndarray, method: str, custom_func: Optional[Callable]) -> Dict[str, float]:
    """Calcul des métriques."""
    mask = np.isnan(original)
    original_filled = original.copy()
    original_filled[mask] = imputed[mask]

    if custom_func is not None:
        return {"custom_metric": custom_func(original_filled, imputed)}
    elif method == 'mse':
        return {"mse": np.mean((original_filled - imputed) ** 2)}
    elif method == 'mae':
        return {"mae": np.mean(np.abs(original_filled - imputed))}
    elif method == 'r2':
        ss_res = np.sum((original_filled - imputed) ** 2)
        ss_tot = np.sum((original_filled - np.nanmean(original)) ** 2)
        return {"r2": 1 - (ss_res / ss_tot)}
    else:
        raise ValueError(f"Unknown metric: {method}")

def _check_warnings(data: np.ndarray) -> Dict[str, str]:
    """Vérification des avertissements."""
    warnings = {}
    if np.any(np.isinf(data)):
        warnings["infinite_values"] = "Infinite values detected in the imputed data"
    if np.any(np.isnan(data)):
        warnings["remaining_nan"] = "Some NaN values remain in the imputed data"
    return warnings

################################################################################
# imputation_moyenne
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for imputation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Input data must be 2-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")

def _normalize_data(
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
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        return (data - mean) / std
    elif method == "minmax":
        min_val = np.nanmin(data, axis=0)
        max_val = np.nanmax(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.nanmedian(data, axis=0)
        iqr = np.nanpercentile(data, 75, axis=0) - np.nanpercentile(data, 25, axis=0)
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_metrics(
    data_original: np.ndarray,
    data_imputed: np.ndarray,
    metrics: Union[str, list, Callable],
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute specified metrics between original and imputed data."""
    if mask is None:
        mask = np.isnan(data_original)

    results = {}

    if isinstance(metrics, str):
        metrics = [metrics]

    for metric in metrics:
        if callable(metric):
            results["custom"] = metric(data_original[mask], data_imputed[mask])
        elif metric == "mse":
            results["mse"] = np.mean((data_original[mask] - data_imputed[mask])**2)
        elif metric == "mae":
            results["mae"] = np.mean(np.abs(data_original[mask] - data_imputed[mask]))
        elif metric == "r2":
            ss_res = np.sum((data_original[mask] - data_imputed[mask])**2)
            ss_tot = np.sum((data_original[mask] - np.mean(data_original[mask]))**2)
            results["r2"] = 1 - (ss_res / (ss_tot + 1e-8))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return results

def _impute_mean(
    data: np.ndarray,
    axis: int = 0
) -> np.ndarray:
    """Impute missing values with column means."""
    imputed = data.copy()
    mask = np.isnan(imputed)
    col_means = np.nanmean(data, axis=axis, keepdims=True)
    imputed[mask] = col_means[mask]
    return imputed

def imputation_moyenne_fit(
    data: np.ndarray,
    normalization_method: str = "standard",
    custom_normalization: Optional[Callable] = None,
    metrics: Union[str, list, Callable] = "mse",
    mask: Optional[np.ndarray] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Impute missing values using mean imputation.

    Parameters:
    -----------
    data : np.ndarray
        Input data array with missing values (NaN)
    normalization_method : str, optional
        Normalization method to apply ("none", "standard", "minmax", "robust")
    custom_normalization : Callable, optional
        Custom normalization function
    metrics : str, list or callable, optional
        Metrics to compute ("mse", "mae", "r2") or custom function
    mask : np.ndarray, optional
        Mask indicating missing values (True for missing)

    Returns:
    --------
    dict
        Dictionary containing:
        - "result": imputed data array
        - "metrics": computed metrics
        - "params_used": parameters used
        - "warnings": any warnings

    Example:
    --------
    >>> data = np.array([[1, 2], [3, np.nan], [np.nan, 6]])
    >>> result = imputation_moyenne_fit(data)
    """
    # Validate input
    _validate_input(data)

    # Normalize data
    normalized_data = _normalize_data(
        data,
        method=normalization_method,
        custom_func=custom_normalization
    )

    # Impute missing values with mean
    imputed_data = _impute_mean(normalized_data)

    # Compute metrics
    computed_metrics = _compute_metrics(
        normalized_data,
        imputed_data,
        metrics,
        mask
    )

    # Return results
    return {
        "result": imputed_data,
        "metrics": computed_metrics,
        "params_used": {
            "normalization_method": normalization_method,
            "custom_normalization": custom_normalization is not None,
            "metrics": metrics
        },
        "warnings": []
    }

################################################################################
# imputation_mediane
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(data: np.ndarray) -> None:
    """Validate input data for imputation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values")

def compute_median_imputation(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute median imputation for missing values."""
    return np.nanmedian(data, axis=axis)

def impute_median_fit(
    data: np.ndarray,
    missing_mask: Optional[np.ndarray] = None,
    median_axis: int = 0,
    custom_median_func: Optional[Callable[[np.ndarray, int], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Fit median imputation model for missing values.

    Parameters:
    -----------
    data : np.ndarray
        Input data with missing values (NaN)
    missing_mask : Optional[np.ndarray], default=None
        Binary mask indicating missing values (1 for missing, 0 otherwise)
    median_axis : int, default=0
        Axis along which to compute the median
    custom_median_func : Optional[Callable], default=None
        Custom function to compute median if provided

    Returns:
    --------
    dict
        Dictionary containing:
        - "result": Imputed data
        - "metrics": Dictionary of metrics
        - "params_used": Dictionary of parameters used
        - "warnings": Dictionary of warnings
    """
    # Validate input data
    validate_input(data)

    # Set default parameters
    params_used = {
        "median_axis": median_axis,
        "custom_median_func": custom_median_func is not None
    }

    # Handle missing mask if provided
    if missing_mask is not None:
        if missing_mask.shape != data.shape:
            raise ValueError("Missing mask must have the same shape as input data")
        if not np.issubdtype(missing_mask.dtype, np.bool_):
            raise TypeError("Missing mask must be boolean")

    # Compute median imputation
    if custom_median_func is not None:
        median = custom_median_func(data, axis=median_axis)
    else:
        median = compute_median_imputation(data, axis=median_axis)

    # Create imputed data
    if missing_mask is not None:
        imputed_data = np.where(missing_mask, median, data)
    else:
        imputed_data = np.where(np.isnan(data), median, data)

    # Calculate metrics
    metrics = {
        "median_value": float(median),
        "missing_values_count": int(np.isnan(data).sum())
    }

    # Check for warnings
    warnings = {}
    if np.isnan(imputed_data).any():
        warnings["remaining_nan"] = "Some NaN values remain after imputation"

    return {
        "result": imputed_data,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

# Example usage:
"""
data = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])
result = imputation_mediane_fit(data)
"""

################################################################################
# imputation_mode
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_input(data: np.ndarray) -> None:
    """Validate input data for imputation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values")

def normalize_data(
    data: np.ndarray,
    method: str = 'none',
    custom_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize data based on specified method."""
    if custom_func is not None:
        return custom_func(data)
    if method == 'standard':
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        return (data - mean) / std
    elif method == 'minmax':
        min_val = np.nanmin(data, axis=0)
        max_val = np.nanmax(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.nanmedian(data, axis=0)
        iqr = np.nanpercentile(data, 75, axis=0) - np.nanpercentile(data, 25, axis=0)
        return (data - median) / (iqr + 1e-8)
    else:
        return data

def compute_mode(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute the mode of each feature."""
    values, counts = np.unique(data, axis=axis, return_counts=True)
    mode_indices = np.argmax(counts, axis=axis)
    modes = values[np.arange(values.shape[0]), mode_indices]
    return modes

def impute_missing_values(
    data: np.ndarray,
    mode_values: np.ndarray
) -> np.ndarray:
    """Impute missing values with the computed modes."""
    masked_data = np.ma.masked_invalid(data)
    imputed_data = masked_data.filled(fill_value=mode_values[np.arange(masked_data.shape[1])])
    return imputed_data

def compute_metrics(
    original: np.ndarray,
    imputed: np.ndarray,
    metric_funcs: Dict[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute specified metrics between original and imputed data."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(original, imputed)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def imputation_mode_fit(
    data: np.ndarray,
    normalize_method: str = 'none',
    custom_normalize: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric_funcs: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_metric_funcs: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None
) -> Dict[str, Any]:
    """
    Perform mode imputation on missing values in the data.

    Parameters:
    -----------
    data : np.ndarray
        Input data array with missing values (NaN or inf).
    normalize_method : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    custom_normalize : Callable, optional
        Custom normalization function.
    metric_funcs : Dict[str, Callable], optional
        Dictionary of metric functions to compute.
    custom_metric_funcs : Dict[str, Callable], optional
        Dictionary of additional metric functions to compute.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate input
    validate_input(data)

    # Set default metric functions if none provided
    if metric_funcs is None:
        metric_funcs = {
            'mse': lambda x, y: np.mean((x - y) ** 2),
            'mae': lambda x, y: np.mean(np.abs(x - y))
        }

    # Combine metric functions
    if custom_metric_funcs is not None:
        metric_funcs.update(custom_metric_funcs)

    # Normalize data
    normalized_data = normalize_data(data, method=normalize_method, custom_func=custom_normalize)

    # Compute mode for each feature
    mode_values = compute_mode(normalized_data, axis=0)

    # Impute missing values
    imputed_data = impute_missing_values(normalized_data, mode_values)

    # Compute metrics
    metrics = compute_metrics(data, imputed_data, metric_funcs)

    # Prepare output
    result = {
        'result': imputed_data,
        'metrics': metrics,
        'params_used': {
            'normalize_method': normalize_method,
            'custom_normalize': custom_normalize is not None,
            'metric_funcs': list(metric_funcs.keys())
        },
        'warnings': []
    }

    return result

################################################################################
# imputation_valeur_fixe
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    data: np.ndarray,
    value_to_impute: float
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if np.isnan(value_to_impute):
        raise ValueError("Imputation value cannot be NaN")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")

def _impute_fixed_value(
    data: np.ndarray,
    value_to_impute: float
) -> np.ndarray:
    """Impute missing values with a fixed value."""
    imputed_data = data.copy()
    mask = np.isnan(imputed_data)
    imputed_data[mask] = value_to_impute
    return imputed_data

def _calculate_metrics(
    original_data: np.ndarray,
    imputed_data: np.ndarray,
    metric_funcs: Dict[str, Callable]
) -> Dict[str, float]:
    """Calculate specified metrics between original and imputed data."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(original_data, imputed_data)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def imputation_valeur_fixe_fit(
    data: np.ndarray,
    value_to_impute: float,
    metric_funcs: Optional[Dict[str, Callable]] = None
) -> Dict:
    """
    Impute missing values with a fixed value and calculate metrics.

    Parameters
    ----------
    data : np.ndarray
        Input data array potentially containing NaN values.
    value_to_impute : float
        Fixed value to use for imputation.
    metric_funcs : Optional[Dict[str, Callable]], default=None
        Dictionary of metric functions to calculate between original and imputed data.
        If None, no metrics are calculated.

    Returns
    -------
    Dict
        Dictionary containing:
        - "result": imputed data array
        - "metrics": calculated metrics (if metric_funcs provided)
        - "params_used": parameters used
        - "warnings": any warnings generated

    Example
    -------
    >>> data = np.array([[1, 2], [3, np.nan]])
    >>> result = imputation_valeur_fixe_fit(data, 0)
    """
    # Validate inputs
    _validate_inputs(data, value_to_impute)

    # Perform imputation
    imputed_data = _impute_fixed_value(data, value_to_impute)

    # Calculate metrics if requested
    metrics = {}
    warnings = []
    if metric_funcs is not None:
        try:
            metrics = _calculate_metrics(data, imputed_data, metric_funcs)
        except Exception as e:
            warnings.append(f"Metric calculation failed: {str(e)}")

    # Prepare output
    return {
        "result": imputed_data,
        "metrics": metrics,
        "params_used": {
            "value_to_impute": value_to_impute
        },
        "warnings": warnings
    }

################################################################################
# imputation_regresion
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def imputation_regresion_fit(
    X: np.ndarray,
    mask: np.ndarray,
    normalizer: Callable = None,
    metric: str = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Impute missing values using regression.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix with missing values (NaN).
    mask : np.ndarray
        Binary mask where 1 indicates missing value.
    normalizer : Callable, optional
        Function to normalize data (e.g., standard, minmax).
    metric : str or Callable
        Metric to evaluate imputation quality.
    distance : str or Callable
        Distance metric for regression.
    solver : str
        Solver algorithm to use.
    regularization : str, optional
        Regularization type (none, l1, l2, elasticnet).
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Callable, optional
        Custom metric function.
    custom_distance : Callable, optional
        Custom distance function.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, mask)

    # Normalize data if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Prepare data for imputation
    X_obs = _prepare_data(X_normalized, mask)

    # Choose solver and fit model
    if solver == 'closed_form':
        imputed_values = _closed_form_solver(X_obs, mask)
    elif solver == 'gradient_descent':
        imputed_values = _gradient_descent_solver(X_obs, mask, tol, max_iter)
    else:
        raise ValueError(f"Solver {solver} not supported.")

    # Apply regularization if specified
    if regularization:
        imputed_values = _apply_regularization(imputed_values, X_normalized, mask, regularization)

    # Calculate metrics
    metrics = _calculate_metrics(X_normalized, imputed_values, mask, metric, custom_metric)

    # Inverse normalization if needed
    X_imputed = _inverse_normalization(X, imputed_values, normalizer)

    return {
        'result': X_imputed,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': _check_warnings(X_imputed)
    }

def _validate_inputs(X: np.ndarray, mask: np.ndarray) -> None:
    """Validate input data and mask."""
    if X.shape != mask.shape:
        raise ValueError("X and mask must have the same shape.")
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError("X must contain numerical values.")
    if not np.issubdtype(mask.dtype, np.bool_):
        raise ValueError("Mask must be boolean.")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to data."""
    if normalizer is None:
        return X
    return normalizer(X)

def _prepare_data(X: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare data for imputation."""
    X_obs = X.copy()
    X_obs[mask] = 0
    return X_obs

def _closed_form_solver(X_obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Closed form solution for regression imputation."""
    # Implement closed form solution
    pass

def _gradient_descent_solver(X_obs: np.ndarray, mask: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
    """Gradient descent solver for regression imputation."""
    # Implement gradient descent
    pass

def _apply_regularization(imputed_values: np.ndarray, X_normalized: np.ndarray,
                         mask: np.ndarray, regularization: str) -> np.ndarray:
    """Apply regularization to imputed values."""
    # Implement regularization
    pass

def _calculate_metrics(X_normalized: np.ndarray, imputed_values: np.ndarray,
                      mask: np.ndarray, metric: str, custom_metric: Optional[Callable]) -> Dict:
    """Calculate metrics for imputation."""
    # Implement metric calculation
    pass

def _inverse_normalization(X: np.ndarray, imputed_values: np.ndarray,
                          normalizer: Optional[Callable]) -> np.ndarray:
    """Inverse normalization of imputed values."""
    if normalizer is None:
        return X
    return normalizer.inverse_transform(imputed_values)

def _check_warnings(X_imputed: np.ndarray) -> List[str]:
    """Check for warnings in imputed data."""
    # Implement warning checks
    pass

################################################################################
# imputation_kplus_proches_voisins
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def validate_input(data: np.ndarray) -> None:
    """Validate input data for imputation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values")

def normalize_data(
    data: np.ndarray,
    method: str = 'standard',
    custom_normalizer: Optional[Callable] = None
) -> np.ndarray:
    """Normalize data using specified method or custom function."""
    if custom_normalizer is not None:
        return custom_normalizer(data)

    if method == 'none':
        return data
    elif method == 'standard':
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        return (data - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.nanmin(data, axis=0)
        max_val = np.nanmax(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.nanmedian(data, axis=0)
        iqr = np.nanpercentile(data, 75, axis=0) - np.nanpercentile(data, 25, axis=0)
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def compute_distance(
    x: np.ndarray,
    y: np.ndarray,
    metric: str = 'euclidean',
    custom_distance: Optional[Callable] = None
) -> np.ndarray:
    """Compute distance between two data points using specified metric or custom function."""
    if custom_distance is not None:
        return custom_distance(x, y)

    if metric == 'euclidean':
        return np.linalg.norm(x - y)
    elif metric == 'manhattan':
        return np.sum(np.abs(x - y))
    elif metric == 'cosine':
        return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    elif metric == 'minkowski':
        return np.sum(np.abs(x - y)**3)**(1/3)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def find_k_nearest_neighbors(
    data: np.ndarray,
    mask: np.ndarray,
    k: int = 5,
    metric: str = 'euclidean',
    custom_distance: Optional[Callable] = None
) -> np.ndarray:
    """Find k nearest neighbors for each missing value."""
    distances = []
    for i in range(data.shape[0]):
        if mask[i].any():
            x = data[i, ~mask[i]]
            dists = np.array([compute_distance(x, data[j, ~mask[j]], metric, custom_distance)
                             for j in range(data.shape[0]) if not np.array_equal(mask[i], mask[j])])
            nearest = np.argsort(dists)[:k]
            distances.append(nearest)
        else:
            distances.append(np.array([]))
    return np.array(distances, dtype=object)

def impute_missing_values(
    data: np.ndarray,
    mask: np.ndarray,
    k_neighbors: np.ndarray,
    method: str = 'mean'
) -> np.ndarray:
    """Impute missing values using specified method."""
    imputed_data = data.copy()
    for i in range(data.shape[0]):
        if mask[i].any():
            neighbors = data[k_neighbors[i]]
            if method == 'mean':
                imputed_data[i, mask[i]] = np.nanmean(neighbors[:, mask[i]], axis=0)
            elif method == 'median':
                imputed_data[i, mask[i]] = np.nanmedian(neighbors[:, mask[i]], axis=0)
            elif method == 'mode':
                imputed_data[i, mask[i]] = np.nanmean(neighbors[:, mask[i]], axis=0)
            else:
                raise ValueError(f"Unknown imputation method: {method}")
    return imputed_data

def compute_metrics(
    original: np.ndarray,
    imputed: np.ndarray,
    mask: np.ndarray,
    metrics: List[str] = ['mse', 'mae']
) -> Dict[str, float]:
    """Compute evaluation metrics for imputation."""
    result = {}
    for metric in metrics:
        if metric == 'mse':
            diff = original[mask] - imputed[mask]
            result['mse'] = np.mean(diff**2)
        elif metric == 'mae':
            diff = original[mask] - imputed[mask]
            result['mae'] = np.mean(np.abs(diff))
        elif metric == 'r2':
            diff = original[mask] - imputed[mask]
            ss_res = np.sum(diff**2)
            ss_tot = np.sum((original[mask] - np.mean(original[mask]))**2)
            result['r2'] = 1 - (ss_res / ss_tot)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    return result

def imputation_kplus_proches_voisins_fit(
    data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    k: int = 5,
    normalization: str = 'standard',
    distance_metric: str = 'euclidean',
    imputation_method: str = 'mean',
    metrics: List[str] = ['mse', 'mae'],
    custom_normalizer: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Impute missing values using k-nearest neighbors method.

    Parameters:
    - data: Input data array with missing values (NaN)
    - mask: Boolean mask indicating missing values (True for missing)
    - k: Number of neighbors to consider
    - normalization: Normalization method ('none', 'standard', 'minmax', 'robust')
    - distance_metric: Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski')
    - imputation_method: Imputation method ('mean', 'median', 'mode')
    - metrics: List of evaluation metrics to compute
    - custom_normalizer: Custom normalization function
    - custom_distance: Custom distance function

    Returns:
    Dictionary containing:
    - result: Imputed data
    - metrics: Evaluation metrics
    - params_used: Parameters used in the computation
    - warnings: Any warnings generated during computation

    Example:
    >>> data = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])
    >>> mask = np.array([[False, False, True], [False, True, False], [False, False, False]])
    >>> result = imputation_kplus_proches_voisins_fit(data, mask)
    """
    # Validate input
    validate_input(data)

    # Create mask if not provided
    if mask is None:
        mask = np.isnan(data)

    # Normalize data
    normalized_data = normalize_data(data, normalization, custom_normalizer)

    # Find k nearest neighbors
    k_neighbors = find_k_nearest_neighbors(normalized_data, mask, k, distance_metric, custom_distance)

    # Impute missing values
    imputed_data = impute_missing_values(normalized_data, mask, k_neighbors, imputation_method)

    # Compute metrics
    evaluation_metrics = compute_metrics(data, imputed_data, mask, metrics)

    # Prepare output
    result = {
        'result': imputed_data,
        'metrics': evaluation_metrics,
        'params_used': {
            'k': k,
            'normalization': normalization,
            'distance_metric': distance_metric,
            'imputation_method': imputation_method
        },
        'warnings': []
    }

    return result

################################################################################
# interpolation_lineaire
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    mask: np.ndarray,
    normalizer: Optional[Callable] = None
) -> None:
    """Validate input data and mask."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if mask.ndim != 1 or mask.shape[0] != X.shape[0]:
        raise ValueError("mask must be a 1D array with same length as X.")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values.")
    if normalizer is not None and not callable(normalizer):
        raise ValueError("normalizer must be a callable or None.")

def _normalize_data(
    X: np.ndarray,
    mask: np.ndarray,
    normalizer: Optional[Callable] = None
) -> np.ndarray:
    """Normalize data based on the provided normalizer."""
    if normalizer is None:
        return X
    return normalizer(X, mask)

def _compute_linear_interpolation(
    X: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """Compute linear interpolation for missing values."""
    n_samples, n_features = X.shape
    complete_mask = ~mask

    # Get indices of known and unknown values
    known_idx = np.where(complete_mask)[0]
    unknown_idx = np.where(mask)[0]

    if len(known_idx) < 2:
        raise ValueError("At least two known values are required for interpolation.")

    # Create design matrix
    A = np.column_stack([np.ones(len(known_idx)), known_idx])
    b = X[known_idx]

    # Solve for coefficients
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Interpolate missing values
    X_interp = np.zeros_like(X)
    X_interp[complete_mask] = X[complete_mask]
    for i in unknown_idx:
        X_interp[i] = coeffs[0] + coeffs[1] * i

    return X_interp

def _compute_metrics(
    X: np.ndarray,
    X_pred: np.ndarray,
    metric_funcs: Dict[str, Callable]
) -> Dict[str, float]:
    """Compute metrics for the interpolation."""
    metrics = {}
    for name, func in metric_funcs.items():
        if name == 'mse':
            metrics[name] = np.mean((X[~np.isnan(X)] - X_pred[~np.isnan(X)])**2)
        elif name == 'mae':
            metrics[name] = np.mean(np.abs(X[~np.isnan(X)] - X_pred[~np.isnan(X)]))
        elif name == 'r2':
            ss_res = np.sum((X[~np.isnan(X)] - X_pred[~np.isnan(X)])**2)
            ss_tot = np.sum((X[~np.isnan(X)] - np.mean(X[~np.isnan(X)]))**2)
            metrics[name] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        elif callable(func):
            metrics[name] = func(X[~np.isnan(X)], X_pred[~np.isnan(X)])
    return metrics

def interpolation_lineaire_fit(
    X: np.ndarray,
    mask: np.ndarray,
    normalizer: Optional[Callable] = None,
    metric_funcs: Dict[str, Union[str, Callable]] = None
) -> Dict:
    """
    Perform linear interpolation for missing values in a dataset.

    Parameters:
    -----------
    X : np.ndarray
        Input data array of shape (n_samples, n_features).
    mask : np.ndarray
        Boolean mask indicating missing values (True for missing).
    normalizer : Optional[Callable], default=None
        Function to normalize the data before interpolation.
    metric_funcs : Dict[str, Union[str, Callable]], default=None
        Dictionary of metrics to compute. Keys are metric names and values are either
        predefined strings ('mse', 'mae', 'r2') or custom callable functions.

    Returns:
    --------
    Dict
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, mask, normalizer)

    # Normalize data if specified
    X_norm = _normalize_data(X, mask, normalizer)

    # Compute linear interpolation
    X_interp = _compute_linear_interpolation(X_norm, mask)

    # Compute metrics if specified
    metrics = {}
    if metric_funcs is not None:
        for name, func in metric_funcs.items():
            if isinstance(func, str):
                if func not in ['mse', 'mae', 'r2']:
                    raise ValueError(f"Unknown metric: {func}")
                metric_funcs[name] = func
            elif not callable(func):
                raise ValueError("Metric functions must be either strings or callables.")
        metrics = _compute_metrics(X_norm, X_interp, metric_funcs)

    # Prepare output
    result = {
        "result": X_interp,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
            "metrics": list(metric_funcs.keys()) if metric_funcs else []
        },
        "warnings": []
    }

    return result

# Example usage:
"""
X = np.array([[1, 2], [np.nan, 5], [3, np.nan], [4, 6]])
mask = np.array([False, True, True, False])
result = interpolation_lineaire_fit(X, mask)
print(result)
"""

################################################################################
# interpolation_polynomiale
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> None:
    """Validate input arrays and parameters."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-dimensional arrays")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if normalizer is not None and not callable(normalizer):
        raise ValueError("normalizer must be a callable or None")

def _normalize_data(
    x: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> tuple:
    """Normalize input data using provided normalizer."""
    if normalizer is None:
        return x, y
    x_norm = normalizer(x)
    y_norm = normalizer(y)
    return x_norm, y_norm

def _denormalize_data(
    x: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> tuple:
    """Denormalize input data using provided normalizer."""
    if normalizer is None:
        return x, y
    inverse_norm = lambda arr: 1 / normalizer(arr) if callable(normalizer) else arr
    x_denorm = inverse_norm(x)
    y_denorm = inverse_norm(y)
    return x_denorm, y_denorm

def _compute_polynomial_coefficients(
    x: np.ndarray,
    y: np.ndarray,
    degree: int
) -> np.ndarray:
    """Compute polynomial coefficients using least squares."""
    X = np.vander(x, degree + 1)
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs

def _evaluate_polynomial(
    x: np.ndarray,
    coeffs: np.ndarray
) -> np.ndarray:
    """Evaluate polynomial at given points."""
    return np.polyval(coeffs, x)

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_funcs: Dict[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute specified metrics between true and predicted values."""
    return {name: func(y_true, y_pred) for name, func in metric_funcs.items()}

def interpolation_polynomiale_fit(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 3,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric_funcs: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict:
    """
    Perform polynomial interpolation to estimate missing values.

    Parameters:
    -----------
    x : np.ndarray
        Input data points (independent variable)
    y : np.ndarray
        Output data points (dependent variable) with possible missing values
    degree : int, optional
        Degree of the polynomial (default: 3)
    normalizer : callable, optional
        Function to normalize data before fitting (default: None)
    metric_funcs : dict, optional
        Dictionary of metric functions to compute (default: None)

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> x = np.array([1, 2, 3, 4])
    >>> y = np.array([1, 4, 9, 16])
    >>> result = interpolation_polynomiale_fit(x, y)
    """
    # Validate inputs
    _validate_inputs(x, y, degree, normalizer)

    # Normalize data if specified
    x_norm, y_norm = _normalize_data(x, y, normalizer)

    # Compute polynomial coefficients
    coeffs = _compute_polynomial_coefficients(x_norm, y_norm, degree)

    # Evaluate polynomial
    y_pred = _evaluate_polynomial(x_norm, coeffs)

    # Denormalize predictions
    _, y_pred_denorm = _denormalize_data(x_norm, y_pred, normalizer)

    # Compute metrics if specified
    metrics = {}
    if metric_funcs is not None:
        metrics = _compute_metrics(y, y_pred_denorm, metric_funcs)

    # Prepare output
    result = {
        "result": y_pred_denorm,
        "metrics": metrics,
        "params_used": {
            "degree": degree,
            "normalizer": normalizer.__name__ if normalizer else None
        },
        "warnings": []
    }

    return result

################################################################################
# imputation_multiple
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List, Any

def validate_input(data: np.ndarray) -> None:
    """Validate input data for imputation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values")

def normalize_data(data: np.ndarray, method: str = 'standard') -> np.ndarray:
    """Normalize data using specified method."""
    if method == 'none':
        return data
    elif method == 'standard':
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        return (data - mean) / std
    elif method == 'minmax':
        min_val = np.nanmin(data, axis=0)
        max_val = np.nanmax(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.nanmedian(data, axis=0)
        iqr = np.nanpercentile(data, 75, axis=0) - np.nanpercentile(data, 25, axis=0)
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: Union[str, Callable]) -> float:
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
        return 1 - (ss_res / ss_tot)
    elif metric == 'logloss':
        return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def impute_missing_values(data: np.ndarray, method: str = 'mean', **kwargs) -> np.ndarray:
    """Impute missing values using specified method."""
    if method == 'mean':
        return np.where(np.isnan(data), np.nanmean(data, axis=0), data)
    elif method == 'median':
        return np.where(np.isnan(data), np.nanmedian(data, axis=0), data)
    elif method == 'knn':
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(**kwargs)
        return imputer.fit_transform(data)
    elif method == 'iterative':
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imputer = IterativeImputer(**kwargs)
        return imputer.fit_transform(data)
    else:
        raise ValueError(f"Unknown imputation method: {method}")

def imputation_multiple_fit(
    data: np.ndarray,
    n_iterations: int = 5,
    normalization: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    imputation_method: str = 'mean',
    **kwargs
) -> Dict[str, Any]:
    """
    Perform multiple imputation on data with missing values.

    Parameters:
    - data: Input data array with missing values (NaN)
    - n_iterations: Number of imputation iterations
    - normalization: Normalization method ('none', 'standard', 'minmax', 'robust')
    - metric: Metric to evaluate imputation quality (str or callable)
    - imputation_method: Method for imputing missing values ('mean', 'median', 'knn', 'iterative')
    - **kwargs: Additional arguments for the imputation method

    Returns:
    - Dictionary containing:
        * 'result': Imputed data array
        * 'metrics': List of metrics for each iteration
        * 'params_used': Parameters used in the imputation
        * 'warnings': List of warnings encountered

    Example:
    >>> data = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])
    >>> result = imputation_multiple_fit(data)
    """
    validate_input(data)

    # Initialize output dictionary
    result_dict = {
        'result': None,
        'metrics': [],
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'imputation_method': imputation_method,
            **kwargs
        },
        'warnings': []
    }

    # Normalize data
    normalized_data = normalize_data(data, normalization)

    # Perform multiple imputation
    for i in range(n_iterations):
        try:
            imputed_data = impute_missing_values(normalized_data, imputation_method, **kwargs)

            # Compute metric (using complete cases)
            mask = ~np.isnan(data)
            if np.any(mask):
                metric_value = compute_metric(data[mask], imputed_data[mask], metric)
            else:
                metric_value = np.nan
                result_dict['warnings'].append("No complete cases available for metric computation")

            result_dict['metrics'].append(metric_value)

        except Exception as e:
            result_dict['warnings'].append(f"Iteration {i+1} failed: {str(e)}")
            continue

    # Use the last imputed data as result
    result_dict['result'] = imputed_data

    return result_dict

################################################################################
# propagation_chaos
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def propagation_chaos_fit(
    data: np.ndarray,
    missing_mask: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    distance_metric: str = 'euclidean',
    solver: str = 'closed_form',
    max_iter: int = 100,
    tol: float = 1e-6,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Fit the propagation chaos method for missing value imputation.

    Parameters:
    -----------
    data : np.ndarray
        Input data array with missing values (NaN).
    missing_mask : np.ndarray
        Boolean mask where True indicates missing values.
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the data. Default is identity.
    distance_metric : str
        Distance metric for similarity computation ('euclidean', 'manhattan', etc.).
    solver : str
        Solver method ('closed_form', 'gradient_descent', etc.).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for convergence.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function for evaluation.
    **kwargs
        Additional solver-specific parameters.

    Returns:
    --------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]
        Dictionary containing:
        - "result": Imputed data array.
        - "metrics": Dictionary of computed metrics.
        - "params_used": Dictionary of parameters used.
        - "warnings": List of warnings encountered.

    Example:
    --------
    >>> data = np.array([[1, 2], [3, np.nan], [np.nan, 6]])
    >>> mask = np.array([[False, False], [False, True], [True, False]])
    >>> result = propagation_chaos_fit(data, mask)
    """
    # Validate inputs
    _validate_inputs(data, missing_mask)

    # Normalize data
    normalized_data = normalizer(data.copy())

    # Initialize parameters
    params_used = {
        'normalizer': str(normalizer),
        'distance_metric': distance_metric,
        'solver': solver,
        'max_iter': max_iter,
        'tol': tol
    }

    # Choose distance metric function
    distance_func = _get_distance_function(distance_metric)

    # Solve for missing values
    imputed_data = _solve_missing_values(
        normalized_data,
        missing_mask,
        distance_func,
        solver,
        max_iter,
        tol,
        **kwargs
    )

    # Denormalize data
    imputed_data = normalizer(imputed_data)

    # Compute metrics
    metrics = _compute_metrics(data, imputed_data, custom_metric)

    # Prepare output
    result = {
        "result": imputed_data,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": []
    }

    return result

def _validate_inputs(data: np.ndarray, missing_mask: np.ndarray) -> None:
    """Validate input data and mask."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if not isinstance(missing_mask, np.ndarray):
        raise TypeError("Missing mask must be a numpy array.")
    if data.shape != missing_mask.shape:
        raise ValueError("Data and mask must have the same shape.")
    if np.any(missing_mask) and np.isnan(data[missing_mask]).any():
        raise ValueError("Data contains NaN values where mask indicates missing.")

def _get_distance_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return the distance function based on the metric name."""
    metrics = {
        'euclidean': lambda x, y: np.linalg.norm(x - y),
        'manhattan': lambda x, y: np.sum(np.abs(x - y)),
        'cosine': lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    }
    if metric not in metrics:
        raise ValueError(f"Unsupported distance metric: {metric}")
    return metrics[metric]

def _solve_missing_values(
    data: np.ndarray,
    missing_mask: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    solver: str,
    max_iter: int,
    tol: float,
    **kwargs
) -> np.ndarray:
    """Solve for missing values using the specified solver."""
    if solver == 'closed_form':
        return _solve_closed_form(data, missing_mask)
    elif solver == 'gradient_descent':
        return _solve_gradient_descent(data, missing_mask, distance_func, max_iter, tol, **kwargs)
    else:
        raise ValueError(f"Unsupported solver: {solver}")

def _solve_closed_form(data: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
    """Solve for missing values using closed-form solution."""
    # Implement closed-form solution logic
    imputed_data = data.copy()
    # Placeholder for actual implementation
    return imputed_data

def _solve_gradient_descent(
    data: np.ndarray,
    missing_mask: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    max_iter: int,
    tol: float,
    **kwargs
) -> np.ndarray:
    """Solve for missing values using gradient descent."""
    # Implement gradient descent logic
    imputed_data = data.copy()
    # Placeholder for actual implementation
    return imputed_data

def _compute_metrics(
    original_data: np.ndarray,
    imputed_data: np.ndarray,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Compute metrics for the imputed data."""
    metrics = {}

    if custom_metric is not None:
        metrics['custom'] = custom_metric(original_data, imputed_data)

    # Add other default metrics as needed
    return metrics

################################################################################
# model_based_imputation
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def model_based_imputation_fit(
    data: np.ndarray,
    mask: np.ndarray,
    model_func: Callable,
    normalize: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_weights: Optional[np.ndarray] = None
) -> Dict:
    """
    Fit a model-based imputation method to complete missing values in the data.

    Parameters:
    -----------
    data : np.ndarray
        Input data array with missing values (NaN).
    mask : np.ndarray
        Binary mask indicating missing values (1 for missing, 0 otherwise).
    model_func : Callable
        Function that returns the imputation model.
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : Union[str, Callable], optional
        Metric for evaluation ('mse', 'mae', 'r2', 'logloss') or custom callable.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    regularization : Optional[str], optional
        Regularization method ('none', 'l1', 'l2', 'elasticnet').
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_weights : Optional[np.ndarray], optional
        Custom weights for the data points.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, mask)

    # Normalize data if required
    normalized_data = _apply_normalization(data, normalize)

    # Initialize model and fit
    imputed_data = _fit_model(
        normalized_data,
        mask,
        model_func,
        solver,
        regularization,
        tol,
        max_iter,
        custom_weights
    )

    # Calculate metrics
    metrics = _calculate_metrics(imputed_data, normalized_data, mask, metric)

    # Inverse normalization if required
    result = _inverse_normalization(imputed_data, data, normalize)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalize': normalize,
            'metric': metric,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': _check_warnings(result)
    }

def _validate_inputs(data: np.ndarray, mask: np.ndarray) -> None:
    """Validate input data and mask."""
    if not isinstance(data, np.ndarray) or not isinstance(mask, np.ndarray):
        raise ValueError("Data and mask must be numpy arrays.")
    if data.shape != mask.shape:
        raise ValueError("Data and mask must have the same shape.")
    if np.any(mask < 0) or np.any(mask > 1):
        raise ValueError("Mask must be binary (0 or 1).")
    if np.isnan(data[mask == 0]).any():
        raise ValueError("Data must not contain NaN values where mask is 0.")

def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to the data."""
    if method == 'none':
        return data
    elif method == 'standard':
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        return (data - mean) / std
    elif method == 'minmax':
        min_val = np.nanmin(data, axis=0)
        max_val = np.nanmax(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.nanmedian(data, axis=0)
        iqr = np.nanpercentile(data, 75, axis=0) - np.nanpercentile(data, 25, axis=0)
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _fit_model(
    data: np.ndarray,
    mask: np.ndarray,
    model_func: Callable,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    custom_weights: Optional[np.ndarray]
) -> np.ndarray:
    """Fit the imputation model to the data."""
    observed_data = data[mask == 0]
    missing_mask = mask == 1

    model = model_func()
    if solver == 'closed_form':
        imputed_data = _solve_closed_form(model, observed_data, missing_mask)
    elif solver == 'gradient_descent':
        imputed_data = _solve_gradient_descent(
            model, observed_data, missing_mask, tol, max_iter, custom_weights
        )
    elif solver == 'newton':
        imputed_data = _solve_newton(
            model, observed_data, missing_mask, tol, max_iter, custom_weights
        )
    elif solver == 'coordinate_descent':
        imputed_data = _solve_coordinate_descent(
            model, observed_data, missing_mask, tol, max_iter, custom_weights
        )
    else:
        raise ValueError(f"Unknown solver method: {solver}")

    if regularization is not None:
        imputed_data = _apply_regularization(imputed_data, regularization)

    return imputed_data

def _solve_closed_form(model: object, observed_data: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
    """Solve the imputation problem using closed-form solution."""
    # Placeholder for actual implementation
    return np.zeros_like(observed_data)

def _solve_gradient_descent(
    model: object,
    observed_data: np.ndarray,
    missing_mask: np.ndarray,
    tol: float,
    max_iter: int,
    custom_weights: Optional[np.ndarray]
) -> np.ndarray:
    """Solve the imputation problem using gradient descent."""
    # Placeholder for actual implementation
    return np.zeros_like(observed_data)

def _solve_newton(
    model: object,
    observed_data: np.ndarray,
    missing_mask: np.ndarray,
    tol: float,
    max_iter: int,
    custom_weights: Optional[np.ndarray]
) -> np.ndarray:
    """Solve the imputation problem using Newton's method."""
    # Placeholder for actual implementation
    return np.zeros_like(observed_data)

def _solve_coordinate_descent(
    model: object,
    observed_data: np.ndarray,
    missing_mask: np.ndarray,
    tol: float,
    max_iter: int,
    custom_weights: Optional[np.ndarray]
) -> np.ndarray:
    """Solve the imputation problem using coordinate descent."""
    # Placeholder for actual implementation
    return np.zeros_like(observed_data)

def _apply_regularization(data: np.ndarray, method: str) -> np.ndarray:
    """Apply regularization to the data."""
    if method == 'none':
        return data
    elif method == 'l1':
        return np.sign(data) * np.maximum(np.abs(data) - 0.1, 0)
    elif method == 'l2':
        return data / (1 + np.linalg.norm(data))
    elif method == 'elasticnet':
        return 0.5 * np.sign(data) * np.maximum(np.abs(data) - 0.1, 0) + 0.5 * data / (1 + np.linalg.norm(data))
    else:
        raise ValueError(f"Unknown regularization method: {method}")

def _calculate_metrics(
    imputed_data: np.ndarray,
    original_data: np.ndarray,
    mask: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Calculate metrics for the imputed data."""
    observed_data = original_data[mask == 0]
    imputed_observed = imputed_data[mask == 0]

    if callable(metric):
        return {'custom_metric': metric(imputed_observed, observed_data)}
    elif metric == 'mse':
        return {'mse': np.mean((imputed_observed - observed_data) ** 2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(imputed_observed - observed_data))}
    elif metric == 'r2':
        ss_res = np.sum((imputed_observed - observed_data) ** 2)
        ss_tot = np.sum((observed_data - np.mean(observed_data)) ** 2)
        return {'r2': 1 - ss_res / (ss_tot + 1e-8)}
    elif metric == 'logloss':
        return {'logloss': -np.mean(observed_data * np.log(imputed_observed + 1e-8) + (1 - observed_data) * np.log(1 - imputed_observed + 1e-8))}
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _inverse_normalization(data: np.ndarray, original_data: np.ndarray, method: str) -> np.ndarray:
    """Inverse normalization of the data."""
    if method == 'none':
        return data
    elif method == 'standard':
        mean = np.nanmean(original_data, axis=0)
        std = np.nanstd(original_data, axis=0)
        return data * std + mean
    elif method == 'minmax':
        min_val = np.nanmin(original_data, axis=0)
        max_val = np.nanmax(original_data, axis=0)
        return data * (max_val - min_val + 1e-8) + min_val
    elif method == 'robust':
        median = np.nanmedian(original_data, axis=0)
        iqr = np.nanpercentile(original_data, 75, axis=0) - np.nanpercentile(original_data, 25, axis=0)
        return data * (iqr + 1e-8) + median
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _check_warnings(data: np.ndarray) -> list:
    """Check for warnings in the imputed data."""
    warnings = []
    if np.isnan(data).any():
        warnings.append("NaN values detected in the imputed data.")
    if np.isinf(data).any():
        warnings.append("Infinite values detected in the imputed data.")
    return warnings

################################################################################
# imputation_machine_learning
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def imputation_machine_learning_fit(
    X: np.ndarray,
    imputer: Callable[[np.ndarray], np.ndarray],
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Impute missing values in a dataset using machine learning techniques.

    Parameters:
    -----------
    X : np.ndarray
        Input data with missing values (NaN).
    imputer : Callable[[np.ndarray], np.ndarray]
        Function to perform the actual imputation.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]]
        Function to normalize the data, None for no normalization.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Metric to evaluate the imputation quality.
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Distance metric for the solver.
    solver : str
        Solver to use for optimization.
    regularization : Optional[str]
        Type of regularization to apply.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    random_state : Optional[int]
        Random seed for reproducibility.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the imputed data, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X)

    # Normalize data if specified
    X_normalized = normalizer(X) if normalizer else X.copy()

    # Initialize parameters
    params_used = {
        'normalizer': normalizer.__name__ if normalizer else None,
        'metric': metric,
        'distance': distance,
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'max_iter': max_iter
    }

    # Perform imputation
    X_imputed = imputer(X_normalized)

    # Calculate metrics
    metrics = _calculate_metrics(X_imputed, X_normalized, metric)

    # Check for warnings
    warnings = _check_warnings(X_imputed, X_normalized)

    return {
        'result': X_imputed,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

def _validate_inputs(X: np.ndarray) -> None:
    """
    Validate the input data.

    Parameters:
    -----------
    X : np.ndarray
        Input data to validate.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if np.isnan(X).any():
        raise ValueError("Input data contains NaN values.")
    if np.isinf(X).any():
        raise ValueError("Input data contains infinite values.")

def _calculate_metrics(
    X_imputed: np.ndarray,
    X_original: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """
    Calculate the metrics for the imputed data.

    Parameters:
    -----------
    X_imputed : np.ndarray
        Imputed data.
    X_original : np.ndarray
        Original data (normalized).
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Metric to calculate.

    Returns:
    --------
    Dict[str, float]
        Dictionary of calculated metrics.
    """
    if callable(metric):
        return {'custom_metric': metric(X_imputed, X_original)}

    metrics = {}
    if metric == 'mse':
        metrics['mse'] = np.mean((X_imputed - X_original) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(X_imputed - X_original))
    elif metric == 'r2':
        ss_res = np.sum((X_imputed - X_original) ** 2)
        ss_tot = np.sum((X_original - np.mean(X_original)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)

    return metrics

def _check_warnings(
    X_imputed: np.ndarray,
    X_original: np.ndarray
) -> List[str]:
    """
    Check for warnings in the imputed data.

    Parameters:
    -----------
    X_imputed : np.ndarray
        Imputed data.
    X_original : np.ndarray
        Original data (normalized).

    Returns:
    --------
    List[str]
        List of warning messages.
    """
    warnings = []
    if np.any(np.isnan(X_imputed)):
        warnings.append("Imputed data contains NaN values.")
    if np.any(np.isinf(X_imputed)):
        warnings.append("Imputed data contains infinite values.")

    return warnings

# Example usage:
"""
X = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])
imputer = lambda x: np.nanmean(x, axis=0)
result = imputation_machine_learning_fit(X, imputer, normalizer=None, metric='mse')
"""

################################################################################
# imputation_advanced
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Any

def validate_input(data: np.ndarray) -> None:
    """Validate input data for imputation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values")

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
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        return (data - mean) / std
    elif method == "minmax":
        min_val = np.nanmin(data, axis=0)
        max_val = np.nanmax(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.nanmedian(data, axis=0)
        iqr = np.nanpercentile(data, 75, axis=0) - np.nanpercentile(data, 25, axis=0)
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Union[str, list, Callable],
    custom_metric_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Compute specified metrics between true and predicted values."""
    metric_results = {}

    if isinstance(metrics, str):
        metrics = [metrics]

    for metric in metrics:
        if callable(metric):
            metric_results["custom"] = metric(y_true, y_pred)
        elif metric == "mse":
            metric_results["mse"] = np.mean((y_true - y_pred) ** 2)
        elif metric == "mae":
            metric_results["mae"] = np.mean(np.abs(y_true - y_pred))
        elif metric == "r2":
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metric_results["r2"] = 1 - (ss_res / (ss_tot + 1e-8))
        elif metric == "logloss":
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            metric_results["logloss"] = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    if custom_metric_func is not None:
        metric_results["custom"] = custom_metric_func(y_true, y_pred)

    return metric_results

def impute_values(
    data: np.ndarray,
    method: str = "mean",
    solver: str = "closed_form",
    distance_metric: str = "euclidean",
    regularization: Optional[str] = None,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> np.ndarray:
    """Impute missing values using specified method and solver."""
    if method == "mean":
        return np.nanmean(data, axis=0)
    elif method == "median":
        return np.nanmedian(data, axis=0)
    elif method == "knn":
        # Placeholder for KNN imputation
        return np.zeros_like(data)
    elif method == "mice":
        # Placeholder for MICE imputation
        return np.zeros_like(data)
    else:
        raise ValueError(f"Unknown imputation method: {method}")

def imputation_advanced_fit(
    data: np.ndarray,
    normalization_method: str = "standard",
    metrics: Union[str, list] = ["mse"],
    solver: str = "closed_form",
    distance_metric: str = "euclidean",
    regularization: Optional[str] = None,
    max_iter: int = 1000,
    tol: float = 1e-4,
    custom_normalization_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_metric_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Advanced imputation function with configurable parameters.

    Parameters:
    - data: Input data array with missing values
    - normalization_method: Normalization method to use
    - metrics: Metrics to compute for evaluation
    - solver: Solver method to use
    - distance_metric: Distance metric for imputation methods that require it
    - regularization: Regularization method to use
    - max_iter: Maximum number of iterations for iterative solvers
    - tol: Tolerance for convergence
    - custom_normalization_func: Custom normalization function
    - custom_metric_func: Custom metric function

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate input data
    validate_input(data)

    # Normalize data
    normalized_data = normalize_data(
        data,
        method=normalization_method,
        custom_func=custom_normalization_func
    )

    # Impute missing values
    imputed_data = impute_values(
        normalized_data,
        method="mean",  # This would be configurable in a real implementation
        solver=solver,
        distance_metric=distance_metric,
        regularization=regularization,
        max_iter=max_iter,
        tol=tol
    )

    # Compute metrics (placeholder - would need true values for proper evaluation)
    metrics_results = compute_metrics(
        normalized_data,
        imputed_data,
        metrics=metrics,
        custom_metric_func=custom_metric_func
    )

    # Prepare results dictionary
    result = {
        "result": imputed_data,
        "metrics": metrics_results,
        "params_used": {
            "normalization_method": normalization_method,
            "metrics": metrics,
            "solver": solver,
            "distance_metric": distance_metric,
            "regularization": regularization,
            "max_iter": max_iter,
            "tol": tol
        },
        "warnings": []
    }

    return result

# Example usage:
"""
data = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])
result = imputation_advanced_fit(
    data,
    normalization_method="standard",
    metrics=["mse", "mae"],
    solver="closed_form"
)
"""
