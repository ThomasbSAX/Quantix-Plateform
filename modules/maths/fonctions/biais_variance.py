"""
Quantix – Module biais_variance
Généré automatiquement
Date: 2026-01-06
"""

################################################################################
# biais_quadratique_moyen
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def biais_quadratique_moyen_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalisation: str = 'standard',
    metrique: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    solveur: str = 'closed_form',
    regul: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metrique: Optional[Callable] = None
) -> Dict:
    """
    Calcule le biais quadratique moyen pour un modèle de régression.

    Parameters
    ----------
    X : np.ndarray
        Matrice des caractéristiques (n_samples, n_features).
    y : np.ndarray
        Vecteur cible (n_samples,).
    normalisation : str, optional
        Type de normalisation ('none', 'standard', 'minmax', 'robust').
    metrique : str or callable, optional
        Métrique à utiliser ('mse', 'mae', 'r2') ou fonction personnalisée.
    distance : str, optional
        Distance à utiliser ('euclidean', 'manhattan', 'cosine').
    solveur : str, optional
        Solveur à utiliser ('closed_form', 'gradient_descent').
    regul : str, optional
        Type de régularisation ('none', 'l1', 'l2', 'elasticnet').
    tol : float, optional
        Tolérance pour la convergence.
    max_iter : int, optional
        Nombre maximal d'itérations.
    custom_metrique : callable, optional
        Fonction personnalisée pour le calcul de la métrique.

    Returns
    -------
    Dict
        Dictionnaire contenant les résultats, métriques, paramètres utilisés et avertissements.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = biais_quadratique_moyen_fit(X, y)
    """
    # Validation des entrées
    _validate_inputs(X, y)

    # Normalisation
    X_norm = _apply_normalisation(X, normalisation)
    y_norm = _apply_normalisation(y.reshape(-1, 1), normalisation).flatten()

    # Choix de la métrique
    if callable(metrique):
        metrique_func = metrique
    else:
        metrique_func = _get_metrique(metrique)

    # Choix du solveur
    if solveur == 'closed_form':
        params = _solve_closed_form(X_norm, y_norm)
    elif solveur == 'gradient_descent':
        params = _solve_gradient_descent(X_norm, y_norm, tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Solveur non pris en charge: {solveur}")

    # Calcul des prédictions
    y_pred = X_norm @ params

    # Calcul de la métrique
    metrics = {
        'metrique': metrique_func(y_norm, y_pred),
        'mse': _mean_squared_error(y_norm, y_pred),
        'mae': _mean_absolute_error(y_norm, y_pred)
    }

    # Calcul du biais quadratique moyen
    bqm = _calculate_bias_variance(y_norm, y_pred)

    return {
        'result': bqm,
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Valide les entrées X et y."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X et y doivent être des tableaux NumPy.")
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X doit être une matrice (n_samples, n_features) et y un vecteur.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("Le nombre d'échantillons dans X et y doit être le même.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("X et y ne doivent pas contenir de valeurs NaN ou inf.")

def _apply_normalisation(data: np.ndarray, normalisation: str) -> np.ndarray:
    """Applique la normalisation spécifiée."""
    if normalisation == 'none':
        return data
    elif normalisation == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif normalisation == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif normalisation == 'robust':
        return (data - np.median(data, axis=0)) / (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0))
    else:
        raise ValueError(f"Normalisation non prise en charge: {normalisation}")

def _get_metrique(metrique: str) -> Callable:
    """Retourne la fonction de métrique spécifiée."""
    if metrique == 'mse':
        return _mean_squared_error
    elif metrique == 'mae':
        return _mean_absolute_error
    elif metrique == 'r2':
        return _r_squared
    else:
        raise ValueError(f"Métrique non prise en charge: {metrique}")

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule l'erreur quadratique moyenne."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule l'erreur absolue moyenne."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule le coefficient de détermination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Résout le problème de régression par la formule fermée."""
    return np.linalg.inv(X.T @ X) @ X.T @ y

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Résout le problème de régression par descente de gradient."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ params - y) / len(y)
        new_params = params - learning_rate * gradient
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params

    return params

def _calculate_bias_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule le biais quadratique moyen."""
    bias = np.mean((np.mean(y_pred) - np.mean(y_true)) ** 2)
    variance = np.mean((y_pred - np.mean(y_pred)) ** 2)
    return bias + variance

################################################################################
# variance_empirique
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def variance_empirique_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: str = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Calculate empirical variance for bias-variance analysis.

    Parameters:
    -----------
    X : np.ndarray
        Input features matrix of shape (n_samples, n_features).
    y : Optional[np.ndarray]
        Target values vector of shape (n_samples,). If None, X is treated as residuals.
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the input data. Default is identity function.
    metric : str
        Metric for variance calculation: 'mse', 'mae', 'r2'. Default is 'mse'.
    distance : str
        Distance metric for variance components: 'euclidean', 'manhattan', 'cosine'. Default is 'euclidean'.
    solver : str
        Solver method: 'closed_form', 'gradient_descent'. Default is 'closed_form'.
    regularization : Optional[str]
        Regularization type: None, 'l1', 'l2'. Default is None.
    tol : float
        Tolerance for convergence. Default is 1e-4.
    max_iter : int
        Maximum number of iterations. Default is 1000.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function. If provided, overrides the `metric` parameter.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm = normalizer(X)
    if y is not None:
        y_norm = normalizer(y.reshape(-1, 1)).flatten()
    else:
        y_norm = None

    # Choose metric
    if custom_metric is not None:
        metric_func = custom_metric
    else:
        metric_func = _get_metric_function(metric)

    # Choose distance
    distance_func = _get_distance_function(distance)

    # Solve for parameters
    params, warnings = _solve_variance(
        X_norm,
        y_norm,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter
    )

    # Calculate variance components
    result = _calculate_variance_components(
        X_norm,
        y_norm,
        params,
        distance_func=distance_func
    )

    # Calculate metrics
    metrics = _calculate_metrics(
        y_norm,
        result['predicted'],
        metric_func=metric_func
    )

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params,
        'warnings': warnings
    }

def _validate_inputs(X: np.ndarray, y: Optional[np.ndarray]) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or (y is not None and np.any(np.isnan(y))):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(X)) or (y is not None and np.any(np.isinf(y))):
        raise ValueError("Input data contains infinite values")

def _get_metric_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return the appropriate metric function based on input string."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _get_distance_function(distance: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return the appropriate distance function based on input string."""
    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance
    }
    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")
    return distances[distance]


def _solve_variance(
    X: np.ndarray,
    y: Optional[np.ndarray],
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> tuple[Dict[str, Any], list]:
    """Solve for variance parameters using the specified solver."""
    warnings = []
    params = {}

    if y is None:
        # For residuals analysis
        params['mean_residual'] = np.mean(X, axis=0)
    else:
        # For target prediction
        if solver == 'closed_form':
            params['coefficients'] = _closed_form_solution(X, y)
        elif solver == 'gradient_descent':
            params['coefficients'], warnings = _gradient_descent(
                X, y, tol=tol, max_iter=max_iter
            )
        else:
            raise ValueError(f"Unknown solver: {solver}")

    if regularization is not None:
        params['regularization'] = _apply_regularization(
            params, regularization
        )

    return params, warnings

def _calculate_variance_components(
    X: np.ndarray,
    y: Optional[np.ndarray],
    params: Dict[str, Any],
    distance_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, Any]:
    """Calculate variance components using the specified distance function."""
    if y is None:
        # For residuals analysis
        mean_residual = params['mean_residual']
        variance = np.var(X - mean_residual, axis=0)
    else:
        # For target prediction
        coefficients = params['coefficients']
        predicted = X @ coefficients
        variance = np.var(predicted - y, axis=0)

    return {
        'variance': variance,
        'predicted': predicted if y is not None else X @ params['mean_residual']
    }

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Calculate metrics using the specified metric function."""
    return {
        'metric_value': metric_func(y_true, y_pred)
    }

# Example minimal usage:
"""
X = np.random.rand(100, 5)
y = np.random.rand(100)

result = variance_empirique_fit(X, y)
print(result['metrics'])
"""

################################################################################
# erreur_quadratique_moyenne
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> None:
    """Validate input arrays."""
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if sample_weight is not None:
        if y_true.shape != sample_weight.shape:
            raise ValueError("y_true and sample_weight must have the same shape")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative values")

def _compute_mse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> float:
    """Compute Mean Squared Error."""
    residuals = y_true - y_pred
    if sample_weight is not None:
        weighted_residuals = residuals * np.sqrt(sample_weight)
    else:
        weighted_residuals = residuals
    return np.mean(weighted_residuals ** 2)

def _compute_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> float:
    """Compute Mean Absolute Error."""
    residuals = np.abs(y_true - y_pred)
    if sample_weight is not None:
        weighted_residuals = residuals * np.sqrt(sample_weight)
    else:
        weighted_residuals = residuals
    return np.mean(weighted_residuals)

def _compute_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> float:
    """Compute R-squared."""
    if sample_weight is not None:
        mean_y = np.average(y_true, weights=sample_weight)
    else:
        mean_y = np.mean(y_true)

    ss_total = np.sum((y_true - mean_y) ** 2)
    if sample_weight is not None:
        ss_res = np.sum((y_true - y_pred) ** 2 * sample_weight)
    else:
        ss_res = np.sum((y_true - y_pred) ** 2)

    return 1 - (ss_res / ss_total)

def erreur_quadratique_moyenne_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "mse",
    sample_weight: Optional[np.ndarray] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Compute error metrics including Mean Squared Error (MSE).

    Parameters
    ----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    metric : str, optional
        Metric to compute ('mse', 'mae', 'r2'). Default is "mse".
    sample_weight : np.ndarray, optional
        Individual weights for each sample. Default is None.
    custom_metric : Callable, optional
        Custom metric function. Must take (y_true, y_pred) as input.

    Returns
    -------
    dict
        Dictionary containing:
        - "result": computed metric value
        - "metrics": dictionary of all computed metrics
        - "params_used": parameters used in computation
        - "warnings": any warnings generated

    Examples
    --------
    >>> y_true = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5, 0.0, 2, 8])
    >>> erreur_quadratique_moyenne_fit(y_true, y_pred)
    """
    _validate_inputs(y_true, y_pred, sample_weight)

    metrics = {}
    warnings = []

    if metric == "mse":
        mse_value = _compute_mse(y_true, y_pred, sample_weight)
        metrics["mse"] = mse_value
    elif metric == "mae":
        mae_value = _compute_mae(y_true, y_pred, sample_weight)
        metrics["mae"] = mae_value
    elif metric == "r2":
        r2_value = _compute_r2(y_true, y_pred, sample_weight)
        metrics["r2"] = r2_value
    else:
        warnings.append(f"Unknown metric '{metric}'. Using MSE as default.")
        mse_value = _compute_mse(y_true, y_pred, sample_weight)
        metrics["mse"] = mse_value

    if custom_metric is not None:
        try:
            custom_value = custom_metric(y_true, y_pred)
            metrics["custom"] = custom_value
        except Exception as e:
            warnings.append(f"Custom metric computation failed: {str(e)}")

    return {
        "result": metrics.get(metric, mse_value),
        "metrics": metrics,
        "params_used": {
            "metric": metric,
            "sample_weight": sample_weight is not None
        },
        "warnings": warnings
    }

################################################################################
# sous_apprentissage
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def sous_apprentissage_fit(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    normalisation: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Fit a model with bias-variance analysis.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    model : Callable
        The model to fit. Must have fit and predict methods.
    normalisation : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or Callable, optional
        Metric to evaluate the model ('mse', 'mae', 'r2', 'logloss').
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    regularization : str, optional
        Regularization method ('none', 'l1', 'l2', 'elasticnet').
    tol : float, optional
        Tolerance for stopping criteria.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : Callable, optional
        Custom metric function if not using built-in metrics.
    **kwargs :
        Additional keyword arguments for the model.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _normalize_data(X, normalisation)

    # Fit model
    fitted_model = _fit_model(
        X_normalized, y, model, solver, regularization,
        tol, max_iter, **kwargs
    )

    # Compute metrics
    metrics = _compute_metrics(
        X_normalized, y, fitted_model, metric,
        custom_metric=custom_metric
    )

    # Prepare results
    result = {
        'result': fitted_model,
        'metrics': metrics,
        'params_used': {
            'normalisation': normalisation,
            'metric': metric,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
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

def _normalize_data(
    X: np.ndarray,
    method: str = 'standard'
) -> np.ndarray:
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

def _fit_model(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    **kwargs
) -> Any:
    """Fit the model using specified solver and regularization."""
    if solver == 'closed_form':
        return _fit_closed_form(X, y, regularization, **kwargs)
    elif solver == 'gradient_descent':
        return _fit_gradient_descent(X, y, tol, max_iter, regularization, **kwargs)
    elif solver == 'newton':
        return _fit_newton(X, y, tol, max_iter, regularization, **kwargs)
    elif solver == 'coordinate_descent':
        return _fit_coordinate_descent(X, y, tol, max_iter, regularization, **kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _fit_closed_form(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str],
    **kwargs
) -> Any:
    """Fit model using closed-form solution."""
    if regularization is None:
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
    elif regularization == 'l2':
        lambda_ = kwargs.get('lambda', 1.0)
        beta = np.linalg.inv(X.T @ X + lambda_ * np.eye(X.shape[1])) @ X.T @ y
    else:
        raise ValueError(f"Unsupported regularization for closed_form: {regularization}")
    return beta

def _fit_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int,
    regularization: Optional[str],
    **kwargs
) -> Any:
    """Fit model using gradient descent."""
    beta = np.zeros(X.shape[1])
    learning_rate = kwargs.get('learning_rate', 0.01)
    for _ in range(max_iter):
        gradient = X.T @ (X @ beta - y)
        if regularization == 'l2':
            lambda_ = kwargs.get('lambda', 1.0)
            gradient += 2 * lambda_ * beta
        beta -= learning_rate * gradient
        if np.linalg.norm(gradient) < tol:
            break
    return beta

def _fit_newton(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int,
    regularization: Optional[str],
    **kwargs
) -> Any:
    """Fit model using Newton's method."""
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        gradient = X.T @ (X @ beta - y)
        hessian = 2 * X.T @ X
        if regularization == 'l2':
            lambda_ = kwargs.get('lambda', 1.0)
            hessian += 2 * lambda_ * np.eye(X.shape[1])
        beta -= np.linalg.inv(hessian) @ gradient
        if np.linalg.norm(gradient) < tol:
            break
    return beta

def _fit_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int,
    regularization: Optional[str],
    **kwargs
) -> Any:
    """Fit model using coordinate descent."""
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        for j in range(X.shape[1]):
            X_j = X[:, j]
            residuals = y - X @ beta + beta[j] * X_j
            if regularization == 'l1':
                lambda_ = kwargs.get('lambda', 1.0)
                beta[j] = np.sign(X_j.T @ residuals) * np.max(
                    [0, np.abs(X_j.T @ residuals) - lambda_]
                ) / (X_j.T @ X_j)
            else:
                beta[j] = (X_j.T @ residuals) / (X_j.T @ X_j)
        if np.linalg.norm(X @ beta - y) < tol:
            break
    return beta

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    model: Any,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """Compute metrics for the model."""
    y_pred = _predict(model, X)
    if custom_metric is not None:
        return {'custom': custom_metric(y, y_pred)}
    if metric == 'mse':
        return {'mse': np.mean((y - y_pred) ** 2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(y - y_pred))}
    elif metric == 'r2':
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return {'r2': 1 - ss_res / (ss_tot + 1e-8)}
    elif metric == 'logloss':
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        return {'logloss': -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))}
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _predict(model: Any, X: np.ndarray) -> np.ndarray:
    """Predict using the fitted model."""
    if hasattr(model, 'predict'):
        return model.predict(X)
    else:
        return X @ model

################################################################################
# sur_apprentissage
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray],
    metric: Callable[[np.ndarray, np.ndarray], float]
) -> None:
    """Validate input data and functions."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input data contains infinite values")

    # Test normalizer
    try:
        _ = normalizer(X.copy())
    except Exception as e:
        raise ValueError(f"Normalizer function failed: {str(e)}")

    # Test metric
    try:
        _ = metric(y.copy(), y.copy())
    except Exception as e:
        raise ValueError(f"Metric function failed: {str(e)}")

def _apply_normalization(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Apply normalization to features and target."""
    X_normalized = normalizer(X)
    y_normalized = normalizer(y.reshape(-1, 1)).flatten()
    return X_normalized, y_normalized

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    custom_metrics: Optional[Dict[str, Callable]] = None
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {
        'primary_metric': metric(y_true, y_pred)
    }

    if custom_metrics:
        for name, func in custom_metrics.items():
            try:
                metrics[name] = func(y_true, y_pred)
            except Exception as e:
                metrics[f"error_{name}"] = f"Metric computation failed: {str(e)}"

    return metrics

def _default_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Default mean squared error metric."""
    return np.mean((y_true - y_pred) ** 2)

def _default_normalizer(X: np.ndarray) -> np.ndarray:
    """Default standard normalizer."""
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def sur_apprentissage_fit(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable[[np.ndarray], np.ndarray],
    normalizer: Callable[[np.ndarray], np.ndarray] = _default_normalizer,
    metric: Callable[[np.ndarray, np.ndarray], float] = _default_mse,
    custom_metrics: Optional[Dict[str, Callable]] = None,
    validation_ratio: float = 0.2
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Estimate overfitting/underfitting using train-test split.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    model : Callable[[np.ndarray], np.ndarray]
        Model function that takes features and returns predictions
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize features/target
    metric : Callable[[np.ndarray, np.ndarray], float]
        Primary evaluation metric function
    custom_metrics : Optional[Dict[str, Callable]]
        Dictionary of additional metrics to compute
    validation_ratio : float
        Ratio of data to use for validation (0-1)

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> model = lambda X: np.mean(X, axis=1)
    >>> result = sur_apprentissage_fit(X_train, y_train, model)
    """
    # Validate inputs
    _validate_inputs(X, y, normalizer, metric)

    # Split data
    n_samples = X.shape[0]
    n_val = int(n_samples * validation_ratio)
    idx = np.random.permutation(n_samples)

    X_train, y_train = X[idx[n_val:]], y[idx[n_val:]]
    X_val, y_val = X[idx[:n_val]], y[idx[:n_val]]

    # Normalize data
    X_train_norm, y_train_norm = _apply_normalization(X_train, y_train, normalizer)
    X_val_norm, y_val_norm = _apply_normalization(X_val, y_val, normalizer)

    # Fit model and make predictions
    try:
        train_pred = model(X_train_norm)
        val_pred = model(X_val_norm)
    except Exception as e:
        return {
            "result": None,
            "metrics": {},
            "params_used": {
                "normalizer": normalizer.__name__,
                "metric": metric.__name__
            },
            "warnings": [f"Model fitting failed: {str(e)}"]
        }

    # Compute metrics
    train_metrics = _compute_metrics(y_train_norm, train_pred, metric, custom_metrics)
    val_metrics = _compute_metrics(y_val_norm, val_pred, metric, custom_metrics)

    return {
        "result": {
            "train_predictions": train_pred,
            "validation_predictions": val_pred
        },
        "metrics": {
            "train": train_metrics,
            "validation": val_metrics
        },
        "params_used": {
            "normalizer": normalizer.__name__,
            "metric": metric.__name__
        },
        "warnings": []
    }

################################################################################
# complexite_modele
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def complexite_modele_fit(
    X: np.ndarray,
    y: np.ndarray,
    model_complexity_range: np.ndarray = np.linspace(0.1, 2, 10),
    metric: Union[str, Callable] = 'mse',
    normalization: str = 'standard',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Estimate model complexity using bias-variance tradeoff analysis.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    model_complexity_range : np.ndarray
        Array of complexity parameters to test
    metric : str or callable
        Metric to evaluate model performance ('mse', 'mae', 'r2')
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    solver : str
        Solver method ('closed_form', 'gradient_descent', 'newton')
    regularization : str or None
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    random_state : int or None
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = complexite_modele_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm = _normalize_data(X, method=normalization)
    y_norm = _normalize_data(y.reshape(-1, 1), method=normalization).flatten()

    # Initialize results dictionary
    results = {
        'result': {},
        'metrics': {},
        'params_used': {
            'model_complexity_range': model_complexity_range,
            'metric': metric,
            'normalization': normalization,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

    # Test each complexity level
    for complexity in model_complexity_range:
        try:
            # Fit model with current complexity
            params = _fit_model(
                X_norm, y_norm,
                complexity=complexity,
                solver=solver,
                regularization=regularization,
                tol=tol,
                max_iter=max_iter,
                random_state=random_state
            )

            # Calculate metrics
            predictions = _predict(X_norm, params)
            current_metrics = _calculate_metrics(y_norm, predictions, metric)

            # Store results
            results['result'][complexity] = params
            results['metrics'][complexity] = current_metrics

        except Exception as e:
            results['warnings'].append(f"Failed at complexity {complexity}: {str(e)}")

    return results

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

def _fit_model(
    X: np.ndarray,
    y: np.ndarray,
    complexity: float,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    random_state: Optional[int]
) -> Dict[str, Any]:
    """Fit model with specified complexity and parameters."""
    if solver == 'closed_form':
        return _fit_closed_form(X, y, complexity, regularization)
    elif solver == 'gradient_descent':
        return _fit_gradient_descent(X, y, complexity, regularization, tol, max_iter, random_state)
    elif solver == 'newton':
        return _fit_newton(X, y, complexity, regularization, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _fit_closed_form(X: np.ndarray, y: np.ndarray, complexity: float, regularization: Optional[str]) -> Dict[str, Any]:
    """Fit model using closed form solution."""
    # Add complexity to feature matrix
    X_complex = _apply_complexity(X, complexity)

    if regularization == 'none':
        params = np.linalg.pinv(X_complex) @ y
    elif regularization == 'l1':
        params = _solve_lasso(X_complex, y)
    elif regularization == 'l2':
        params = _solve_ridge(X_complex, y)
    elif regularization == 'elasticnet':
        params = _solve_elasticnet(X_complex, y)
    else:
        raise ValueError(f"Unknown regularization: {regularization}")

    return {'params': params, 'complexity': complexity}

def _fit_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    complexity: float,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    random_state: Optional[int]
) -> Dict[str, Any]:
    """Fit model using gradient descent."""
    np.random.seed(random_state)
    X_complex = _apply_complexity(X, complexity)

    # Initialize parameters
    n_features = X_complex.shape[1]
    params = np.random.randn(n_features)

    for _ in range(max_iter):
        grad = _compute_gradient(X_complex, y, params, regularization)
        params -= tol * grad

    return {'params': params, 'complexity': complexity}

def _fit_newton(
    X: np.ndarray,
    y: np.ndarray,
    complexity: float,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Fit model using Newton's method."""
    X_complex = _apply_complexity(X, complexity)

    # Initialize parameters
    n_features = X_complex.shape[1]
    params = np.zeros(n_features)

    for _ in range(max_iter):
        grad = _compute_gradient(X_complex, y, params, regularization)
        hess = _compute_hessian(X_complex, regularization)
        params -= np.linalg.pinv(hess) @ grad

    return {'params': params, 'complexity': complexity}

def _apply_complexity(X: np.ndarray, complexity: float) -> np.ndarray:
    """Apply model complexity to feature matrix."""
    # This is a placeholder - actual implementation would depend on the specific model
    return X * complexity

def _compute_gradient(X: np.ndarray, y: np.ndarray, params: np.ndarray, regularization: Optional[str]) -> np.ndarray:
    """Compute gradient for optimization."""
    residuals = X @ params - y
    grad = X.T @ residuals / len(y)

    if regularization == 'l1':
        grad += np.sign(params)
    elif regularization == 'l2':
        grad += params

    return grad

def _compute_hessian(X: np.ndarray, regularization: Optional[str]) -> np.ndarray:
    """Compute Hessian matrix for optimization."""
    hess = X.T @ X / len(X)

    if regularization == 'l2':
        hess += np.eye(hess.shape[0])

    return hess

def _predict(X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Make predictions using fitted model."""
    X_complex = _apply_complexity(X, params['complexity'])
    return X_complex @ params['params']

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    metrics = {}

    if metric == 'mse' or isinstance(metric, str) and callable(getattr(__builtins__, metric, None)):
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    if metric == 'mae' or isinstance(metric, str) and callable(getattr(__builtins__, metric, None)):
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    if metric == 'r2' or isinstance(metric, str) and callable(getattr(__builtins__, metric, None)):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - ss_res / (ss_tot + 1e-8)

    if callable(metric):
        metrics['custom'] = metric(y_true, y_pred)

    return metrics

def _solve_lasso(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve L1-regularized least squares problem."""
    # Placeholder for actual implementation
    return np.linalg.pinv(X) @ y

def _solve_ridge(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve L2-regularized least squares problem."""
    # Placeholder for actual implementation
    return np.linalg.pinv(X) @ y

def _solve_elasticnet(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve elastic net regularized least squares problem."""
    # Placeholder for actual implementation
    return np.linalg.pinv(X) @ y

################################################################################
# bias_variance_decomposition
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def bias_variance_decomposition_fit(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    metric: Union[str, Callable] = 'mse',
    n_bootstrap_samples: int = 100,
    normalize: bool = True,
    random_state: Optional[int] = None
) -> Dict:
    """
    Compute bias-variance decomposition for a given model.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    model : Callable
        A callable that implements fit and predict methods (e.g., sklearn-like model)
    metric : str or Callable
        Metric to use for evaluation ('mse', 'mae', 'r2') or custom callable
    n_bootstrap_samples : int
        Number of bootstrap samples to use for variance estimation
    normalize : bool
        Whether to normalize the features before fitting
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': dict with bias, variance and total error components
        - 'metrics': computed metrics
        - 'params_used': parameters used in computation
        - 'warnings': any warnings encountered

    Example
    -------
    >>> from sklearn.linear_model import LinearRegression
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = bias_variance_decomposition_fit(X, y, LinearRegression())
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set random state if provided
    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    # Normalize features if requested
    X_normalized = _normalize_features(X) if normalize else X

    # Get the true model prediction (average over bootstrap samples)
    y_pred_avg = _compute_bootstrap_predictions(X_normalized, y, model, n_bootstrap_samples, rng)

    # Compute bias and variance components
    result = _compute_bias_variance_components(X_normalized, y, model, y_pred_avg,
                                              n_bootstrap_samples, rng)

    # Compute metrics
    metrics = _compute_metrics(y, y_pred_avg, metric)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'metric': metric,
            'n_bootstrap_samples': n_bootstrap_samples,
            'normalize': normalize
        },
        'warnings': []
    }

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

def _normalize_features(X: np.ndarray) -> np.ndarray:
    """Normalize features using standardization."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / (std + 1e-8)

def _compute_bootstrap_predictions(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    n_samples: int,
    rng: np.random.RandomState
) -> np.ndarray:
    """Compute predictions using bootstrap sampling."""
    y_pred_samples = np.zeros((n_samples, X.shape[0]))

    for i in range(n_samples):
        indices = rng.choice(X.shape[0], size=X.shape[0], replace=True)
        X_sample = X[indices]
        y_sample = y[indices]

        model.fit(X_sample, y_sample)
        y_pred_samples[i] = model.predict(X)

    return np.mean(y_pred_samples, axis=0)

def _compute_bias_variance_components(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    y_pred_avg: np.ndarray,
    n_samples: int,
    rng: np.random.RandomState
) -> Dict:
    """Compute bias and variance components."""
    # Compute true model prediction (average over bootstrap samples)
    y_true_avg = np.mean(y)

    # Compute bias
    bias_squared = (y_pred_avg - y_true_avg) ** 2

    # Compute variance
    y_pred_samples = np.zeros((n_samples, X.shape[0]))
    for i in range(n_samples):
        indices = rng.choice(X.shape[0], size=X.shape[0], replace=True)
        X_sample = X[indices]
        y_sample = y[indices]

        model.fit(X_sample, y_sample)
        y_pred_samples[i] = model.predict(X)

    variance = np.mean((y_pred_samples - y_pred_avg) ** 2, axis=0)

    return {
        'bias_squared': np.mean(bias_squared),
        'variance': np.mean(variance),
        'total_error': np.mean(bias_squared) + np.mean(variance)
    }

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Compute evaluation metrics."""
    if callable(metric):
        return {'custom_metric': metric(y_true, y_pred)}

    metrics = {}
    if metric == 'mse' or (isinstance(metric, str) and any(m in metric for m in ['mse', 'MSE'])):
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    if metric == 'mae' or (isinstance(metric, str) and any(m in metric for m in ['mae', 'MAE'])):
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    if metric == 'r2' or (isinstance(metric, str) and any(m in metric for m in ['r2', 'R2'])):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)

    return metrics

################################################################################
# regularisation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def regularisation_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalisation: str = "standard",
    metric: Union[str, Callable] = "mse",
    distance: str = "euclidean",
    solver: str = "closed_form",
    regularisation_type: str = "none",
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fit a regularized model to the data.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    normalisation : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to evaluate ('mse', 'mae', 'r2', 'logloss') or custom callable.
    distance : str, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski').
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    regularisation_type : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet').
    alpha : float, optional
        Regularization strength.
    l1_ratio : float, optional
        ElasticNet mixing parameter (0 <= l1_ratio <= 1).
    tol : float, optional
        Tolerance for stopping criteria.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function.
    custom_distance : callable, optional
        Custom distance function.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _normalize_data(X, normalisation=normalisation)

    # Choose metric
    if isinstance(metric, str):
        metric_func = _get_metric(metric)
    else:
        metric_func = metric

    # Choose distance
    if custom_distance is not None:
        distance_func = custom_distance
    else:
        distance_func = _get_distance(distance)

    # Choose solver
    if solver == "closed_form":
        params = _solve_closed_form(X_normalized, y, regularisation_type, alpha, l1_ratio)
    elif solver == "gradient_descent":
        params = _solve_gradient_descent(X_normalized, y, regularisation_type, alpha,
                                        l1_ratio, tol, max_iter)
    elif solver == "newton":
        params = _solve_newton(X_normalized, y, regularisation_type, alpha,
                              l1_ratio, tol, max_iter)
    elif solver == "coordinate_descent":
        params = _solve_coordinate_descent(X_normalized, y, regularisation_type, alpha,
                                          l1_ratio, tol, max_iter)
    else:
        raise ValueError("Invalid solver specified.")

    # Calculate metrics
    metrics = _calculate_metrics(X_normalized, y, params, metric_func)

    # Prepare output
    result = {
        "result": params,
        "metrics": metrics,
        "params_used": {
            "normalisation": normalisation,
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularisation_type": regularisation_type,
            "alpha": alpha,
            "l1_ratio": l1_ratio
        },
        "warnings": []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")

def _normalize_data(X: np.ndarray, normalisation: str) -> np.ndarray:
    """Normalize the data according to specified method."""
    if normalisation == "none":
        return X
    elif normalisation == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif normalisation == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif normalisation == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError("Invalid normalisation method specified.")

def _get_metric(metric: str) -> Callable:
    """Return the metric function based on specified name."""
    metrics = {
        "mse": _mean_squared_error,
        "mae": _mean_absolute_error,
        "r2": _r_squared,
        "logloss": _log_loss
    }
    if metric not in metrics:
        raise ValueError("Invalid metric specified.")
    return metrics[metric]

def _get_distance(distance: str) -> Callable:
    """Return the distance function based on specified name."""
    distances = {
        "euclidean": _euclidean_distance,
        "manhattan": _manhattan_distance,
        "cosine": _cosine_distance,
        "minkowski": lambda x, y: np.sum(np.abs(x - y)**3, axis=1)**(1/3)
    }
    if distance not in distances:
        raise ValueError("Invalid distance specified.")
    return distances[distance]

def _solve_closed_form(X: np.ndarray, y: np.ndarray,
                      regularisation_type: str, alpha: float, l1_ratio: float) -> np.ndarray:
    """Solve using closed-form solution."""
    n_samples, n_features = X.shape
    if regularisation_type == "none":
        params = np.linalg.inv(X.T @ X) @ X.T @ y
    elif regularisation_type == "l2":
        identity = alpha * np.eye(n_features)
        params = np.linalg.inv(X.T @ X + identity) @ X.T @ y
    elif regularisation_type == "l1":
        # Simplified L1 solution (in practice, use coordinate descent)
        params = np.linalg.pinv(X) @ y
    elif regularisation_type == "elasticnet":
        # Simplified ElasticNet solution (in practice, use coordinate descent)
        params = np.linalg.pinv(X) @ y
    else:
        raise ValueError("Invalid regularisation type specified.")
    return params

def _solve_gradient_descent(X: np.ndarray, y: np.ndarray,
                           regularisation_type: str, alpha: float, l1_ratio: float,
                           tol: float, max_iter: int) -> np.ndarray:
    """Solve using gradient descent."""
    n_samples, n_features = X.shape
    params = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradient = X.T @ (X @ params - y) / n_samples
        if regularisation_type == "l2":
            gradient += 2 * alpha * params
        elif regularisation_type == "l1":
            gradient += alpha * np.sign(params)
        elif regularisation_type == "elasticnet":
            gradient += alpha * (l1_ratio * np.sign(params) + (1 - l1_ratio) * 2 * params)

        new_params = params - learning_rate * gradient
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params

    return params

def _solve_newton(X: np.ndarray, y: np.ndarray,
                  regularisation_type: str, alpha: float, l1_ratio: float,
                  tol: float, max_iter: int) -> np.ndarray:
    """Solve using Newton's method."""
    n_samples, n_features = X.shape
    params = np.zeros(n_features)

    for _ in range(max_iter):
        gradient = X.T @ (X @ params - y) / n_samples
        hessian = X.T @ X / n_samples

        if regularisation_type == "l2":
            hessian += 2 * alpha * np.eye(n_features)
        elif regularisation_type == "l1":
            hessian += alpha * np.diag(np.sign(params))
        elif regularisation_type == "elasticnet":
            hessian += alpha * np.diag(l1_ratio * np.sign(params) + (1 - l1_ratio) * 2)

        new_params = params - np.linalg.inv(hessian) @ gradient
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params

    return params

def _solve_coordinate_descent(X: np.ndarray, y: np.ndarray,
                             regularisation_type: str, alpha: float, l1_ratio: float,
                             tol: float, max_iter: int) -> np.ndarray:
    """Solve using coordinate descent."""
    n_samples, n_features = X.shape
    params = np.zeros(n_features)

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residual = y - X @ params + params[j] * X_j
            if regularisation_type == "l1":
                params[j] = np.sign(X_j.T @ residual) * np.maximum(
                    np.abs(X_j.T @ residual) - alpha, 0) / (X_j.T @ X_j)
            elif regularisation_type == "l2":
                params[j] = (X_j.T @ residual) / (X_j.T @ X_j + 2 * alpha)
            elif regularisation_type == "elasticnet":
                params[j] = np.sign(X_j.T @ residual) * np.maximum(
                    np.abs(X_j.T @ residual) - alpha * l1_ratio, 0) / (
                        X_j.T @ X_j + 2 * alpha * (1 - l1_ratio))

        if np.linalg.norm(params) < tol:
            break

    return params

def _calculate_metrics(X: np.ndarray, y: np.ndarray,
                      params: np.ndarray, metric_func: Callable) -> Dict:
    """Calculate metrics for the model."""
    predictions = X @ params
    return {"metric": metric_func(y, predictions)}

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate log loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Euclidean distance."""
    return np.linalg.norm(x - y)

def _manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Manhattan distance."""
    return np.sum(np.abs(x - y))

def _cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate cosine distance."""
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

################################################################################
# cross_validation
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def cross_validation_fit(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    cv_splits: int = 5,
    scoring: Union[str, Callable] = 'mse',
    normalize: Optional[str] = None,
    solver: str = 'closed_form',
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform cross-validation to estimate model performance and diagnose bias-variance tradeoff.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    model : Callable
        Model class or function with fit/predict interface
    cv_splits : int, optional
        Number of cross-validation folds (default: 5)
    scoring : str or Callable, optional
        Metric to evaluate ('mse', 'mae', 'r2', or custom callable)
    normalize : str, optional
        Normalization method ('standard', 'minmax', 'robust') or None
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent')
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    Dict containing:
        - 'result': cross-validated performance metrics
        - 'metrics': detailed metrics per fold
        - 'params_used': parameters used in computation
        - 'warnings': any warnings encountered

    Example:
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = cross_validation_fit(X, y, LinearRegression())
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if requested
    X_norm = _apply_normalization(X, normalize)

    # Initialize results dictionary
    results: Dict[str, Any] = {
        'result': {},
        'metrics': [],
        'params_used': {
            'cv_splits': cv_splits,
            'scoring': scoring,
            'normalize': normalize,
            'solver': solver
        },
        'warnings': []
    }

    # Get scoring function
    score_func = _get_scoring_function(scoring)

    # Perform cross-validation
    fold_sizes = np.full(cv_splits, len(X) // cv_splits)
    fold_sizes[:len(X) % cv_splits] += 1

    current = 0
    for fold in range(cv_splits):
        # Split data into train and validation sets
        val_indices = np.arange(current, current + fold_sizes[fold])
        train_indices = np.setdiff1d(np.arange(len(X)), val_indices)

        X_train, y_train = X_norm[train_indices], y[train_indices]
        X_val, y_val = X_norm[val_indices], y[val_indices]

        # Fit model
        model_instance = _initialize_model(model, solver)
        model_instance.fit(X_train, y_train)

        # Make predictions
        y_pred = model_instance.predict(X_val)

        # Calculate metrics
        fold_metrics = {
            'fold': fold,
            'score': score_func(y_val, y_pred),
            'predictions': y_pred
        }
        results['metrics'].append(fold_metrics)

        current += fold_sizes[fold]

    # Calculate overall results
    results['result']['mean_score'] = np.mean([m['score'] for m in results['metrics']])
    results['result']['std_score'] = np.std([m['score'] for m in results['metrics']])

    return results

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
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

def _apply_normalization(X: np.ndarray, method: Optional[str]) -> np.ndarray:
    """Apply requested normalization to features."""
    if method is None:
        return X

    if method == 'standard':
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

def _get_scoring_function(scoring: Union[str, Callable]) -> Callable:
    """Get scoring function based on input."""
    if callable(scoring):
        return scoring

    if isinstance(scoring, str):
        if scoring == 'mse':
            return _mean_squared_error
        elif scoring == 'mae':
            return _mean_absolute_error
        elif scoring == 'r2':
            return _r_squared
        else:
            raise ValueError(f"Unknown scoring metric: {scoring}")

    raise TypeError("Scoring must be either a string or callable")

def _initialize_model(model: Callable, solver: str) -> Any:
    """Initialize model with requested solver."""
    if hasattr(model, 'set_params'):
        # For scikit-learn style models
        model.set_params(solver=solver)
    return model

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))

################################################################################
# ensemble_methods
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def ensemble_methods_fit(
    X: np.ndarray,
    y: np.ndarray,
    base_models: list,
    aggregation_method: str = 'mean',
    metric: Union[str, Callable] = 'mse',
    normalization: str = 'none',
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Fit an ensemble of models and compute the aggregated prediction.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    base_models : list
        List of fitted base models with a predict method.
    aggregation_method : str, optional
        Method to aggregate predictions ('mean', 'median', 'max', 'min').
    metric : str or callable, optional
        Metric to evaluate the ensemble ('mse', 'mae', 'r2', custom callable).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    custom_metric : callable, optional
        Custom metric function if metric is not predefined.
    **kwargs :
        Additional keyword arguments for the aggregation method.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_normalized = _apply_normalization(X, normalization)

    # Get predictions from base models
    predictions = np.array([model.predict(X_normalized) for model in base_models])

    # Aggregate predictions
    aggregated_prediction = _aggregate_predictions(predictions, aggregation_method)

    # Compute metrics
    metrics = _compute_metrics(y, aggregated_prediction, metric, custom_metric)

    # Prepare output
    result = {
        'result': aggregated_prediction,
        'metrics': metrics,
        'params_used': {
            'aggregation_method': aggregation_method,
            'metric': metric,
            'normalization': normalization
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays.")
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
    """Apply normalization to the input data."""
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

def _aggregate_predictions(predictions: np.ndarray, method: str) -> np.ndarray:
    """Aggregate predictions from base models."""
    if method == 'mean':
        return np.mean(predictions, axis=0)
    elif method == 'median':
        return np.median(predictions, axis=0)
    elif method == 'max':
        return np.max(predictions, axis=0)
    elif method == 'min':
        return np.min(predictions, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, metric: Union[str, Callable], custom_metric: Optional[Callable] = None) -> Dict[str, float]:
    """Compute metrics for the ensemble predictions."""
    metrics = {}

    if isinstance(metric, str):
        if metric == 'mse':
            metrics['mse'] = np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['r2'] = 1 - (ss_res / (ss_tot + 1e-8))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        metrics['custom'] = metric(y_true, y_pred)
    else:
        raise TypeError("Metric must be a string or callable.")

    if custom_metric is not None:
        metrics['custom_metric'] = custom_metric(y_true, y_pred)

    return metrics
