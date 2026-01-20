"""
Quantix – Module robustesse_modeles
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# sensibilite_aux_donnees
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def sensibilite_aux_donnees_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalisation: str = 'standard',
    distance_metric: Union[str, Callable] = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    metric: Union[str, Callable] = 'mse',
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_normalize: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None,
    custom_metric: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Estimate the sensitivity of a model to data variations.

    Parameters:
    - X: Input features (n_samples, n_features)
    - y: Target values (n_samples,)
    - normalisation: Normalization method ('none', 'standard', 'minmax', 'robust')
    - distance_metric: Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable
    - solver: Optimization method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    - regularization: Regularization type ('none', 'l1', 'l2', 'elasticnet')
    - metric: Evaluation metric ('mse', 'mae', 'r2', 'logloss') or custom callable
    - tol: Tolerance for convergence
    - max_iter: Maximum number of iterations
    - custom_normalize: Custom normalization function
    - custom_distance: Custom distance function
    - custom_metric: Custom metric function

    Returns:
    - Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _apply_normalization(
        X, normalisation, custom_normalize=custom_normalize
    )

    # Prepare distance metric
    distance_func = _get_distance_function(distance_metric, custom_distance)

    # Prepare solver
    params, model_func = _get_solver_function(
        solver, regularization, tol, max_iter
    )

    # Prepare metric
    metric_func = _get_metric_function(metric, custom_metric)

    # Fit model
    params_estimated = _fit_model(
        X_normalized, y, model_func, distance_func
    )

    # Calculate metrics
    metrics = _calculate_metrics(
        y, model_func(params_estimated), metric_func
    )

    # Prepare output
    result = {
        'result': params_estimated,
        'metrics': metrics,
        'params_used': {
            'normalisation': normalisation,
            'distance_metric': distance_metric,
            'solver': solver,
            'regularization': regularization,
            'metric': metric
        },
        'warnings': _check_warnings(params_estimated)
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalization(
    X: np.ndarray,
    method: str,
    custom_normalize: Optional[Callable] = None
) -> np.ndarray:
    """Apply specified normalization to data."""
    if method == 'none':
        return X
    elif method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    elif custom_normalize is not None:
        return custom_normalize(X)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _get_distance_function(
    metric: Union[str, Callable],
    custom_distance: Optional[Callable] = None
) -> Callable:
    """Get distance function based on input."""
    if callable(metric):
        return metric
    elif custom_distance is not None:
        return custom_distance
    elif metric == 'euclidean':
        return lambda x, y: np.linalg.norm(x - y)
    elif metric == 'manhattan':
        return lambda x, y: np.sum(np.abs(x - y))
    elif metric == 'cosine':
        return lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    elif metric == 'minkowski':
        return lambda x, y: np.sum(np.abs(x - y)**3)**(1/3)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _get_solver_function(
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> tuple:
    """Get solver function and parameters based on input."""
    if solver == 'closed_form':
        return _closed_form_solver, {}
    elif solver == 'gradient_descent':
        params = {
            'learning_rate': 0.01,
            'tol': tol,
            'max_iter': max_iter
        }
        return _gradient_descent_solver, params
    elif solver == 'newton':
        params = {
            'tol': tol,
            'max_iter': max_iter
        }
        return _newton_solver, params
    elif solver == 'coordinate_descent':
        params = {
            'tol': tol,
            'max_iter': max_iter
        }
        return _coordinate_descent_solver, params
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _closed_form_solver(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed form solution for linear regression."""
    return np.linalg.inv(X.T @ X) @ X.T @ y

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Gradient descent solver."""
    n_samples, n_features = X.shape
    params = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = 2/n_samples * X.T @ (X @ params - y)
        new_params = params - learning_rate * gradient
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params
    return params

def _newton_solver(
    X: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Newton's method solver."""
    n_samples, n_features = X.shape
    params = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = 2/n_samples * X.T @ (X @ params - y)
        hessian = 2/n_samples * X.T @ X
        new_params = params - np.linalg.inv(hessian) @ gradient
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params
    return params

def _coordinate_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Coordinate descent solver."""
    n_samples, n_features = X.shape
    params = np.zeros(n_features)
    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            params_j_old = params[j]
            params[j] = 0
            r = y - X @ params
            numerator = X_j.T @ r
            denominator = X_j.T @ X_j
            params[j] = numerator / denominator
            if np.linalg.norm(params - params_j_old) < tol:
                break
    return params

def _get_metric_function(
    metric: Union[str, Callable],
    custom_metric: Optional[Callable] = None
) -> Callable:
    """Get metric function based on input."""
    if callable(metric):
        return metric
    elif custom_metric is not None:
        return custom_metric
    elif metric == 'mse':
        return lambda y_true, y_pred: np.mean((y_true - y_pred)**2)
    elif metric == 'mae':
        return lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        return lambda y_true, y_pred: 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    elif metric == 'logloss':
        return lambda y_true, y_pred: -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _fit_model(
    X: np.ndarray,
    y: np.ndarray,
    model_func: Callable,
    distance_func: Callable
) -> np.ndarray:
    """Fit model using specified solver and distance metric."""
    return model_func(X, y)

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable
) -> Dict[str, float]:
    """Calculate metrics for model evaluation."""
    return {'metric': metric_func(y_true, y_pred)}

def _check_warnings(params: np.ndarray) -> list:
    """Check for potential warnings in the results."""
    warnings = []
    if np.any(np.isnan(params)):
        warnings.append("Parameters contain NaN values")
    if np.any(np.isinf(params)):
        warnings.append("Parameters contain infinite values")
    return warnings

################################################################################
# stabilité_modele
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List, Any

def stabilité_modele_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalisation: str = 'standard',
    métrique: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solveur: str = 'gradient_descent',
    régularisation: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Évalue la stabilité d'un modèle en fonction des paramètres donnés.

    Parameters:
    -----------
    X : np.ndarray
        Matrice des caractéristiques (n_samples, n_features).
    y : np.ndarray
        Vecteur cible (n_samples,).
    normalisation : str, optional
        Type de normalisation à appliquer ('none', 'standard', 'minmax', 'robust').
    métrique : str or callable, optional
        Métrique d'évaluation ('mse', 'mae', 'r2', 'logloss') ou fonction personnalisée.
    distance : str or callable, optional
        Distance à utiliser ('euclidean', 'manhattan', 'cosine', 'minkowski') ou fonction personnalisée.
    solveur : str, optional
        Solveur à utiliser ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    régularisation : str, optional
        Type de régularisation ('none', 'l1', 'l2', 'elasticnet').
    tol : float, optional
        Tolérance pour la convergence.
    max_iter : int, optional
        Nombre maximal d'itérations.
    custom_metric : callable, optional
        Fonction personnalisée pour la métrique.
    custom_distance : callable, optional
        Fonction personnalisée pour la distance.

    Returns:
    --------
    Dict[str, Any]
        Dictionnaire contenant les résultats, métriques, paramètres utilisés et avertissements.
    """
    # Validation des entrées
    _validate_inputs(X, y)

    # Normalisation
    X_normalized = _apply_normalization(X, normalisation)

    # Choix de la métrique
    metric_func = _get_metric_function(métrique, custom_metric)

    # Choix de la distance
    distance_func = _get_distance_function(distance, custom_distance)

    # Choix du solveur
    solver_func = _get_solver_function(solveur)

    # Estimation des paramètres
    params, warnings = solver_func(
        X_normalized,
        y,
        distance=distance_func,
        régularisation=régularisation,
        tol=tol,
        max_iter=max_iter
    )

    # Calcul des métriques
    metrics = _compute_metrics(X_normalized, y, params, metric_func)

    # Retourne le dictionnaire structuré
    return {
        "result": params,
        "metrics": metrics,
        "params_used": {
            "normalisation": normalisation,
            "métrique": métrique if isinstance(métrique, str) else "custom",
            "distance": distance if isinstance(distance, str) else "custom",
            "solveur": solveur,
            "régularisation": régularisation,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": warnings
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Valide les entrées pour éviter les erreurs silencieuses."""
    if X.ndim != 2:
        raise ValueError("X doit être une matrice 2D.")
    if y.ndim != 1:
        raise ValueError("y doit être un vecteur 1D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X et y doivent avoir le même nombre d'échantillons.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("X et y ne doivent pas contenir de NaN.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("X et y ne doivent pas contenir d'inf.")

def _apply_normalization(X: np.ndarray, normalisation: str) -> np.ndarray:
    """Applique la normalisation choisie."""
    if normalisation == 'none':
        return X
    elif normalisation == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalisation == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalisation == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError(f"Normalisation '{normalisation}' non reconnue.")

def _get_metric_function(metric: Union[str, Callable], custom_metric: Optional[Callable]) -> Callable:
    """Retourne la fonction de métrique choisie."""
    if isinstance(metric, str):
        if metric == 'mse':
            return _mse
        elif metric == 'mae':
            return _mae
        elif metric == 'r2':
            return _r2
        elif metric == 'logloss':
            return _logloss
        else:
            raise ValueError(f"Métrique '{metric}' non reconnue.")
    elif callable(metric):
        return metric
    elif custom_metric is not None:
        return custom_metric
    else:
        raise ValueError("Aucune métrique valide fournie.")

def _get_distance_function(distance: Union[str, Callable], custom_distance: Optional[Callable]) -> Callable:
    """Retourne la fonction de distance choisie."""
    if isinstance(distance, str):
        if distance == 'euclidean':
            return _euclidean_distance
        elif distance == 'manhattan':
            return _manhattan_distance
        elif distance == 'cosine':
            return _cosine_distance
        elif distance == 'minkowski':
            return lambda x, y: np.sum(np.abs(x - y) ** 3, axis=1) ** (1/3)
        else:
            raise ValueError(f"Distance '{distance}' non reconnue.")
    elif callable(distance):
        return distance
    elif custom_distance is not None:
        return custom_distance
    else:
        raise ValueError("Aucune distance valide fournie.")

def _get_solver_function(solveur: str) -> Callable:
    """Retourne la fonction de solveur choisie."""
    if solveur == 'closed_form':
        return _closed_form_solver
    elif solveur == 'gradient_descent':
        return _gradient_descent_solver
    elif solveur == 'newton':
        return _newton_solver
    elif solveur == 'coordinate_descent':
        return _coordinate_descent_solver
    else:
        raise ValueError(f"Solveur '{solveur}' non reconnu.")

def _compute_metrics(X: np.ndarray, y: np.ndarray, params: Dict[str, Any], metric_func: Callable) -> Dict[str, float]:
    """Calcule les métriques pour les paramètres estimés."""
    predictions = _predict(X, params)
    return {metric_func.__name__: metric_func(y, predictions)}

def _predict(X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Prédit les valeurs à partir des paramètres estimés."""
    return X @ params['coefficients'] + params['intercept']

def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule l'erreur quadratique moyenne."""
    return np.mean((y_true - y_pred) ** 2)

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule l'erreur absolue moyenne."""
    return np.mean(np.abs(y_true - y_pred))

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule le coefficient de détermination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule la log loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calcule la distance euclidienne."""
    return np.linalg.norm(x - y)

def _manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calcule la distance de Manhattan."""
    return np.sum(np.abs(x - y))

def _cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calcule la distance cosinus."""
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def _closed_form_solver(X: np.ndarray, y: np.ndarray, **kwargs) -> tuple:
    """Solveur en forme fermée."""
    X_tx = np.linalg.inv(X.T @ X)
    coefficients = X_tx @ X.T @ y
    intercept = np.mean(y - X @ coefficients)
    return {'coefficients': coefficients, 'intercept': intercept}, []

def _gradient_descent_solver(X: np.ndarray, y: np.ndarray, **kwargs) -> tuple:
    """Solveur par descente de gradient."""
    warnings = []
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)
    intercept = 0.0
    learning_rate = kwargs.get('learning_rate', 0.01)
    tol = kwargs.get('tol', 1e-4)
    max_iter = kwargs.get('max_iter', 1000)

    for _ in range(max_iter):
        gradients = (2 / n_samples) * X.T @ (X @ coefficients + intercept - y)
        coefficients -= learning_rate * gradients
        intercept -= learning_rate * np.mean(X @ coefficients + intercept - y)

        if np.linalg.norm(gradients) < tol:
            break

    return {'coefficients': coefficients, 'intercept': intercept}, warnings

def _newton_solver(X: np.ndarray, y: np.ndarray, **kwargs) -> tuple:
    """Solveur par méthode de Newton."""
    warnings = []
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)
    intercept = 0.0
    tol = kwargs.get('tol', 1e-4)
    max_iter = kwargs.get('max_iter', 1000)

    for _ in range(max_iter):
        residuals = X @ coefficients + intercept - y
        hessian = (2 / n_samples) * X.T @ X

        update = np.linalg.inv(hessian) @ (2 / n_samples) * X.T @ residuals
        coefficients -= update[:n_features]
        intercept -= np.mean(residuals)

        if np.linalg.norm(update) < tol:
            break

    return {'coefficients': coefficients, 'intercept': intercept}, warnings

def _coordinate_descent_solver(X: np.ndarray, y: np.ndarray, **kwargs) -> tuple:
    """Solveur par descente de coordonnées."""
    warnings = []
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)
    intercept = 0.0
    tol = kwargs.get('tol', 1e-4)
    max_iter = kwargs.get('max_iter', 1000)

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - intercept - X @ coefficients + coefficients[j] * X_j
            coefficients[j] = np.sum(X_j * residuals) / np.sum(X_j ** 2)

        intercept = np.mean(y - X @ coefficients)
        if np.linalg.norm(coefficients) < tol:
            break

    return {'coefficients': coefficients, 'intercept': intercept}, warnings

################################################################################
# generalisation_modele
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def generalisation_modele_fit(
    X: np.ndarray,
    y: np.ndarray,
    model_func: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, Dict]],
    metric_func: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    normalization: Optional[str] = None,
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_params: Optional[Dict] = None
) -> Dict:
    """
    Fit a model and evaluate its generalization performance.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    model_func : Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, Dict]]
        A callable that fits the model and returns (predictions, model_params).
    metric_func : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Metric to evaluate model performance. Can be 'mse', 'mae', 'r2', or a custom callable.
    normalization : Optional[str]
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    solver : str
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'.
    regularization : Optional[str]
        Regularization method: 'none', 'l1', 'l2', or 'elasticnet'.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_params : Optional[Dict]
        Additional parameters for the model.

    Returns:
    --------
    Dict
        A dictionary containing 'result', 'metrics', 'params_used', and 'warnings'.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_normalized = _apply_normalization(X, normalization)

    # Initialize custom parameters if provided
    params = custom_params or {}

    # Fit the model
    predictions, model_params = _fit_model(
        X_normalized, y, model_func, solver, regularization,
        tol, max_iter, params
    )

    # Compute metrics
    metrics = _compute_metrics(y, predictions, metric_func)

    return {
        'result': predictions,
        'metrics': metrics,
        'params_used': model_params,
        'warnings': []
    }

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
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("X and y must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("X and y must not contain infinite values.")

def _apply_normalization(X: np.ndarray, method: Optional[str]) -> np.ndarray:
    """Apply normalization to the input data."""
    if method is None or method == 'none':
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
    model_func: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, Dict]],
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    params: Dict
) -> Tuple[np.ndarray, Dict]:
    """Fit the model using the specified solver and regularization."""
    if solver == 'closed_form':
        predictions, model_params = _fit_closed_form(X, y, regularization)
    elif solver == 'gradient_descent':
        predictions, model_params = _fit_gradient_descent(X, y, tol, max_iter, params)
    elif solver == 'newton':
        predictions, model_params = _fit_newton(X, y, tol, max_iter, params)
    elif solver == 'coordinate_descent':
        predictions, model_params = _fit_coordinate_descent(X, y, tol, max_iter, params)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    return predictions, model_params

def _fit_closed_form(X: np.ndarray, y: np.ndarray, regularization: Optional[str]) -> Tuple[np.ndarray, Dict]:
    """Fit the model using closed-form solution."""
    # Placeholder for closed-form implementation
    return np.zeros(X.shape[0]), {'method': 'closed_form'}

def _fit_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int,
    params: Dict
) -> Tuple[np.ndarray, Dict]:
    """Fit the model using gradient descent."""
    # Placeholder for gradient descent implementation
    return np.zeros(X.shape[0]), {'method': 'gradient_descent'}

def _fit_newton(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int,
    params: Dict
) -> Tuple[np.ndarray, Dict]:
    """Fit the model using Newton's method."""
    # Placeholder for Newton's method implementation
    return np.zeros(X.shape[0]), {'method': 'newton'}

def _fit_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int,
    params: Dict
) -> Tuple[np.ndarray, Dict]:
    """Fit the model using coordinate descent."""
    # Placeholder for coordinate descent implementation
    return np.zeros(X.shape[0]), {'method': 'coordinate_descent'}

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict:
    """Compute the specified metrics."""
    metrics = {}

    if isinstance(metric_func, str):
        if metric_func == 'mse':
            metrics['mse'] = np.mean((y_true - y_pred) ** 2)
        elif metric_func == 'mae':
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metric_func == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot)
        else:
            raise ValueError(f"Unknown metric: {metric_func}")
    else:
        metrics['custom'] = metric_func(y_true, y_pred)

    return metrics

################################################################################
# overfitting
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> None:
    """Validate input data dimensions and types."""
    if not all(isinstance(arr, np.ndarray) for arr in [X_train, y_train, X_val, y_val]):
        raise TypeError("All inputs must be numpy arrays")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train must have the same number of samples")
    if X_val.shape[0] != y_val.shape[0]:
        raise ValueError("X_val and y_val must have the same number of samples")
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("X_train and X_val must have the same number of features")

def _normalize_data(
    data: np.ndarray,
    method: str = "standard",
    custom_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize data using specified method or custom function."""
    if custom_func is not None:
        return custom_func(data)

    if method == "standard":
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
    elif method == "none":
        return data
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "mse",
    custom_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> float:
    """Compute specified metric or custom function."""
    if custom_func is not None:
        return custom_func(y_true, y_pred)

    if metric == "mse":
        return np.mean((y_true - y_pred) ** 2)
    elif metric == "mae":
        return np.mean(np.abs(y_true - y_pred))
    elif metric == "r2":
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    elif metric == "logloss":
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _fit_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    solver: str = "closed_form",
    custom_solver: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    **kwargs
) -> np.ndarray:
    """Fit model using specified solver or custom function."""
    if custom_solver is not None:
        return custom_solver(X_train, y_train)

    if solver == "closed_form":
        XtX = np.dot(X_train.T, X_train)
        if not np.allclose(np.linalg.det(XtX), 0):
            return np.linalg.solve(XtX, np.dot(X_train.T, y_train))
        else:
            raise ValueError("Matrix is singular")
    elif solver == "gradient_descent":
        learning_rate = kwargs.get("learning_rate", 0.01)
        n_iter = kwargs.get("n_iter", 1000)
        tol = kwargs.get("tol", 1e-4)

        n_samples, n_features = X_train.shape
        weights = np.zeros(n_features)
        prev_loss = float('inf')

        for _ in range(n_iter):
            gradients = 2/n_samples * X_train.T.dot(X_train.dot(weights) - y_train)
            weights -= learning_rate * gradients
            current_loss = np.mean((X_train.dot(weights) - y_train) ** 2)

            if abs(prev_loss - current_loss) < tol:
                break
            prev_loss = current_loss

        return weights
    else:
        raise ValueError(f"Unknown solver: {solver}")

def overfitting_fit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    normalize_method: str = "standard",
    metric: str = "mse",
    solver: str = "closed_form",
    custom_normalize: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_solver: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute overfitting metrics between training and validation sets.

    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training targets
    X_val : np.ndarray
        Validation features
    y_val : np.ndarray
        Validation targets
    normalize_method : str, optional
        Normalization method for data (default: "standard")
    metric : str, optional
        Metric to compute (default: "mse")
    solver : str, optional
        Solver method for model fitting (default: "closed_form")
    custom_normalize : callable, optional
        Custom normalization function
    custom_metric : callable, optional
        Custom metric function
    custom_solver : callable, optional
        Custom solver function

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X_train = np.random.rand(100, 5)
    >>> y_train = np.random.rand(100)
    >>> X_val = np.random.rand(20, 5)
    >>> y_val = np.random.rand(20)
    >>> result = overfitting_fit(X_train, y_train, X_val, y_val)
    """
    # Validate inputs
    _validate_inputs(X_train, y_train, X_val, y_val)

    # Normalize data
    X_train_norm = _normalize_data(X_train, normalize_method, custom_normalize)
    X_val_norm = _normalize_data(X_val, normalize_method, custom_normalize)

    # Fit model
    params = _fit_model(X_train_norm, y_train, solver, custom_solver, **kwargs)

    # Make predictions
    y_train_pred = X_train_norm.dot(params)
    y_val_pred = X_val_norm.dot(params)

    # Compute metrics
    train_metric = _compute_metric(y_train, y_train_pred, metric, custom_metric)
    val_metric = _compute_metric(y_val, y_val_pred, metric, custom_metric)

    # Check for overfitting
    overfit = train_metric < val_metric

    return {
        "result": {
            "overfitting_detected": overfit,
            "train_metric_value": train_metric,
            "val_metric_value": val_metric
        },
        "metrics": {
            "train_metric": metric if custom_metric is None else "custom",
            "val_metric": metric if custom_metric is None else "custom"
        },
        "params_used": {
            "normalization_method": normalize_method if custom_normalize is None else "custom",
            "metric": metric if custom_metric is None else "custom",
            "solver": solver if custom_solver is None else "custom"
        },
        "warnings": []
    }

################################################################################
# underfitting
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def underfitting_fit(
    X: np.ndarray,
    y: np.ndarray,
    model_func: Callable,
    metric_func: Callable = None,
    normalizer: Optional[Callable] = None,
    solver: str = 'closed_form',
    max_iter: int = 1000,
    tol: float = 1e-4,
    l2_penalty: float = 0.0,
    verbose: bool = False
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Detect underfitting by comparing model performance with different complexities.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    model_func : Callable
        Function that returns a model with given complexity (e.g., polynomial degree)
    metric_func : Callable, optional
        Metric function to evaluate model performance (default: MSE)
    normalizer : Callable, optional
        Normalization function for features (default: None)
    solver : str
        Solver method ('closed_form', 'gradient_descent')
    max_iter : int
        Maximum iterations for iterative solvers
    tol : float
        Tolerance for convergence
    l2_penalty : float
        L2 regularization parameter
    verbose : bool
        Whether to print progress information

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': Model predictions
        - 'metrics': Performance metrics
        - 'params_used': Parameters used in fitting
        - 'warnings': Any warnings encountered

    Example
    -------
    >>> def polynomial_model(degree):
    ...     return lambda X: np.polyfit(X, y, degree)
    ...
    >>> result = underfitting_fit(X_train, y_train,
    ...                          model_func=polynomial_model(2),
    ...                          metric_func='mse')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Initialize warnings dictionary
    warnings = {}

    # Apply normalization if specified
    X_normalized, norm_warning = _apply_normalization(X, normalizer)
    if norm_warning:
        warnings['normalization'] = norm_warning

    # Get model with specified complexity
    model = model_func()

    # Fit the model using selected solver
    params, fit_warning = _fit_model(
        X_normalized, y,
        model=model,
        solver=solver,
        max_iter=max_iter,
        tol=tol,
        l2_penalty=l2_penalty
    )
    if fit_warning:
        warnings['fitting'] = fit_warning

    # Make predictions
    y_pred = model.predict(X_normalized)

    # Calculate metrics
    metrics = _calculate_metrics(y, y_pred, metric_func)

    return {
        'result': y_pred,
        'metrics': metrics,
        'params_used': {
            'solver': solver,
            'max_iter': max_iter,
            'tol': tol,
            'l2_penalty': l2_penalty
        },
        'warnings': warnings
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalization(
    X: np.ndarray,
    normalizer: Optional[Callable]
) -> tuple[np.ndarray, str]:
    """Apply normalization to features."""
    if normalizer is None:
        return X, ""

    try:
        X_normalized = normalizer(X)
        if not isinstance(X_normalized, np.ndarray):
            raise ValueError("Normalizer must return numpy array")
        if X_normalized.shape != X.shape:
            raise ValueError("Normalizer must preserve input shape")
        return X_normalized, ""
    except Exception as e:
        warning_msg = f"Normalization failed: {str(e)}. Using original data."
        return X, warning_msg

def _fit_model(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    solver: str,
    max_iter: int,
    tol: float,
    l2_penalty: float
) -> tuple[np.ndarray, str]:
    """Fit model using specified solver."""
    if solver == 'closed_form':
        params = _fit_closed_form(X, y, l2_penalty)
        return params, ""
    elif solver == 'gradient_descent':
        try:
            params = _fit_gradient_descent(
                X, y,
                model=model,
                max_iter=max_iter,
                tol=tol,
                l2_penalty=l2_penalty
            )
            return params, ""
        except Exception as e:
            warning_msg = f"Gradient descent failed: {str(e)}. Using closed form solution."
            return _fit_closed_form(X, y, l2_penalty), warning_msg
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _fit_closed_form(
    X: np.ndarray,
    y: np.ndarray,
    l2_penalty: float
) -> np.ndarray:
    """Fit model using closed form solution with optional L2 regularization."""
    XTX = X.T @ X
    if l2_penalty > 0:
        XTX += np.eye(X.shape[1]) * l2_penalty
    params = np.linalg.solve(XTX, X.T @ y)
    return params

def _fit_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    max_iter: int,
    tol: float,
    l2_penalty: float
) -> np.ndarray:
    """Fit model using gradient descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    prev_loss = float('inf')

    for _ in range(max_iter):
        grad = _compute_gradient(X, y, params, l2_penalty)
        params -= grad
        current_loss = _compute_loss(X, y, params)

        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return params

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    l2_penalty: float
) -> np.ndarray:
    """Compute gradient for L2 regularized linear regression."""
    residuals = X @ params - y
    grad = (X.T @ residuals) / len(y)
    if l2_penalty > 0:
        grad += params * l2_penalty
    return grad

def _compute_loss(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray
) -> float:
    """Compute MSE loss."""
    return np.mean((X @ params - y) ** 2)

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Optional[Callable]
) -> Dict[str, float]:
    """Calculate performance metrics."""
    if metric_func is None:
        return {'mse': np.mean((y_true - y_pred) ** 2)}

    try:
        if callable(metric_func):
            return {str(metric_func).__name__: metric_func(y_true, y_pred)}
        else:
            raise ValueError("metric_func must be callable")
    except Exception as e:
        warning_msg = f"Metric calculation failed: {str(e)}. Using MSE as fallback."
        return {'mse': np.mean((y_true - y_pred) ** 2), 'warning': warning_msg}

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
    solver: str = "gradient_descent",
    regularisation_type: str = "none",
    alpha: float = 1.0,
    l1_ratio: Optional[float] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fit a regularized model with various options for normalization, metrics, distances, and solvers.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalisation : str, optional
        Normalization method: "none", "standard", "minmax", or "robust"
    metric : str or callable, optional
        Metric to optimize: "mse", "mae", "r2", "logloss", or custom callable
    distance : str, optional
        Distance metric: "euclidean", "manhattan", "cosine", or "minkowski"
    solver : str, optional
        Solver method: "closed_form", "gradient_descent", "newton", or "coordinate_descent"
    regularisation_type : str, optional
        Regularization type: "none", "l1", "l2", or "elasticnet"
    alpha : float, optional
        Regularization strength (default=1.0)
    l1_ratio : float, optional
        ElasticNet mixing parameter (default=None)
    tol : float, optional
        Tolerance for stopping criteria (default=1e-4)
    max_iter : int, optional
        Maximum number of iterations (default=1000)
    custom_metric : callable, optional
        Custom metric function (default=None)
    custom_distance : callable, optional
        Custom distance function (default=None)

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = regularisation_fit(X, y, normalisation="standard", metric="mse")
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm = _apply_normalisation(X, normalisation=normalisation)

    # Prepare metric and distance functions
    metric_func = _get_metric_function(metric, custom_metric)
    distance_func = _get_distance_function(distance, custom_distance)

    # Prepare solver
    if solver == "closed_form":
        params = _solve_closed_form(X_norm, y, regularisation_type, alpha, l1_ratio)
    elif solver == "gradient_descent":
        params = _solve_gradient_descent(
            X_norm, y, metric_func, distance_func,
            regularisation_type, alpha, l1_ratio,
            tol, max_iter
        )
    elif solver == "newton":
        params = _solve_newton(
            X_norm, y, metric_func,
            regularisation_type, alpha, l1_ratio,
            tol, max_iter
        )
    elif solver == "coordinate_descent":
        params = _solve_coordinate_descent(
            X_norm, y, metric_func,
            regularisation_type, alpha, l1_ratio,
            tol, max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(X_norm, y, params, metric_func)

    # Prepare results
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
            "l1_ratio": l1_ratio,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    return result

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

def _apply_normalisation(X: np.ndarray, normalisation: str) -> np.ndarray:
    """Apply specified normalization to the feature matrix."""
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
        raise ValueError(f"Unknown normalisation method: {normalisation}")

def _get_metric_function(metric: Union[str, Callable], custom_metric: Optional[Callable]) -> Callable:
    """Get the appropriate metric function."""
    if custom_metric is not None:
        return custom_metric

    if isinstance(metric, str):
        if metric == "mse":
            return _mean_squared_error
        elif metric == "mae":
            return _mean_absolute_error
        elif metric == "r2":
            return _r_squared
        elif metric == "logloss":
            return _log_loss
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        return metric

def _get_distance_function(distance: str, custom_distance: Optional[Callable]) -> Callable:
    """Get the appropriate distance function."""
    if custom_distance is not None:
        return custom_distance

    if distance == "euclidean":
        return _euclidean_distance
    elif distance == "manhattan":
        return _manhattan_distance
    elif distance == "cosine":
        return _cosine_distance
    elif distance == "minkowski":
        return _minkowski_distance
    else:
        raise ValueError(f"Unknown distance: {distance}")

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

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate log loss."""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _euclidean_distance(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate Euclidean distance."""
    if Y is None:
        Y = X
    return np.sqrt(np.sum((X[:, np.newaxis] - Y) ** 2, axis=2))

def _manhattan_distance(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate Manhattan distance."""
    if Y is None:
        Y = X
    return np.sum(np.abs(X[:, np.newaxis] - Y), axis=2)

def _cosine_distance(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate cosine distance."""
    if Y is None:
        Y = X
    dot_product = np.dot(X, Y.T)
    norm_X = np.linalg.norm(X, axis=1)[:, np.newaxis]
    norm_Y = np.linalg.norm(Y, axis=1)[np.newaxis, :]
    return 1 - (dot_product / (norm_X * norm_Y + 1e-8))

def _minkowski_distance(X: np.ndarray, Y: Optional[np.ndarray] = None, p: float = 2) -> np.ndarray:
    """Calculate Minkowski distance."""
    if Y is None:
        Y = X
    return np.sum(np.abs(X[:, np.newaxis] - Y) ** p, axis=2) ** (1/p)

def _solve_closed_form(
    X: np.ndarray,
    y: np.ndarray,
    regularisation_type: str,
    alpha: float,
    l1_ratio: Optional[float]
) -> np.ndarray:
    """Solve using closed-form solution."""
    if regularisation_type == "none":
        params = np.linalg.inv(X.T @ X + 1e-8) @ X.T @ y
    elif regularisation_type == "l2":
        params = np.linalg.inv(X.T @ X + alpha * np.eye(X.shape[1])) @ X.T @ y
    elif regularisation_type == "l1":
        # For L1, we need to use coordinate descent
        raise ValueError("Closed form solution not available for L1 regularization")
    elif regularisation_type == "elasticnet":
        if l1_ratio is None:
            raise ValueError("l1_ratio must be specified for elasticnet")
        # ElasticNet requires coordinate descent
        raise ValueError("Closed form solution not available for elasticnet regularization")
    else:
        raise ValueError(f"Unknown regularisation type: {regularisation_type}")
    return params

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable,
    distance_func: Callable,
    regularisation_type: str,
    alpha: float,
    l1_ratio: Optional[float],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve using gradient descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    learning_rate = 0.01
    prev_loss = float('inf')

    for _ in range(max_iter):
        y_pred = X @ params
        loss = metric_func(y, y_pred)

        if np.abs(prev_loss - loss) < tol:
            break

        prev_loss = loss
        gradient = _compute_gradient(X, y, params, metric_func, regularisation_type, alpha, l1_ratio)
        params -= learning_rate * gradient

    return params

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric_func: Callable,
    regularisation_type: str,
    alpha: float,
    l1_ratio: Optional[float]
) -> np.ndarray:
    """Compute gradient for regularized loss."""
    y_pred = X @ params

    if metric_func == _mean_squared_error:
        gradient = 2 * X.T @ (y_pred - y)
    elif metric_func == _mean_absolute_error:
        gradient = X.T @ np.sign(y_pred - y)
    else:
        raise ValueError("Gradient computation not implemented for this metric")

    if regularisation_type == "l2":
        gradient += 2 * alpha * params
    elif regularisation_type == "l1":
        gradient += alpha * np.sign(params)
    elif regularisation_type == "elasticnet" and l1_ratio is not None:
        gradient += alpha * (l1_ratio * np.sign(params) + (1 - l1_ratio) * 2 * params)

    return gradient

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable,
    regularisation_type: str,
    alpha: float,
    l1_ratio: Optional[float],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve using Newton's method."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    prev_loss = float('inf')

    for _ in range(max_iter):
        y_pred = X @ params
        loss = metric_func(y, y_pred)

        if np.abs(prev_loss - loss) < tol:
            break

        prev_loss = loss
        gradient = _compute_gradient(X, y, params, metric_func, regularisation_type, alpha, l1_ratio)
        hessian = _compute_hessian(X, metric_func, regularisation_type, alpha)
        params -= np.linalg.inv(hessian + 1e-8) @ gradient

    return params

def _compute_hessian(
    X: np.ndarray,
    metric_func: Callable,
    regularisation_type: str,
    alpha: float
) -> np.ndarray:
    """Compute Hessian matrix."""
    if metric_func == _mean_squared_error:
        hessian = 2 * X.T @ X
    else:
        raise ValueError("Hessian computation not implemented for this metric")

    if regularisation_type == "l2":
        hessian += 2 * alpha * np.eye(X.shape[1])

    return hessian

def _solve_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable,
    regularisation_type: str,
    alpha: float,
    l1_ratio: Optional[float],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve using coordinate descent."""
    n_samples, n_features = X.shape
    params = np.zeros(n_features)
    prev_loss = float('inf')

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residual = y - (X @ params) + params[j] * X_j

            if regularisation_type == "none":
                params[j] = np.sum(X_j * residual) / (np.sum(X_j ** 2) + 1e-8)
            elif regularisation_type == "l1":
                corr = np.sum(X_j * residual)
                if corr < -alpha / 2:
                    params[j] = (corr + alpha / 2) / np.sum(X_j ** 2)
                elif corr > alpha / 2:
                    params[j] = (corr - alpha / 2) / np.sum(X_j ** 2)
                else:
                    params[j] = 0
            elif regularisation_type == "l2":
                params[j] = np.sum(X_j * residual) / (np.sum(X_j ** 2) + alpha)
            elif regularisation_type == "elasticnet" and l1_ratio is not None:
                corr = np.sum(X_j * residual)
                denom = np.sum(X_j ** 2) + alpha * (1 - l1_ratio)
                if corr < -alpha * l1_ratio / 2:
                    params[j] = (corr + alpha * l1_ratio / 2) / denom
                elif corr > alpha * l1_ratio / 2:
                    params[j] = (corr - alpha * l1_ratio / 2) / denom
                else:
                    params[j] = 0

        y_pred = X @ params
        loss = metric_func(y, y_pred)

        if np.abs(prev_loss - loss) < tol:
            break

        prev_loss = loss

    return params

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric_func: Callable
) -> Dict[str, float]:
    """Calculate various metrics."""
    y_pred = X @ params
    metrics = {
        "metric": metric_func(y, y_pred)
    }

    # Add additional common metrics
    if metric_func != _r_squared:
        metrics["r2"] = _r_squared(y, y_pred)
    if metric_func != _mean_absolute_error:
        metrics["mae"] = _mean_absolute_error(y, y_pred)
    if metric_func != _log_loss:
        metrics["logloss"] = _log_loss(y, y_pred)

    return metrics

################################################################################
# cross_validation
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def cross_validation_fit(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    normalizer: Optional[Callable] = None,
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None,
    custom_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Perform cross-validation with configurable parameters.

    Parameters:
    -----------
    model : Callable
        The model to be validated. Must accept X, y and return a fitted model.
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    n_splits : int, optional
        Number of cross-validation splits. Default is 5.
    normalizer : Callable, optional
        Function to normalize the data. Default is None.
    metric : str or Callable, optional
        Metric to evaluate the model. Default is 'mse'.
    solver : str, optional
        Solver to use for model fitting. Default is 'closed_form'.
    regularization : str, optional
        Type of regularization to apply. Default is None.
    tol : float, optional
        Tolerance for convergence. Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    random_state : int, optional
        Random seed for reproducibility. Default is None.
    custom_params : Dict[str, Any], optional
        Additional parameters for the model. Default is None.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Initialize results dictionary
    results = {
        'result': None,
        'metrics': [],
        'params_used': {
            'n_splits': n_splits,
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter,
            'random_state': random_state
        },
        'warnings': []
    }

    # Normalize data if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Perform cross-validation
    fold_sizes = np.full(n_splits, len(X) // n_splits, dtype=int)
    fold_sizes[:len(X) % n_splits] += 1

    metrics = []
    for i in range(n_splits):
        # Split data into train and test sets
        test_indices = np.arange(i * fold_sizes[i], (i + 1) * fold_sizes[i])
        train_indices = np.setdiff1d(np.arange(len(X)), test_indices)

        X_train, y_train = X_normalized[train_indices], y[train_indices]
        X_test, y_test = X_normalized[test_indices], y[test_indices]

        # Fit model
        fitted_model = _fit_model(model, X_train, y_train, solver, regularization,
                                 tol, max_iter, random_state, custom_params)

        # Predict and compute metric
        y_pred = fitted_model.predict(X_test)
        current_metric = _compute_metric(y_test, y_pred, metric)

        metrics.append(current_metric)

    # Store results
    results['result'] = np.mean(metrics)
    results['metrics'] = metrics

    return results

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to the input data."""
    if normalizer is None:
        return X
    return normalizer(X)

def _fit_model(
    model: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    random_state: Optional[int],
    custom_params: Optional[Dict[str, Any]]
) -> Any:
    """Fit the model with specified parameters."""
    # Here you would implement the actual fitting logic based on solver, etc.
    # This is a placeholder for the actual implementation
    params = {
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'max_iter': max_iter
    }
    if custom_params:
        params.update(custom_params)

    if random_state is not None:
        np.random.seed(random_state)

    return model(X_train, y_train, **params)

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> float:
    """Compute the specified metric."""
    if callable(metric):
        return metric(y_true, y_pred)

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
        raise ValueError(f"Unknown metric: {metric}")

################################################################################
# bootstrap
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    n_bootstrap_samples: int = 1000,
    random_state: Optional[int] = None
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if n_bootstrap_samples <= 0:
        raise ValueError("n_bootstrap_samples must be positive")
    if random_state is not None and (not isinstance(random_state, int) or random_state < 0):
        raise ValueError("random_state must be a non-negative integer")

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse'
) -> float:
    """Compute the specified metric between true and predicted values."""
    if callable(metric):
        return metric(y_true, y_pred)

    metrics = {
        'mse': lambda y1, y2: np.mean((y1 - y2) ** 2),
        'mae': lambda y1, y2: np.mean(np.abs(y1 - y2)),
        'r2': lambda y1, y2: 1 - np.sum((y1 - y2) ** 2) / np.sum((y1 - np.mean(y1)) ** 2),
        'logloss': lambda y1, y2: -np.mean(y1 * np.log(y2) + (1 - y1) * np.log(1 - y2))
    }

    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric](y_true, y_pred)

def _bootstrap_sample(
    X: np.ndarray,
    y: np.ndarray,
    random_state: Optional[int] = None
) -> tuple:
    """Generate a bootstrap sample from the data."""
    n_samples = len(X)
    rng = np.random.RandomState(random_state)
    indices = rng.choice(n_samples, size=n_samples, replace=True)
    return X[indices], y[indices]

def _fit_model(
    X: np.ndarray,
    y: np.ndarray,
    model_func: Callable[[np.ndarray, np.ndarray], Any],
    **model_kwargs
) -> Any:
    """Fit a model to the data using the provided model function."""
    return model_func(X, y, **model_kwargs)

def bootstrap_fit(
    X: np.ndarray,
    y: np.ndarray,
    model_func: Callable[[np.ndarray, np.ndarray], Any],
    n_bootstrap_samples: int = 1000,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    random_state: Optional[int] = None,
    **model_kwargs
) -> Dict[str, Any]:
    """
    Perform bootstrap resampling to estimate the robustness of a model.

    Parameters:
    -----------
    X : np.ndarray
        Input features (2D array)
    y : np.ndarray
        Target values (1D array)
    model_func : callable
        Function to fit the model (must take X, y and return a fitted model)
    n_bootstrap_samples : int, optional
        Number of bootstrap samples to generate (default: 1000)
    metric : str or callable, optional
        Metric to evaluate the model performance (default: 'mse')
    random_state : int, optional
        Random seed for reproducibility (default: None)
    **model_kwargs :
        Additional keyword arguments to pass to the model function

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': List of metrics for each bootstrap sample
        - 'metrics': Dictionary of aggregated metrics
        - 'params_used': Parameters used in the computation
        - 'warnings': List of warnings encountered

    Example:
    --------
    >>> def linear_regression(X, y):
    ...     # Simple implementation of linear regression
    ...     X = np.c_[np.ones(X.shape[0]), X]
    ...     beta = np.linalg.inv(X.T @ X) @ X.T @ y
    ...     return lambda x: np.dot(np.c_[np.ones(x.shape[0]), x], beta)
    ...
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> results = bootstrap_fit(X, y, linear_regression, n_bootstrap_samples=100)
    """
    _validate_inputs(X, y, n_bootstrap_samples, random_state)

    results = []
    warnings_list = []

    rng = np.random.RandomState(random_state)
    for i in range(n_bootstrap_samples):
        try:
            X_sample, y_sample = _bootstrap_sample(X, y, rng.randint(0, 10000))
            model = _fit_model(X_sample, y_sample, model_func, **model_kwargs)
            y_pred = np.array([model(x) for x in X_sample])
            metric_value = _compute_metric(y_sample, y_pred, metric)
            results.append(metric_value)
        except Exception as e:
            warnings_list.append(f"Sample {i} failed: {str(e)}")

    if not results:
        raise RuntimeError("All bootstrap samples failed. Check your model and data.")

    metrics = {
        'mean': np.mean(results),
        'std': np.std(results),
        'median': np.median(results),
        'min': np.min(results),
        'max': np.max(results)
    }

    return {
        'result': results,
        'metrics': metrics,
        'params_used': {
            'n_bootstrap_samples': n_bootstrap_samples,
            'metric': metric,
            'random_state': random_state,
            **model_kwargs
        },
        'warnings': warnings_list
    }

################################################################################
# resampling
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def resampling_fit(
    X: np.ndarray,
    y: np.ndarray,
    model_func: Callable,
    n_resamples: int = 100,
    resampling_method: str = 'bootstrap',
    metric_func: Callable = None,
    normalization: Optional[str] = None,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform resampling to estimate the robustness of a model.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    model_func : Callable
        Function that takes X and returns predicted y.
    n_resamples : int, optional
        Number of resampling iterations (default: 100).
    resampling_method : str, optional
        Resampling method ('bootstrap', 'kfold') (default: 'bootstrap').
    metric_func : Callable, optional
        Custom metric function. If None, uses default MSE.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: None).
    random_state : int, optional
        Random seed for reproducibility (default: None).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_norm = _apply_normalization(X, normalization)

    # Initialize results storage
    results = {
        'result': [],
        'metrics': [],
        'params_used': {
            'n_resamples': n_resamples,
            'resampling_method': resampling_method,
            'normalization': normalization
        },
        'warnings': []
    }

    # Set random seed if specified
    rng = np.random.RandomState(random_state)

    # Perform resampling
    for _ in range(n_resamples):
        if resampling_method == 'bootstrap':
            indices = rng.choice(len(X), size=len(X), replace=True)
        elif resampling_method == 'kfold':
            indices = _kfold_indices(len(X), n_resamples, rng)
        else:
            raise ValueError(f"Unknown resampling method: {resampling_method}")

        X_resampled = X_norm[indices]
        y_resampled = y[indices]

        # Fit model and compute metrics
        y_pred = model_func(X_resampled)
        metric_value = _compute_metric(y_resampled, y_pred, metric_func)

        results['result'].append(y_pred)
        results['metrics'].append(metric_value)

    return results

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def _apply_normalization(X: np.ndarray, method: Optional[str]) -> np.ndarray:
    """Apply specified normalization to input data."""
    if method is None or method == 'none':
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

def _kfold_indices(n_samples: int, n_splits: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate k-fold indices."""
    indices = rng.permutation(n_samples)
    fold_size = n_samples // n_splits
    return indices[:n_splits * fold_size]

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Optional[Callable] = None
) -> float:
    """Compute specified metric or default to MSE."""
    if metric_func is not None:
        return metric_func(y_true, y_pred)
    else:
        return np.mean((y_true - y_pred) ** 2)

# Example usage:
"""
def example_model(X):
    return np.mean(X, axis=1)

X_example = np.random.rand(100, 5)
y_example = np.random.rand(100)

results = resampling_fit(
    X=X_example,
    y=y_example,
    model_func=example_model,
    n_resamples=10,
    resampling_method='bootstrap',
    normalization='standard'
)
"""

################################################################################
# outliers_impact
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def outliers_impact_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    distance_metric: str = 'euclidean',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, str]]:
    """
    Compute the impact of outliers on model performance.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the input features.
    distance_metric : str
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski').
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Performance metric to evaluate ('mse', 'mae', 'r2', 'logloss') or custom callable.
    solver : str
        Solver to use ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    regularization : Optional[str]
        Regularization type ('none', 'l1', 'l2', 'elasticnet').
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom distance function.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function.

    Returns
    -------
    Dict[str, Union[Dict, float, str]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = outliers_impact_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = normalizer(X)

    # Choose distance metric
    if custom_distance is not None:
        distance_func = custom_distance
    else:
        distance_func = _get_distance_metric(distance_metric)

    # Choose metric
    if custom_metric is not None:
        metric_func = custom_metric
    else:
        metric_func = _get_metric(metric)

    # Choose solver
    if solver == 'closed_form':
        params = _closed_form_solver(X_normalized, y)
    elif solver == 'gradient_descent':
        params = _gradient_descent_solver(X_normalized, y, tol, max_iter)
    elif solver == 'newton':
        params = _newton_solver(X_normalized, y, tol, max_iter)
    elif solver == 'coordinate_descent':
        params = _coordinate_descent_solver(X_normalized, y, tol, max_iter)
    else:
        raise ValueError("Invalid solver specified.")

    # Apply regularization if needed
    if regularization is not None:
        params = _apply_regularization(params, X_normalized, y, regularization)

    # Compute metrics
    metrics = _compute_metrics(y, params['predictions'], metric_func)

    # Compute outlier impact
    result = _compute_outliers_impact(X_normalized, y, params['predictions'], distance_func)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

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

def _get_distance_metric(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get distance metric function."""
    metrics = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance,
        'minkowski': lambda x, y: np.linalg.norm(x - y, ord=3)
    }
    if metric not in metrics:
        raise ValueError(f"Unknown distance metric: {metric}")
    return metrics[metric]

def _get_metric(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get metric function."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared,
        'logloss': _log_loss
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _closed_form_solver(X: np.ndarray, y: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
    """Closed form solution for linear regression."""
    X_tx = np.dot(X.T, X)
    if np.linalg.det(X_tx) == 0:
        raise ValueError("Matrix is singular.")
    params = np.linalg.solve(X_tx, np.dot(X.T, y))
    predictions = np.dot(X, params)
    return {'params': params, 'predictions': predictions}

def _gradient_descent_solver(X: np.ndarray, y: np.ndarray, tol: float, max_iter: int) -> Dict[str, Union[np.ndarray, float]]:
    """Gradient descent solver."""
    n_samples, n_features = X.shape
    params = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradients = 2/n_samples * X.T.dot(X.dot(params) - y)
        new_params = params - learning_rate * gradients
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params

    predictions = np.dot(X, params)
    return {'params': params, 'predictions': predictions}

def _newton_solver(X: np.ndarray, y: np.ndarray, tol: float, max_iter: int) -> Dict[str, Union[np.ndarray, float]]:
    """Newton's method solver."""
    n_samples, n_features = X.shape
    params = np.zeros(n_features)

    for _ in range(max_iter):
        residuals = X.dot(params) - y
        gradient = 2/n_samples * X.T.dot(residuals)
        hessian = 2/n_samples * X.T.dot(X)
        new_params = params - np.linalg.solve(hessian, gradient)
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params

    predictions = np.dot(X, params)
    return {'params': params, 'predictions': predictions}

def _coordinate_descent_solver(X: np.ndarray, y: np.ndarray, tol: float, max_iter: int) -> Dict[str, Union[np.ndarray, float]]:
    """Coordinate descent solver."""
    n_samples, n_features = X.shape
    params = np.zeros(n_features)

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - X.dot(params) + params[j] * X_j
            params[j] = np.sum(X_j * residuals) / np.sum(X_j ** 2)

        if np.linalg.norm(params - np.zeros(n_features)) < tol:
            break

    predictions = np.dot(X, params)
    return {'params': params, 'predictions': predictions}

def _apply_regularization(params: np.ndarray, X: np.ndarray, y: np.ndarray, regularization: str) -> Dict[str, Union[np.ndarray, float]]:
    """Apply regularization to the parameters."""
    if regularization == 'l1':
        params = _l1_regularization(params, X, y)
    elif regularization == 'l2':
        params = _l2_regularization(params, X, y)
    elif regularization == 'elasticnet':
        params = _elasticnet_regularization(params, X, y)
    else:
        raise ValueError("Invalid regularization type.")
    predictions = np.dot(X, params)
    return {'params': params, 'predictions': predictions}

def _l1_regularization(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """L1 regularization."""
    n_samples = X.shape[0]
    alpha = 0.1
    gradients = 2/n_samples * X.T.dot(X.dot(params) - y)
    gradients += alpha * np.sign(params)
    return params - 0.01 * gradients

def _l2_regularization(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """L2 regularization."""
    n_samples = X.shape[0]
    alpha = 0.1
    gradients = 2/n_samples * X.T.dot(X.dot(params) - y)
    gradients += 2 * alpha * params
    return params - 0.01 * gradients

def _elasticnet_regularization(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ElasticNet regularization."""
    n_samples = X.shape[0]
    alpha1, alpha2 = 0.1, 0.1
    gradients = 2/n_samples * X.T.dot(X.dot(params) - y)
    gradients += alpha1 * np.sign(params) + 2 * alpha2 * params
    return params - 0.01 * gradients

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, metric_func: Callable[[np.ndarray, np.ndarray], float]) -> Dict[str, float]:
    """Compute metrics."""
    return {'metric': metric_func(y_true, y_pred)}

def _compute_outliers_impact(X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, distance_func: Callable[[np.ndarray, np.ndarray], float]) -> Dict[str, Union[float, np.ndarray]]:
    """Compute the impact of outliers."""
    residuals = y - y_pred
    distances = np.array([distance_func(x, y_pred) for x in X])
    outlier_indices = np.where(np.abs(residuals) > 2 * np.std(residuals))[0]
    outlier_impact = {
        'outliers': outlier_indices,
        'residuals': residuals[outlier_indices],
        'distances': distances[outlier_indices]
    }
    return outlier_impact

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Log loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Euclidean distance."""
    return np.linalg.norm(x - y)

def _manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Manhattan distance."""
    return np.sum(np.abs(x - y))

def _cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine distance."""
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

################################################################################
# noise_resistance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def noise_resistance_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: str = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, List, str]]:
    """
    Estimate model parameters with noise resistance.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize features.
    metric : str
        Metric to evaluate model performance. Options: 'mse', 'mae', 'r2', 'logloss'.
    distance : str
        Distance metric for model. Options: 'euclidean', 'manhattan', 'cosine', 'minkowski'.
    solver : str
        Solver to use. Options: 'closed_form', 'gradient_descent', 'newton', 'coordinate_descent'.
    regularization : Optional[str]
        Regularization type. Options: None, 'l1', 'l2', 'elasticnet'.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom distance function.

    Returns
    -------
    Dict[str, Union[Dict, List, str]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = noise_resistance_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features
    X_normalized = normalizer(X)

    # Choose metric
    if custom_metric is not None:
        metric_func = custom_metric
    else:
        metric_func = _get_metric(metric)

    # Choose distance
    if custom_distance is not None:
        distance_func = custom_distance
    else:
        distance_func = _get_distance(distance)

    # Choose solver
    if solver == 'closed_form':
        params = _solve_closed_form(X_normalized, y)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(X_normalized, y, distance_func, tol, max_iter)
    elif solver == 'newton':
        params = _solve_newton(X_normalized, y, distance_func, tol, max_iter)
    elif solver == 'coordinate_descent':
        params = _solve_coordinate_descent(X_normalized, y, distance_func, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if specified
    if regularization is not None:
        params = _apply_regularization(params, X_normalized, y, regularization)

    # Calculate metrics
    predictions = _predict(X_normalized, params)
    metrics = {
        'metric': metric_func(y, predictions),
        'mse': _mean_squared_error(y, predictions),
        'mae': _mean_absolute_error(y, predictions),
        'r2': _r_squared(y, predictions)
    }

    # Prepare output
    result = {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': _check_warnings(X_normalized, y)
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
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

def _get_metric(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get metric function based on input string."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared,
        'logloss': _log_loss
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _get_distance(distance: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get distance function based on input string."""
    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance,
        'minkowski': lambda x, y: np.sum(np.abs(x - y) ** 3, axis=1) ** (1/3)
    }
    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")
    return distances[distance]

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve using closed form solution."""
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve using gradient descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ params - y)
        params -= tol * gradient
    return params

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve using Newton's method."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ params - y)
        hessian = 2 * X.T @ X
        params -= np.linalg.pinv(hessian) @ gradient
    return params

def _solve_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve using coordinate descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            params[j] = (y - np.dot(X, params) + params[j] * X_j).T @ X_j / (X_j.T @ X_j)
    return params

def _apply_regularization(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    regularization: str
) -> np.ndarray:
    """Apply regularization to parameters."""
    if regularization == 'l1':
        params -= np.sign(params) * 0.1
    elif regularization == 'l2':
        params -= 0.1 * params
    elif regularization == 'elasticnet':
        params -= (np.sign(params) + 0.1 * params) / 2
    return params

def _predict(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Predict using model parameters."""
    return X @ params

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
    return np.linalg.norm(x - y, axis=1)

def _manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Manhattan distance."""
    return np.sum(np.abs(x - y), axis=1)

def _cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate cosine distance."""
    return 1 - np.sum(x * y, axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))

def _check_warnings(X: np.ndarray, y: np.ndarray) -> List[str]:
    """Check for potential warnings."""
    warnings = []
    if np.any(np.isclose(X.std(axis=0), 0)):
        warnings.append("Some features have zero variance.")
    if np.any(np.isclose(y.std(), 0)):
        warnings.append("Target variable has zero variance.")
    return warnings

################################################################################
# feature_importance
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def feature_importance_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]:
    """
    Compute feature importance using various statistical and machine learning methods.

    Parameters
    ----------
    X : np.ndarray
        Input features matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the input features. Default is identity.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Metric to evaluate model performance. Can be "mse", "mae", "r2", or a custom callable.
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Distance metric for feature importance calculation. Can be "euclidean", "manhattan",
        "cosine", or a custom callable.
    solver : str
        Solver to use for optimization. Options: "closed_form", "gradient_descent",
        "newton", or "coordinate_descent".
    regularization : Optional[str]
        Regularization type. Options: None, "l1", "l2", or "elasticnet".
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function if not using built-in metrics.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom distance function if not using built-in distances.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]
        Dictionary containing:
        - "result": Feature importance scores.
        - "metrics": Computed metrics.
        - "params_used": Parameters used in the computation.
        - "warnings": Any warnings encountered.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = feature_importance_fit(X, y, normalizer=np.std, metric="r2", solver="gradient_descent")
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features
    X_normalized = normalizer(X)

    # Set metric and distance functions
    metric_func, distance_func = _set_metric_and_distance(metric, distance,
                                                         custom_metric, custom_distance)

    # Solve for feature importance
    if solver == "closed_form":
        result = _closed_form_solution(X_normalized, y, distance_func)
    elif solver == "gradient_descent":
        result = _gradient_descent(X_normalized, y, distance_func,
                                  regularization=regularization,
                                  tol=tol, max_iter=max_iter)
    elif solver == "newton":
        result = _newton_method(X_normalized, y, distance_func,
                               regularization=regularization,
                               tol=tol, max_iter=max_iter)
    elif solver == "coordinate_descent":
        result = _coordinate_descent(X_normalized, y, distance_func,
                                    regularization=regularization,
                                    tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(y, result, metric_func)

    return {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if hasattr(normalizer, '__name__') else "custom",
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
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

def _set_metric_and_distance(
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> tuple:
    """Set metric and distance functions based on user input."""
    if isinstance(metric, str):
        if metric == "mse":
            metric_func = _mse
        elif metric == "mae":
            metric_func = _mae
        elif metric == "r2":
            metric_func = _r2
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        metric_func = metric
    else:
        raise TypeError("Metric must be a string or callable")

    if isinstance(distance, str):
        if distance == "euclidean":
            distance_func = _euclidean_distance
        elif distance == "manhattan":
            distance_func = _manhattan_distance
        elif distance == "cosine":
            distance_func = _cosine_distance
        else:
            raise ValueError(f"Unknown distance: {distance}")
    elif callable(distance):
        distance_func = distance
    else:
        raise TypeError("Distance must be a string or callable")

    if custom_metric is not None:
        metric_func = custom_metric
    if custom_distance is not None:
        distance_func = custom_distance

    return metric_func, distance_func

def _closed_form_solution(X: np.ndarray, y: np.ndarray,
                         distance_func: Callable) -> np.ndarray:
    """Compute feature importance using closed form solution."""
    # This is a placeholder for the actual implementation
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    distance_func: Callable,
    *,
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Compute feature importance using gradient descent."""
    # This is a placeholder for the actual implementation
    return np.random.rand(X.shape[1])

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    distance_func: Callable,
    *,
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Compute feature importance using Newton's method."""
    # This is a placeholder for the actual implementation
    return np.random.rand(X.shape[1])

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    distance_func: Callable,
    *,
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Compute feature importance using coordinate descent."""
    # This is a placeholder for the actual implementation
    return np.random.rand(X.shape[1])

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable
) -> Dict[str, float]:
    """Compute metrics for the model."""
    return {"metric": metric_func(y_true, y_pred)}

def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance."""
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

################################################################################
# sensitivity_analysis
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def sensitivity_analysis_fit(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable] = None,
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_params: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform sensitivity analysis on a given model.

    Parameters:
    -----------
    model : Callable
        The model to analyze. Must be a callable that takes X and returns predictions.
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    normalizer : Optional[Callable], default=None
        Function to normalize the input data. If None, no normalization is applied.
    metric : Union[str, Callable], default='mse'
        Metric to evaluate the model. Can be 'mse', 'mae', 'r2', 'logloss', or a custom callable.
    distance : Union[str, Callable], default='euclidean'
        Distance metric for sensitivity analysis. Can be 'euclidean', 'manhattan', 'cosine',
        'minkowski', or a custom callable.
    solver : str, default='gradient_descent'
        Solver to use for optimization. Options: 'closed_form', 'gradient_descent',
        'newton', 'coordinate_descent'.
    regularization : Optional[str], default=None
        Regularization type. Options: None, 'l1', 'l2', 'elasticnet'.
    tol : float, default=1e-4
        Tolerance for convergence.
    max_iter : int, default=1000
        Maximum number of iterations.
    custom_params : Optional[Dict[str, Any]], default=None
        Additional parameters for the model or solver.
    **kwargs : dict
        Additional keyword arguments passed to the model or solver.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the results of the sensitivity analysis.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Initialize model and solver
    params_used = {
        'normalizer': normalizer.__name__ if normalizer else None,
        'metric': metric,
        'distance': distance,
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'max_iter': max_iter
    }

    # Fit the model using the specified solver
    if solver == 'closed_form':
        predictions = _fit_closed_form(model, X_normalized, y, regularization, **kwargs)
    elif solver == 'gradient_descent':
        predictions = _fit_gradient_descent(model, X_normalized, y, metric, regularization,
                                           tol, max_iter, **kwargs)
    elif solver == 'newton':
        predictions = _fit_newton(model, X_normalized, y, metric, regularization,
                                 tol, max_iter, **kwargs)
    elif solver == 'coordinate_descent':
        predictions = _fit_coordinate_descent(model, X_normalized, y, metric, regularization,
                                             tol, max_iter, **kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(y, predictions, metric)

    # Perform sensitivity analysis
    sensitivity_results = _perform_sensitivity_analysis(X_normalized, y, predictions,
                                                       distance)

    # Prepare the output
    result = {
        'result': sensitivity_results,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': []
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
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values.")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to the input data."""
    if normalizer is None:
        return X
    return normalizer(X)

def _fit_closed_form(model: Callable, X: np.ndarray, y: np.ndarray,
                    regularization: Optional[str], **kwargs) -> np.ndarray:
    """Fit the model using closed-form solution."""
    # Placeholder for closed-form implementation
    return model(X, **kwargs)

def _fit_gradient_descent(model: Callable, X: np.ndarray, y: np.ndarray,
                         metric: Union[str, Callable], regularization: Optional[str],
                         tol: float, max_iter: int, **kwargs) -> np.ndarray:
    """Fit the model using gradient descent."""
    # Placeholder for gradient descent implementation
    return model(X, **kwargs)

def _fit_newton(model: Callable, X: np.ndarray, y: np.ndarray,
               metric: Union[str, Callable], regularization: Optional[str],
               tol: float, max_iter: int, **kwargs) -> np.ndarray:
    """Fit the model using Newton's method."""
    # Placeholder for Newton's method implementation
    return model(X, **kwargs)

def _fit_coordinate_descent(model: Callable, X: np.ndarray, y: np.ndarray,
                           metric: Union[str, Callable], regularization: Optional[str],
                           tol: float, max_iter: int, **kwargs) -> np.ndarray:
    """Fit the model using coordinate descent."""
    # Placeholder for coordinate descent implementation
    return model(X, **kwargs)

def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      metric: Union[str, Callable]) -> Dict[str, float]:
    """Calculate the specified metrics."""
    metrics = {}
    if metric == 'mse':
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)
    elif metric == 'logloss':
        metrics['logloss'] = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    elif callable(metric):
        metrics['custom'] = metric(y_true, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics

def _perform_sensitivity_analysis(X: np.ndarray, y: np.ndarray,
                                predictions: np.ndarray,
                                distance: Union[str, Callable]) -> Dict[str, Any]:
    """Perform sensitivity analysis using the specified distance metric."""
    sensitivity_results = {}
    if distance == 'euclidean':
        sensitivity_results['distance'] = np.linalg.norm(X - predictions, axis=1)
    elif distance == 'manhattan':
        sensitivity_results['distance'] = np.sum(np.abs(X - predictions), axis=1)
    elif distance == 'cosine':
        sensitivity_results['distance'] = 1 - np.sum(X * predictions, axis=1) / (
            np.linalg.norm(X, axis=1) * np.linalg.norm(predictions, axis=1))
    elif distance == 'minkowski':
        p = 3  # Default Minkowski parameter
        sensitivity_results['distance'] = np.sum(np.abs(X - predictions) ** p, axis=1) ** (1/p)
    elif callable(distance):
        sensitivity_results['custom_distance'] = distance(X, predictions)
    else:
        raise ValueError(f"Unknown distance metric: {distance}")
    return sensitivity_results

################################################################################
# perturbation_tests
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def perturbation_tests_fit(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    n_perturbations: int = 100,
    perturbation_scale: float = 0.01,
    normalization: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None,
    custom_metric_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Perform perturbation tests on a given model to assess its robustness.

    Parameters:
    -----------
    model : Callable
        The model to test. Must be a callable that takes X and y as input.
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    n_perturbations : int, optional
        Number of perturbations to perform. Default is 100.
    perturbation_scale : float, optional
        Scale of the perturbations. Default is 0.01.
    normalization : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'. Default is 'standard'.
    metric : Union[str, Callable], optional
        Metric to evaluate the model: 'mse', 'mae', 'r2', 'logloss', or a custom callable. Default is 'mse'.
    distance : str, optional
        Distance metric for perturbations: 'euclidean', 'manhattan', 'cosine', or 'minkowski'. Default is 'euclidean'.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'. Default is 'closed_form'.
    regularization : Optional[str], optional
        Regularization method: None, 'l1', 'l2', or 'elasticnet'. Default is None.
    tol : float, optional
        Tolerance for convergence. Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    random_state : Optional[int], optional
        Random seed for reproducibility. Default is None.
    custom_metric_kwargs : Optional[Dict[str, Any]], optional
        Additional keyword arguments for the custom metric. Default is None.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize data
    X_norm, y_norm = _normalize_data(X, y, normalization)

    # Initialize results dictionary
    results = {
        'result': [],
        'metrics': [],
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    # Perform perturbation tests
    for _ in range(n_perturbations):
        # Perturb the data
        X_perturbed = _perturb_data(X_norm, perturbation_scale, distance)

        # Fit the model
        params = _fit_model(model, X_perturbed, y_norm, solver, regularization, tol, max_iter)

        # Compute metrics
        metric_value = _compute_metric(model, X_norm, y_norm, metric, custom_metric_kwargs)

        # Store results
        results['result'].append(params)
        results['metrics'].append(metric_value)

    return results

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate the input data."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("X and y must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("X and y must not contain infinite values.")

def _normalize_data(
    X: np.ndarray,
    y: np.ndarray,
    normalization: str
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize the input data."""
    if normalization == 'standard':
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y_norm = (y - np.mean(y)) / np.std(y)
    elif normalization == 'minmax':
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif normalization == 'robust':
        X_norm = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        y_norm = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    else:
        X_norm = X.copy()
        y_norm = y.copy()
    return X_norm, y_norm

def _perturb_data(
    X: np.ndarray,
    scale: float,
    distance: str
) -> np.ndarray:
    """Perturb the input data."""
    perturbation = np.random.normal(0, scale, size=X.shape)
    if distance == 'euclidean':
        return X + perturbation
    elif distance == 'manhattan':
        return X + np.sign(perturbation) * np.abs(perturbation)
    elif distance == 'cosine':
        norm_X = np.linalg.norm(X, axis=1, keepdims=True)
        norm_perturbation = np.linalg.norm(perturbation, axis=1, keepdims=True)
        return (X / norm_X) * (norm_X + perturbation)
    elif distance == 'minkowski':
        return X + np.sign(perturbation) * np.abs(perturbation) ** 2
    else:
        return X + perturbation

def _fit_model(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Fit the model using the specified solver and regularization."""
    if solver == 'closed_form':
        params = _fit_closed_form(model, X, y)
    elif solver == 'gradient_descent':
        params = _fit_gradient_descent(model, X, y, tol, max_iter)
    elif solver == 'newton':
        params = _fit_newton(model, X, y, tol, max_iter)
    elif solver == 'coordinate_descent':
        params = _fit_coordinate_descent(model, X, y, tol, max_iter)
    else:
        raise ValueError("Invalid solver specified.")

    if regularization is not None:
        params = _apply_regularization(params, X, y, regularization)

    return params

def _fit_closed_form(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, Any]:
    """Fit the model using closed-form solution."""
    # Placeholder for actual implementation
    return {'params': np.linalg.pinv(X) @ y}

def _fit_gradient_descent(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Fit the model using gradient descent."""
    # Placeholder for actual implementation
    return {'params': np.zeros(X.shape[1])}

def _fit_newton(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Fit the model using Newton's method."""
    # Placeholder for actual implementation
    return {'params': np.zeros(X.shape[1])}

def _fit_coordinate_descent(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Fit the model using coordinate descent."""
    # Placeholder for actual implementation
    return {'params': np.zeros(X.shape[1])}

def _apply_regularization(
    params: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    regularization: str
) -> Dict[str, Any]:
    """Apply regularization to the model parameters."""
    if regularization == 'l1':
        params['params'] = _apply_l1_regularization(params['params'], X, y)
    elif regularization == 'l2':
        params['params'] = _apply_l2_regularization(params['params'], X, y)
    elif regularization == 'elasticnet':
        params['params'] = _apply_elasticnet_regularization(params['params'], X, y)
    return params

def _apply_l1_regularization(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """Apply L1 regularization."""
    # Placeholder for actual implementation
    return params

def _apply_l2_regularization(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """Apply L2 regularization."""
    # Placeholder for actual implementation
    return params

def _apply_elasticnet_regularization(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """Apply elastic net regularization."""
    # Placeholder for actual implementation
    return params

def _compute_metric(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    custom_metric_kwargs: Optional[Dict[str, Any]]
) -> float:
    """Compute the specified metric."""
    if isinstance(metric, str):
        if metric == 'mse':
            return _compute_mse(model, X, y)
        elif metric == 'mae':
            return _compute_mae(model, X, y)
        elif metric == 'r2':
            return _compute_r2(model, X, y)
        elif metric == 'logloss':
            return _compute_logloss(model, X, y)
        else:
            raise ValueError("Invalid metric specified.")
    else:
        return metric(model, X, y, **(custom_metric_kwargs or {}))

def _compute_mse(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray
) -> float:
    """Compute the mean squared error."""
    predictions = model(X)
    return np.mean((predictions - y) ** 2)

def _compute_mae(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray
) -> float:
    """Compute the mean absolute error."""
    predictions = model(X)
    return np.mean(np.abs(predictions - y))

def _compute_r2(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray
) -> float:
    """Compute the R-squared metric."""
    predictions = model(X)
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

def _compute_logloss(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray
) -> float:
    """Compute the log loss."""
    predictions = model(X)
    return -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

################################################################################
# robustness_metrics
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List, Any

def robustness_metrics_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    normalization: str = "none",
    metrics: Union[str, List[str], Callable] = ["mse", "mae"],
    distance: str = "euclidean",
    solver: Optional[str] = None,
    regularization: str = "none",
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    weights: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute robustness metrics for model predictions.

    Parameters:
    -----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values from the model.
    normalization : str, optional (default="none")
        Normalization method: "none", "standard", "minmax", or "robust".
    metrics : str, List[str], or Callable, optional (default=["mse", "mae"])
        Metrics to compute. Can be a single string, list of strings, or custom callable.
    distance : str, optional (default="euclidean")
        Distance metric for robustness calculations: "euclidean", "manhattan", etc.
    solver : str, optional (default=None)
        Solver method for optimization: "gradient_descent", "newton", etc.
    regularization : str, optional (default="none")
        Regularization type: "none", "l1", "l2", or "elasticnet".
    tol : float, optional (default=1e-4)
        Tolerance for convergence.
    max_iter : int, optional (default=1000)
        Maximum number of iterations.
    custom_metric : Callable, optional (default=None)
        Custom metric function to use.
    weights : np.ndarray, optional (default=None)
        Weights for weighted metrics.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Normalize data if required
    y_true_norm, y_pred_norm = _apply_normalization(y_true, y_pred, normalization)

    # Compute metrics
    computed_metrics = _compute_metrics(
        y_true_norm, y_pred_norm,
        metrics=metrics,
        custom_metric=custom_metric,
        weights=weights
    )

    # Prepare output dictionary
    result = {
        "result": computed_metrics,
        "metrics": list(computed_metrics.keys()),
        "params_used": {
            "normalization": normalization,
            "metrics": metrics,
            "distance": distance,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    return result

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or infinite values.")
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError("y_pred contains NaN or infinite values.")

def _apply_normalization(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply normalization to input arrays."""
    if method == "none":
        return y_true, y_pred
    elif method == "standard":
        mean = np.mean(y_true)
        std = np.std(y_true)
        y_true_norm = (y_true - mean) / (std + 1e-8)
        y_pred_norm = (y_pred - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(y_true)
        max_val = np.max(y_true)
        y_true_norm = (y_true - min_val) / ((max_val - min_val) + 1e-8)
        y_pred_norm = (y_pred - min_val) / ((max_val - min_val) + 1e-8)
    elif method == "robust":
        median = np.median(y_true)
        iqr = np.percentile(y_true, 75) - np.percentile(y_true, 25)
        y_true_norm = (y_true - median) / (iqr + 1e-8)
        y_pred_norm = (y_pred - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return y_true_norm, y_pred_norm

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metrics: Union[str, List[str], Callable],
    custom_metric: Optional[Callable] = None,
    weights: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute specified metrics."""
    if isinstance(metrics, str):
        metrics = [metrics]
    elif callable(metrics):
        return {str(metrics).__name__: metrics(y_true, y_pred)}

    computed_metrics = {}

    for metric in metrics:
        if metric == "mse":
            computed_metrics["mse"] = _mean_squared_error(y_true, y_pred, weights)
        elif metric == "mae":
            computed_metrics["mae"] = _mean_absolute_error(y_true, y_pred, weights)
        elif metric == "r2":
            computed_metrics["r2"] = _r_squared(y_true, y_pred)
        elif metric == "logloss":
            computed_metrics["logloss"] = _log_loss(y_true, y_pred)
        elif callable(metric):
            computed_metrics[str(metric).__name__] = metric(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    if custom_metric is not None:
        computed_metrics["custom"] = custom_metric(y_true, y_pred)

    return computed_metrics

def _mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> float:
    """Compute mean squared error."""
    residuals = y_true - y_pred
    if weights is not None:
        residuals = residuals * weights
    return np.mean(residuals ** 2)

def _mean_absolute_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> float:
    """Compute mean absolute error."""
    residuals = np.abs(y_true - y_pred)
    if weights is not None:
        residuals = residuals * weights
    return np.mean(residuals)

def _r_squared(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """Compute R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))

def _log_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """Compute log loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
