"""
Quantix – Module regression_lineaire
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# modele_lineaire
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Literal

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
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

def _normalize_data(X: np.ndarray, y: np.ndarray,
                   normalization: Literal['none', 'standard', 'minmax', 'robust'] = 'none') -> tuple:
    """Normalize input data."""
    X_normalized = X.copy()
    y_normalized = y.copy()

    if normalization == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_normalized = (X - X_mean) / X_std
    elif normalization == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_normalized = (X - X_min) / (X_max - X_min + 1e-8)
    elif normalization == 'robust':
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_normalized = (X - X_median) / (X_iqr + 1e-8)

    return X_normalized, y_normalized

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metric_functions: Dict[str, Callable] = None) -> Dict[str, float]:
    """Compute regression metrics."""
    if metric_functions is None:
        metric_functions = {
            'mse': lambda yt, yp: np.mean((yt - yp)**2),
            'mae': lambda yt, yp: np.mean(np.abs(yt - yp)),
            'r2': lambda yt, yp: 1 - np.sum((yt - yp)**2)/np.sum((yt - np.mean(yt))**2)
        }

    metrics = {}
    for name, func in metric_functions.items():
        try:
            metrics[name] = func(y_true, y_pred)
        except Exception as e:
            metrics[f"{name}_error"] = str(e)

    return metrics

def _closed_form_solver(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed form solution for linear regression."""
    return np.linalg.inv(X.T @ X) @ X.T @ y

def _gradient_descent_solver(X: np.ndarray, y: np.ndarray,
                           learning_rate: float = 0.01,
                           max_iter: int = 1000,
                           tol: float = 1e-4) -> np.ndarray:
    """Gradient descent solver for linear regression."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    prev_loss = float('inf')

    for _ in range(max_iter):
        gradients = 2/n_samples * X.T @ (X @ weights - y)
        weights -= learning_rate * gradients
        current_loss = np.mean((X @ weights - y)**2)

        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return weights

def modele_lineaire_fit(X: np.ndarray, y: np.ndarray,
                      solver: Literal['closed_form', 'gradient_descent'] = 'closed_form',
                      normalization: Literal['none', 'standard', 'minmax', 'robust'] = 'none',
                      metric_functions: Optional[Dict[str, Callable]] = None,
                      **solver_kwargs) -> Dict:
    """Fit a linear regression model.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
        solver: Solver to use
        normalization: Normalization method
        metric_functions: Dictionary of custom metrics
        **solver_kwargs: Additional solver-specific arguments

    Returns:
        Dictionary containing results, metrics, parameters used and warnings
    """
    _validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = _normalize_data(X, y, normalization)

    # Add intercept term
    X_with_intercept = np.column_stack([np.ones(X_norm.shape[0]), X_norm])

    # Solve for weights
    if solver == 'closed_form':
        weights = _closed_form_solver(X_with_intercept, y_norm)
    elif solver == 'gradient_descent':
        weights = _gradient_descent_solver(X_with_intercept, y_norm, **solver_kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Make predictions
    y_pred = X_with_intercept @ weights

    # Compute metrics
    metrics = _compute_metrics(y_norm, y_pred, metric_functions)

    return {
        'result': {
            'weights': weights,
            'intercept': weights[0],
            'coefficients': weights[1:],
            'predictions': y_pred
        },
        'metrics': metrics,
        'params_used': {
            'solver': solver,
            'normalization': normalization,
            'metric_functions': list(metric_functions.keys()) if metric_functions else None
        },
        'warnings': []
    }

# Example usage:
"""
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([1, 2, 3])

result = modele_lineaire_fit(X, y,
                           solver='closed_form',
                           normalization='standard')
"""

################################################################################
# coefficient_regression
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def coefficient_regression_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    solver: str = 'closed_form',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Compute linear regression coefficients with configurable options.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    normalizer : Callable[[np.ndarray], np.ndarray], optional
        Function to normalize features, by default None
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent'), by default 'closed_form'
    metric : Union[str, Callable], optional
        Metric to compute ('mse', 'mae', 'r2'), by default 'mse'
    regularization : str, optional
        Regularization type (None, 'l1', 'l2', 'elasticnet'), by default None
    alpha : float, optional
        Regularization strength, by default 1.0
    l1_ratio : float, optional
        ElasticNet mixing parameter (0 <= l1_ratio <= 1), by default 0.5
    max_iter : int, optional
        Maximum number of iterations for iterative solvers, by default 1000
    tol : float, optional
        Tolerance for stopping criteria, by default 1e-4
    random_state : int, optional
        Random seed for reproducibility, by default None

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': Estimated coefficients
        - 'metrics': Computed metrics
        - 'params_used': Parameters used in computation
        - 'warnings': Any warnings generated

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> y = np.array([5, 6])
    >>> result = coefficient_regression_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if requested
    X_normalized = _apply_normalization(X, normalizer)

    # Solve for coefficients
    if solver == 'closed_form':
        coef = _solve_closed_form(X_normalized, y, regularization, alpha, l1_ratio)
    else:
        coef = _solve_iterative(X_normalized, y,
                              solver=solver,
                              metric=metric,
                              regularization=regularization,
                              alpha=alpha,
                              l1_ratio=l1_ratio,
                              max_iter=max_iter,
                              tol=tol,
                              random_state=random_state)

    # Compute metrics
    y_pred = X_normalized @ coef
    metrics = _compute_metrics(y, y_pred, metric)

    # Prepare output
    result = {
        'result': coef,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'solver': solver,
            'metric': metric if isinstance(metric, str) else None,
            'regularization': regularization,
            'alpha': alpha,
            'l1_ratio': l1_ratio
        },
        'warnings': _check_warnings(X_normalized, y)
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

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to features."""
    if normalizer is None:
        return X
    try:
        return normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _solve_closed_form(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str],
    alpha: float,
    l1_ratio: float
) -> np.ndarray:
    """Solve linear regression using closed-form solution."""
    if regularization is None:
        return np.linalg.pinv(X.T @ X) @ X.T @ y
    elif regularization == 'l2':
        return _solve_ridge(X, y, alpha)
    elif regularization == 'l1':
        return _solve_lasso(X, y, alpha)
    elif regularization == 'elasticnet':
        return _solve_elasticnet(X, y, alpha, l1_ratio)
    else:
        raise ValueError(f"Unknown regularization type: {regularization}")

def _solve_iterative(
    X: np.ndarray,
    y: np.ndarray,
    *,
    solver: str,
    metric: Union[str, Callable],
    regularization: Optional[str],
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tol: float,
    random_state: Optional[int]
) -> np.ndarray:
    """Solve linear regression using iterative methods."""
    if solver == 'gradient_descent':
        return _gradient_descent(X, y,
                               metric=metric,
                               regularization=regularization,
                               alpha=alpha,
                               l1_ratio=l1_ratio,
                               max_iter=max_iter,
                               tol=tol,
                               random_state=random_state)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """Compute regression metrics."""
    if isinstance(metric, str):
        if metric == 'mse':
            return {'mse': np.mean((y_true - y_pred) ** 2)}
        elif metric == 'mae':
            return {'mae': np.mean(np.abs(y_true - y_pred))}
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return {'r2': 1 - ss_res / ss_tot}
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        return {'custom': metric(y_true, y_pred)}

def _check_warnings(X: np.ndarray, y: np.ndarray) -> Dict[str, str]:
    """Check for potential issues and generate warnings."""
    warnings = {}
    if np.linalg.cond(X) > 1e6:
        warnings['ill_conditioned'] = "Input matrix appears ill-conditioned"
    if np.std(y) < 1e-6:
        warnings['small_variance'] = "Target variable has very small variance"
    return warnings

# Additional internal functions would be implemented here
def _solve_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Solve ridge regression."""
    return np.linalg.pinv(X.T @ X + alpha * np.eye(X.shape[1])) @ X.T @ y

def _solve_lasso(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Solve Lasso regression using coordinate descent."""
    # Implementation would go here
    raise NotImplementedError("Lasso solver not implemented")

def _solve_elasticnet(X: np.ndarray, y: np.ndarray, alpha: float, l1_ratio: float) -> np.ndarray:
    """Solve ElasticNet regression."""
    # Implementation would go here
    raise NotImplementedError("ElasticNet solver not implemented")

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    metric: Union[str, Callable],
    regularization: Optional[str],
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tol: float,
    random_state: Optional[int]
) -> np.ndarray:
    """Perform gradient descent for linear regression."""
    # Implementation would go here
    raise NotImplementedError("Gradient descent solver not implemented")

################################################################################
# interception
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def interception_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalize: str = "none",
    metric: Union[str, Callable] = "mse",
    solver: str = "closed_form",
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fit an interception model to the data using linear regression.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    normalize : str, optional
        Normalization method: "none", "standard", "minmax", or "robust".
    metric : str or callable, optional
        Metric to evaluate: "mse", "mae", "r2", or custom callable.
    solver : str, optional
        Solver method: "closed_form", "gradient_descent", or other.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    custom_distance : callable, optional
        Custom distance function for solver.

    Returns
    -------
    Dict
        Dictionary containing:
        - "result": fitted model parameters.
        - "metrics": computed metrics.
        - "params_used": parameters used during fitting.
        - "warnings": any warnings encountered.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> y = np.array([5, 6])
    >>> result = interception_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_norm, y_norm = _apply_normalization(X, y, normalize)

    # Choose solver
    if solver == "closed_form":
        params = _solve_closed_form(X_norm, y_norm)
    else:
        raise ValueError(f"Solver {solver} not implemented.")

    # Compute metrics
    metrics = _compute_metrics(X_norm, y_norm, params, metric, custom_metric)

    return {
        "result": {"coefficients": params},
        "metrics": metrics,
        "params_used": {
            "normalize": normalize,
            "metric": metric,
            "solver": solver,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values.")

def _apply_normalization(
    X: np.ndarray,
    y: np.ndarray,
    method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply normalization to input data."""
    if method == "standard":
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_norm = (X - X_mean) / X_std
    elif method == "minmax":
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_norm = (X - X_min) / (X_max - X_min)
    elif method == "robust":
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - X_median) / X_iqr
    else:
        X_norm = X.copy()

    return X_norm, y.copy()

def _solve_closed_form(
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """Solve linear regression using closed-form solution."""
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
    params = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
    return params

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Compute metrics for the fitted model."""
    y_pred = X @ params[1:] + params[0]

    metrics_dict = {}

    if metric == "mse" or (custom_metric is None and metric == "mse"):
        metrics_dict["mse"] = np.mean((y - y_pred) ** 2)
    elif metric == "mae" or (custom_metric is None and metric == "mae"):
        metrics_dict["mae"] = np.mean(np.abs(y - y_pred))
    elif metric == "r2" or (custom_metric is None and metric == "r2"):
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics_dict["r2"] = 1 - (ss_res / ss_tot)
    elif callable(metric) or custom_metric:
        if custom_metric:
            metrics_dict["custom"] = custom_metric(y, y_pred)
        else:
            metrics_dict["custom"] = metric(y, y_pred)

    return metrics_dict

################################################################################
# erreur_residuelle
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def erreur_residuelle_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalisation: str = 'none',
    metrique: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solveur: str = 'closed_form',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metrique: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_solveur: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Calcule l'erreur résiduelle pour une régression linéaire.

    Parameters:
    -----------
    X : np.ndarray
        Matrice des caractéristiques (n_samples, n_features).
    y : np.ndarray
        Vecteur cible (n_samples,).
    normalisation : str, optional
        Type de normalisation ('none', 'standard', 'minmax', 'robust').
    metrique : str or callable, optional
        Métrique à utiliser ('mse', 'mae', 'r2', 'logloss') ou fonction personnalisée.
    solveur : str, optional
        Solveur à utiliser ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    tol : float, optional
        Tolérance pour la convergence.
    max_iter : int, optional
        Nombre maximal d'itérations.
    custom_metrique : callable, optional
        Fonction personnalisée pour calculer la métrique.
    custom_solveur : callable, optional
        Fonction personnalisée pour le solveur.

    Returns:
    --------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionnaire contenant les résultats, métriques, paramètres utilisés et avertissements.
    """
    # Validation des entrées
    _validate_inputs(X, y)

    # Normalisation
    X_norm, y_norm = _apply_normalisation(X, y, normalisation)

    # Choix du solveur
    if custom_solveur is not None:
        coefficients = custom_solveur(X_norm, y_norm)
    else:
        if solveur == 'closed_form':
            coefficients = _solve_closed_form(X_norm, y_norm)
        elif solveur == 'gradient_descent':
            coefficients = _solve_gradient_descent(X_norm, y_norm, tol=tol, max_iter=max_iter)
        elif solveur == 'newton':
            coefficients = _solve_newton(X_norm, y_norm, tol=tol, max_iter=max_iter)
        elif solveur == 'coordinate_descent':
            coefficients = _solve_coordinate_descent(X_norm, y_norm, tol=tol, max_iter=max_iter)
        else:
            raise ValueError("Solveur non reconnu.")

    # Calcul des résidus
    residus = y_norm - X_norm @ coefficients

    # Choix de la métrique
    if custom_metrique is not None:
        metrique_value = custom_metrique(y_norm, X_norm @ coefficients)
    else:
        if metrique == 'mse':
            metrique_value = _compute_mse(y_norm, X_norm @ coefficients)
        elif metrique == 'mae':
            metrique_value = _compute_mae(y_norm, X_norm @ coefficients)
        elif metrique == 'r2':
            metrique_value = _compute_r2(y_norm, X_norm @ coefficients)
        elif metrique == 'logloss':
            metrique_value = _compute_logloss(y_norm, X_norm @ coefficients)
        else:
            raise ValueError("Métrique non reconnue.")

    # Retour des résultats
    return {
        'result': coefficients,
        'metrics': {'erreur_residuelle': metrique_value},
        'params_used': {
            'normalisation': normalisation,
            'metrique': metrique if custom_metrique is None else 'custom',
            'solveur': solveur if custom_solveur is None else 'custom'
        },
        'warnings': _check_warnings(residus)
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Valide les entrées X et y."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X et y doivent être des tableaux NumPy.")
    if X.ndim != 2:
        raise ValueError("X doit être une matrice 2D.")
    if y.ndim != 1:
        raise ValueError("y doit être un vecteur 1D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X et y doivent avoir le même nombre d'échantillons.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("X et y ne doivent pas contenir de NaN.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("X et y ne doivent pas contenir de valeurs infinies.")

def _apply_normalisation(X: np.ndarray, y: np.ndarray, normalisation: str) -> tuple:
    """Applique la normalisation choisie."""
    if normalisation == 'standard':
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y_norm = (y - np.mean(y)) / np.std(y)
    elif normalisation == 'minmax':
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif normalisation == 'robust':
        X_norm = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        y_norm = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    else:
        X_norm, y_norm = X.copy(), y.copy()
    return X_norm, y_norm

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Résolution par la forme fermée."""
    return np.linalg.inv(X.T @ X) @ X.T @ y

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> np.ndarray:
    """Résolution par descente de gradient."""
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)
    learning_rate = 0.01
    for _ in range(max_iter):
        gradient = -2 * X.T @ (y - X @ coefficients) / y.size
        new_coefficients = coefficients - learning_rate * gradient
        if np.linalg.norm(new_coefficients - coefficients) < tol:
            break
        coefficients = new_coefficients
    return coefficients

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> np.ndarray:
    """Résolution par méthode de Newton."""
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)
    for _ in range(max_iter):
        residuals = y - X @ coefficients
        gradient = -2 * X.T @ residuals / y.size
        hessian = 2 * X.T @ X / y.size
        new_coefficients = coefficients - np.linalg.inv(hessian) @ gradient
        if np.linalg.norm(new_coefficients - coefficients) < tol:
            break
        coefficients = new_coefficients
    return coefficients

def _solve_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> np.ndarray:
    """Résolution par descente de coordonnées."""
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)
    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - X @ coefficients + coefficients[j] * X_j
            coefficients[j] = np.sum(X_j * residuals) / np.sum(X_j ** 2)
        if np.linalg.norm(coefficients - coefficients) < tol:
            break
    return coefficients

def _compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule l'erreur quadratique moyenne (MSE)."""
    return np.mean((y_true - y_pred) ** 2)

def _compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule l'erreur absolue moyenne (MAE)."""
    return np.mean(np.abs(y_true - y_pred))

def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule le coefficient de détermination (R²)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def _compute_logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule la log-loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _check_warnings(residus: np.ndarray) -> list:
    """Vérifie les avertissements potentiels."""
    warnings = []
    if np.any(np.isnan(residus)):
        warnings.append("Les résidus contiennent des NaN.")
    if np.any(np.isinf(residus)):
        warnings.append("Les résidus contiennent des valeurs infinies.")
    return warnings

################################################################################
# moindres_carres
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input matrices for linear regression."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be a 2D array and y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def normalize_data(X: np.ndarray, y: Optional[np.ndarray], method: str) -> tuple:
    """Normalize data according to specified method."""
    if method == 'none':
        return X, y
    elif method == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_normalized = (X - X_mean) / X_std
        if y is not None:
            y_mean = np.mean(y)
            y_std = np.std(y)
            return X_normalized, (y - y_mean) / y_std
        return X_normalized, None
    elif method == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_normalized = (X - X_min) / (X_max - X_min + 1e-8)
        if y is not None:
            y_min = np.min(y)
            y_max = np.max(y)
            return X_normalized, (y - y_min) / (y_max - y_min + 1e-8)
        return X_normalized, None
    elif method == 'robust':
        X_median = np.median(X, axis=0)
        X_q75, X_q25 = np.percentile(X, [75, 25], axis=0)
        X_iqr = X_q75 - X_q25
        X_normalized = (X - X_median) / (X_iqr + 1e-8)
        if y is not None:
            y_median = np.median(y)
            y_q75, y_q25 = np.percentile(y, [75, 25])
            y_iqr = y_q75 - y_q25
            return X_normalized, (y - y_median) / (y_iqr + 1e-8)
        return X_normalized, None
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metrics: Union[str, Callable]) -> Dict[str, float]:
    """Compute specified metrics between true and predicted values."""
    result = {}
    if isinstance(metrics, str):
        if metrics == 'mse':
            result['mse'] = np.mean((y_true - y_pred) ** 2)
        elif metrics == 'mae':
            result['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metrics == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            result['r2'] = 1 - (ss_res / ss_tot)
        elif metrics == 'logloss':
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            result['logloss'] = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        result['custom'] = metrics(y_true, y_pred)
    return result

def closed_form_solver(X: np.ndarray, y: np.ndarray,
                       regularization: str = 'none',
                       alpha: float = 1.0) -> np.ndarray:
    """Solve linear regression using closed form solution."""
    if regularization == 'none':
        coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
    elif regularization == 'l1':
        # Using coordinate descent for L1 (simplified)
        coefficients = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            X_i = X[:, i]
            residuals = y - X @ coefficients + coefficients[i] * X_i
            corr = np.corrcoef(X_i, residuals)[0, 1]
            coefficients[i] = alpha * np.sign(corr) if corr != 0 else 0
    elif regularization == 'l2':
        identity = np.eye(X.shape[1])
        coefficients = np.linalg.inv(X.T @ X + alpha * identity) @ X.T @ y
    elif regularization == 'elasticnet':
        identity = np.eye(X.shape[1])
        coefficients = np.linalg.inv(X.T @ X + alpha * identity) @ X.T @ y
        # Additional L1 step would be needed here for full elasticnet
    else:
        raise ValueError(f"Unknown regularization method: {regularization}")
    return coefficients

def gradient_descent_solver(X: np.ndarray, y: np.ndarray,
                           learning_rate: float = 0.01,
                           n_iter: int = 1000,
                           tol: float = 1e-4) -> np.ndarray:
    """Solve linear regression using gradient descent."""
    coefficients = np.zeros(X.shape[1])
    for _ in range(n_iter):
        gradients = 2 * X.T @ (X @ coefficients - y) / len(y)
        new_coefficients = coefficients - learning_rate * gradients
        if np.linalg.norm(new_coefficients - coefficients) < tol:
            break
        coefficients = new_coefficients
    return coefficients

def moindres_carres_fit(X: np.ndarray, y: np.ndarray,
                       normalization: str = 'none',
                       metrics: Union[str, Callable] = 'mse',
                       solver: str = 'closed_form',
                       regularization: str = 'none',
                       alpha: float = 1.0,
                       **kwargs) -> Dict:
    """
    Perform linear regression using least squares method.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metrics : str or callable
        Metrics to compute ('mse', 'mae', 'r2', 'logloss') or custom function
    solver : str
        Solver method ('closed_form', 'gradient_descent')
    regularization : str
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    alpha : float
        Regularization strength
    **kwargs :
        Additional solver-specific parameters

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = moindres_carres_fit(X, y,
    ...                            normalization='standard',
    ...                            metrics=['mse', 'r2'],
    ...                            solver='closed_form')
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = normalize_data(X, y, normalization)

    # Choose solver
    if solver == 'closed_form':
        coefficients = closed_form_solver(X_norm, y_norm,
                                        regularization=regularization,
                                        alpha=alpha)
    elif solver == 'gradient_descent':
        coefficients = gradient_descent_solver(X_norm, y_norm,
                                             learning_rate=kwargs.get('learning_rate', 0.01),
                                             n_iter=kwargs.get('n_iter', 1000),
                                             tol=kwargs.get('tol', 1e-4))
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Make predictions
    y_pred = X_norm @ coefficients

    # Compute metrics
    if isinstance(metrics, str):
        metrics_list = [metrics]
    else:
        metrics_list = metrics

    all_metrics = {}
    for metric in metrics_list:
        all_metrics.update(compute_metrics(y_norm if normalization != 'none' else y,
                                         y_pred, metric))

    # Prepare results
    result = {
        'result': {
            'coefficients': coefficients,
            'intercept': 0.0,  # For simplicity, assuming centered data
            'predictions': y_pred
        },
        'metrics': all_metrics,
        'params_used': {
            'normalization': normalization,
            'metrics': metrics_list if isinstance(metrics, list) else [metrics],
            'solver': solver,
            'regularization': regularization,
            'alpha': alpha
        },
        'warnings': []
    }

    return result

################################################################################
# regression_simple
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def regression_simple_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    alpha: float = 1.0,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform simple linear regression with configurable options.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, 1)
    y : np.ndarray
        Target values of shape (n_samples,)
    normalizer : Optional[Callable]
        Function to normalize features. Default is None.
    metric : Union[str, Callable]
        Metric to evaluate performance. Can be "mse", "mae", "r2" or custom callable.
    solver : str
        Solver to use. Options: "closed_form", "gradient_descent".
    regularization : Optional[str]
        Regularization type. Options: None, "l1", "l2".
    tol : float
        Tolerance for stopping criteria.
    max_iter : int
        Maximum number of iterations.
    learning_rate : float
        Learning rate for gradient descent.
    alpha : float
        Regularization strength.
    random_state : Optional[int]
        Random seed for reproducibility.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Initialize parameters
    params = {"intercept": 0.0, "slope": 0.0}

    # Choose solver
    if solver == "closed_form":
        params = _solve_closed_form(X_normalized, y)
    elif solver == "gradient_descent":
        params = _solve_gradient_descent(
            X_normalized, y,
            learning_rate=learning_rate,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if specified
    if regularization:
        params = _apply_regularization(params, X_normalized.shape[0], alpha, regularization)

    # Calculate predictions and metrics
    y_pred = _predict(X_normalized, params)
    metrics = _calculate_metrics(y, y_pred, metric)

    # Prepare output
    result = {
        "result": {"intercept": params["intercept"], "slope": params["slope"]},
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric": metric,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "alpha": alpha
        },
        "warnings": []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or X.shape[1] != 1:
        raise ValueError("X must be a 2D array with shape (n_samples, 1)")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to features."""
    if normalizer is None:
        return X
    return normalizer(X)

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Solve linear regression using closed form solution."""
    X_t = np.transpose(X)
    theta = np.linalg.inv(X_t @ X) @ X_t @ y
    return {"intercept": theta[0], "slope": theta[1]}

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None
) -> Dict[str, float]:
    """Solve linear regression using gradient descent."""
    if random_state is not None:
        np.random.seed(random_state)

    theta = np.random.randn(2)
    n_samples = X.shape[0]

    for _ in range(max_iter):
        gradient = (2/n_samples) * X.T @ (X @ theta - y)
        new_theta = theta - learning_rate * gradient

        if np.linalg.norm(new_theta - theta) < tol:
            break

        theta = new_theta

    return {"intercept": theta[0], "slope": theta[1]}

def _apply_regularization(
    params: Dict[str, float],
    n_samples: int,
    alpha: float,
    regularization: str
) -> Dict[str, float]:
    """Apply regularization to parameters."""
    if regularization == "l1":
        params["slope"] = np.sign(params["slope"]) * max(0, abs(params["slope"]) - alpha)
    elif regularization == "l2":
        params["slope"] = params["slope"] * (1 - alpha / n_samples)
    return params

def _predict(X: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Make predictions using the model."""
    return params["intercept"] + X[:, 0] * params["slope"]

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    if callable(metric):
        return {"custom_metric": metric(y_true, y_pred)}

    metrics = {}
    if metric == "mse" or "mse" in metric:
        metrics["mse"] = np.mean((y_true - y_pred) ** 2)
    if metric == "mae" or "mae" in metric:
        metrics["mae"] = np.mean(np.abs(y_true - y_pred))
    if metric == "r2" or "r2" in metric:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics["r2"] = 1 - (ss_res / ss_tot)

    return metrics

# Example usage:
"""
X = np.array([[1], [2], [3]])
y = np.array([2, 4, 6])

result = regression_simple_fit(
    X,
    y,
    normalizer=None,
    metric="mse",
    solver="closed_form"
)
"""

################################################################################
# regression_multiple
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def regression_multiple_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    alpha: float = 1.0,
    l1_ratio: Optional[float] = None,
    custom_metric: Optional[Callable] = None,
    weights: Optional[np.ndarray] = None
) -> Dict:
    """
    Perform multiple linear regression with configurable options.

    Parameters:
    - X: Input features (n_samples, n_features)
    - y: Target values (n_samples,)
    - normalizer: Callable for feature normalization
    - metric: Metric to evaluate performance ('mse', 'mae', 'r2', or custom callable)
    - solver: Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    - regularization: Regularization type ('none', 'l1', 'l2', 'elasticnet')
    - tol: Tolerance for convergence
    - max_iter: Maximum number of iterations
    - learning_rate: Learning rate for gradient-based solvers
    - alpha: Regularization strength
    - l1_ratio: ElasticNet mixing parameter (0 <= l1_ratio <= 1)
    - custom_metric: Custom metric function
    - weights: Sample weights

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Initialize parameters
    n_features = X_normalized.shape[1]
    coef = np.zeros(n_features)
    intercept = 0.0

    # Choose solver
    if solver == 'closed_form':
        coef, intercept = _solve_closed_form(X_normalized, y)
    elif solver == 'gradient_descent':
        coef, intercept = _solve_gradient_descent(
            X_normalized, y, tol, max_iter, learning_rate,
            regularization, alpha, l1_ratio
        )
    elif solver == 'newton':
        coef, intercept = _solve_newton(
            X_normalized, y, tol, max_iter,
            regularization, alpha
        )
    elif solver == 'coordinate_descent':
        coef, intercept = _solve_coordinate_descent(
            X_normalized, y, tol, max_iter,
            regularization, alpha, l1_ratio
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate predictions
    y_pred = _predict(X_normalized, coef, intercept)

    # Calculate metrics
    metrics = _calculate_metrics(y, y_pred, metric, custom_metric)

    # Prepare results
    result = {
        'coef': coef,
        'intercept': intercept,
        'y_pred': y_pred
    }

    params_used = {
        'normalizer': normalizer.__name__ if normalizer else None,
        'metric': metric,
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'max_iter': max_iter,
        'learning_rate': learning_rate,
        'alpha': alpha,
        'l1_ratio': l1_ratio
    }

    warnings = _check_warnings(X, y, coef)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply feature normalization if specified."""
    if normalizer is None:
        return X
    try:
        return normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> tuple:
    """Solve using closed-form solution."""
    X_tx = X.T @ X
    if np.linalg.det(X_tx) == 0:
        raise ValueError("Matrix is singular, cannot compute closed-form solution")
    coef = np.linalg.inv(X_tx) @ X.T @ y
    intercept = 0.0
    return coef, intercept

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int,
    learning_rate: float,
    regularization: Optional[str],
    alpha: float,
    l1_ratio: Optional[float]
) -> tuple:
    """Solve using gradient descent."""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)
    intercept = 0.0

    for _ in range(max_iter):
        # Compute predictions
        y_pred = X @ coef + intercept

        # Compute gradients
        gradient_coef = (2/n_samples) * X.T @ (y_pred - y)
        gradient_intercept = (2/n_samples) * np.sum(y_pred - y)

        # Apply regularization
        if regularization == 'l1':
            gradient_coef += alpha * np.sign(coef)
        elif regularization == 'l2':
            gradient_coef += 2 * alpha * coef
        elif regularization == 'elasticnet' and l1_ratio is not None:
            gradient_coef += alpha * (l1_ratio * np.sign(coef) + (1 - l1_ratio) * 2 * coef)

        # Update parameters
        coef_new = coef - learning_rate * gradient_coef
        intercept_new = intercept - learning_rate * gradient_intercept

        # Check convergence
        if np.linalg.norm(coef_new - coef) < tol and abs(intercept_new - intercept) < tol:
            break

        coef, intercept = coef_new, intercept_new

    return coef, intercept

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int,
    regularization: Optional[str],
    alpha: float
) -> tuple:
    """Solve using Newton's method."""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)
    intercept = 0.0

    for _ in range(max_iter):
        # Compute predictions
        y_pred = X @ coef + intercept

        # Compute Hessian and gradient
        hessian_coef = (2/n_samples) * X.T @ X
        gradient_coef = (2/n_samples) * X.T @ (y_pred - y)
        gradient_intercept = (2/n_samples) * np.sum(y_pred - y)

        # Apply regularization
        if regularization == 'l2':
            hessian_coef += 2 * alpha * np.eye(n_features)

        # Update parameters
        delta_coef = np.linalg.solve(hessian_coef, -gradient_coef)
        delta_intercept = -gradient_intercept / (2/n_samples)

        coef += delta_coef
        intercept += delta_intercept

        # Check convergence
        if np.linalg.norm(delta_coef) < tol and abs(delta_intercept) < tol:
            break

    return coef, intercept

def _solve_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int,
    regularization: Optional[str],
    alpha: float,
    l1_ratio: Optional[float]
) -> tuple:
    """Solve using coordinate descent."""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)
    intercept = 0.0

    for _ in range(max_iter):
        # Compute predictions
        y_pred = X @ coef + intercept

        # Update each coefficient one at a time
        for j in range(n_features):
            X_j = X[:, j]
            residual = y - (y_pred - coef[j] * X_j)

            # Compute optimal coefficient
            numerator = X_j @ residual
            denominator = np.sum(X_j**2)

            if regularization == 'l1':
                coef[j] = np.sign(numerator) * np.maximum(
                    abs(numerator) - alpha, 0
                ) / denominator
            elif regularization == 'l2':
                coef[j] = numerator / (denominator + 2 * alpha)
            elif regularization == 'elasticnet' and l1_ratio is not None:
                coef[j] = np.sign(numerator) * np.maximum(
                    abs(numerator) - alpha * l1_ratio, 0
                ) / (denominator + 2 * alpha * (1 - l1_ratio))
            else:
                coef[j] = numerator / denominator

        # Update intercept
        intercept = np.mean(y - X @ coef)

        # Check convergence
        if np.linalg.norm(X @ (coef - np.zeros_like(coef))) < tol:
            break

    return coef, intercept

def _predict(X: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    """Make predictions using the model."""
    return X @ coef + intercept

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Calculate evaluation metrics."""
    metrics = {}

    if metric == 'mse' or (custom_metric is None and metric == 'mse'):
        metrics['mse'] = np.mean((y_true - y_pred)**2)
    if metric == 'mae' or (custom_metric is None and metric == 'mae'):
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    if metric == 'r2' or (custom_metric is None and metric == 'r2'):
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        metrics['r2'] = 1 - (ss_res / ss_tot)

    if custom_metric is not None:
        try:
            metrics['custom'] = custom_metric(y_true, y_pred)
        except Exception as e:
            raise ValueError(f"Custom metric calculation failed: {str(e)}")

    return metrics

def _check_warnings(X: np.ndarray, y: np.ndarray, coef: np.ndarray) -> list:
    """Check for potential issues and generate warnings."""
    warnings = []

    if np.any(np.isnan(coef)):
        warnings.append("Coefficients contain NaN values")
    if np.any(np.isinf(coef)):
        warnings.append("Coefficients contain infinite values")

    if np.linalg.norm(X @ coef) > 1e6:
        warnings.append("Large predicted values detected")

    return warnings

# Example usage
"""
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([1, 2, 3])

result = regression_multiple_fit(
    X=X,
    y=y,
    normalizer=None,
    metric='mse',
    solver='closed_form'
)

print(result)
"""

################################################################################
# variable_explicative
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def variable_explicative_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Fit a linear regression model with configurable options.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalizer : Optional[Callable]
        Function to normalize features. Default is None.
    metric : Union[str, Callable]
        Metric to evaluate model performance. Can be 'mse', 'mae', 'r2' or custom callable.
    solver : str
        Solver to use. Options: 'closed_form', 'gradient_descent', 'newton'.
    regularization : Optional[str]
        Regularization type. Options: None, 'l1', 'l2', 'elasticnet'.
    tol : float
        Tolerance for stopping criteria.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable]
        Custom metric function if needed.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Choose solver
    if solver == 'closed_form':
        coefficients = _solve_closed_form(X_normalized, y)
    elif solver == 'gradient_descent':
        coefficients = _solve_gradient_descent(X_normalized, y, tol, max_iter, **kwargs)
    elif solver == 'newton':
        coefficients = _solve_newton(X_normalized, y, tol, max_iter, **kwargs)
    else:
        raise ValueError("Unsupported solver specified")

    # Apply regularization if needed
    if regularization:
        coefficients = _apply_regularization(coefficients, X_normalized, y, regularization)

    # Calculate metrics
    metrics = _calculate_metrics(X_normalized, y, coefficients, metric, custom_metric)

    return {
        'result': {'coefficients': coefficients},
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays contain infinite values")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to features."""
    if normalizer is None:
        return X
    try:
        return normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve linear regression using closed form solution."""
    XTX = X.T @ X
    if np.linalg.det(XTX) == 0:
        raise ValueError("Matrix is singular, cannot compute closed form solution")
    return np.linalg.inv(XTX) @ X.T @ y

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int,
    learning_rate: float = 0.01,
    **kwargs
) -> np.ndarray:
    """Solve linear regression using gradient descent."""
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ coefficients - y) / len(y)
        new_coefficients = coefficients - learning_rate * gradient
        if np.linalg.norm(new_coefficients - coefficients) < tol:
            break
        coefficients = new_coefficients
    return coefficients

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int,
    **kwargs
) -> np.ndarray:
    """Solve linear regression using Newton's method."""
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ coefficients - y) / len(y)
        hessian = 2 * X.T @ X / len(y)
        new_coefficients = coefficients - np.linalg.inv(hessian) @ gradient
        if np.linalg.norm(new_coefficients - coefficients) < tol:
            break
        coefficients = new_coefficients
    return coefficients

def _apply_regularization(
    coefficients: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    regularization: str
) -> np.ndarray:
    """Apply regularization to coefficients."""
    if regularization == 'l1':
        return _apply_l1_regularization(coefficients, X, y)
    elif regularization == 'l2':
        return _apply_l2_regularization(coefficients, X, y)
    elif regularization == 'elasticnet':
        return _apply_elasticnet_regularization(coefficients, X, y)
    else:
        raise ValueError("Unsupported regularization type")

def _apply_l1_regularization(
    coefficients: np.ndarray,
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """Apply L1 regularization."""
    # Simplified implementation - in practice would use coordinate descent
    return coefficients

def _apply_l2_regularization(
    coefficients: np.ndarray,
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """Apply L2 regularization."""
    # Simplified implementation - in practice would use ridge regression
    return coefficients

def _apply_elasticnet_regularization(
    coefficients: np.ndarray,
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """Apply elastic net regularization."""
    # Simplified implementation - in practice would combine L1 and L2
    return coefficients

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Calculate evaluation metrics."""
    predictions = X @ coefficients
    metrics_dict = {}

    if metric == 'mse':
        metrics_dict['mse'] = np.mean((y - predictions) ** 2)
    elif metric == 'mae':
        metrics_dict['mae'] = np.mean(np.abs(y - predictions))
    elif metric == 'r2':
        metrics_dict['r2'] = 1 - np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)
    elif callable(metric):
        metrics_dict['custom'] = metric(y, predictions)

    if custom_metric is not None:
        metrics_dict['custom'] = custom_metric(y, predictions)

    return metrics_dict

################################################################################
# variable_cible
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def variable_cible_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Fit a linear regression model with customizable options.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalizer : Callable, optional
        Function to normalize features. Default is None.
    metric : str or Callable, optional
        Metric to evaluate model performance. Default is 'mse'.
    solver : str, optional
        Solver to use for optimization. Default is 'closed_form'.
    regularization : str, optional
        Type of regularization to apply. Default is None.
    tol : float, optional
        Tolerance for convergence. Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    custom_metric : Callable, optional
        Custom metric function. Default is None.

    Returns:
    --------
    Dict containing:
        - result: fitted model parameters
        - metrics: computed metrics
        - params_used: parameters used in fitting
        - warnings: any warnings encountered

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = variable_cible_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Choose solver
    if solver == 'closed_form':
        params = _solve_closed_form(X_normalized, y)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(X_normalized, y, tol, max_iter, **kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if specified
    params = _apply_regularization(params, X_normalized.shape[1], regularization)

    # Compute metrics
    metrics = _compute_metrics(X_normalized, y, params, metric, custom_metric)

    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to features."""
    if normalizer is None:
        return X
    try:
        return normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve linear regression using closed form solution."""
    XTX = X.T @ X
    if np.linalg.det(XTX) == 0:
        raise ValueError("Matrix is singular, cannot compute closed form solution")
    return np.linalg.inv(XTX) @ X.T @ y

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int,
    **kwargs
) -> np.ndarray:
    """Solve linear regression using gradient descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    learning_rate = kwargs.get('learning_rate', 0.01)

    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ params - y) / len(y)
        new_params = params - learning_rate * gradient
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params

    return params

def _apply_regularization(
    params: np.ndarray,
    n_features: int,
    regularization: Optional[str]
) -> np.ndarray:
    """Apply regularization to parameters."""
    if regularization is None:
        return params
    alpha = 1.0  # Default regularization strength

    if regularization == 'l1':
        params = np.sign(params) * np.maximum(np.abs(params) - alpha, 0)
    elif regularization == 'l2':
        params = params / (1 + alpha * n_features)
    elif regularization == 'elasticnet':
        l1_ratio = 0.5
        params_l1 = np.sign(params) * np.maximum(np.abs(params) - alpha * l1_ratio, 0)
        params_l2 = params / (1 + alpha * (1 - l1_ratio) * n_features)
        params = np.where(np.abs(params) > alpha * l1_ratio, params_l1, params_l2)
    else:
        raise ValueError(f"Unknown regularization: {regularization}")

    return params

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Compute metrics for the model."""
    y_pred = X @ params
    metrics = {}

    if metric == 'mse':
        metrics['mse'] = np.mean((y - y_pred) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics['r2'] = 1 - ss_res / ss_tot
    elif callable(metric):
        metrics['custom'] = metric(y, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if custom_metric is not None:
        metrics['custom_metric'] = custom_metric(y, y_pred)

    return metrics

################################################################################
# hypotheses_linearite
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def hypotheses_linearite_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, str]]:
    """
    Test the linearity hypotheses for a linear regression model.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the data. Default is None.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Metric to evaluate the model. Can be 'mse', 'mae', 'r2', or a custom callable.
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Distance metric for residuals. Can be 'euclidean', 'manhattan', 'cosine', or a custom callable.
    solver : str, optional
        Solver to use for fitting the model. Can be 'closed_form', 'gradient_descent', etc.
    regularization : Optional[str], optional
        Regularization type. Can be 'l1', 'l2', or None.
    tol : float, optional
        Tolerance for stopping criteria. Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function. Default is None.

    Returns
    -------
    Dict[str, Union[Dict, float, str]]
        Dictionary containing the results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = hypotheses_linearite_fit(X, y, normalizer=np.std, metric='r2')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if a normalizer is provided
    X_norm = _apply_normalization(X, normalizer)
    y_norm = _apply_normalization(y.reshape(-1, 1), normalizer).flatten()

    # Choose solver
    if solver == 'closed_form':
        beta = _closed_form_solver(X_norm, y_norm)
    else:
        raise ValueError(f"Solver {solver} not implemented.")

    # Calculate residuals
    residuals = y_norm - X_norm @ beta

    # Choose metric
    if isinstance(metric, str):
        if metric == 'mse':
            m = _mean_squared_error
        elif metric == 'mae':
            m = _mean_absolute_error
        elif metric == 'r2':
            m = _r_squared
        else:
            raise ValueError(f"Metric {metric} not recognized.")
    elif callable(metric):
        m = metric
    else:
        raise ValueError("Metric must be a string or callable.")

    # Calculate metric
    metric_value = m(y_norm, X_norm @ beta)

    # Choose distance
    if isinstance(distance, str):
        if distance == 'euclidean':
            d = _euclidean_distance
        elif distance == 'manhattan':
            d = _manhattan_distance
        elif distance == 'cosine':
            d = _cosine_distance
        else:
            raise ValueError(f"Distance {distance} not recognized.")
    elif callable(distance):
        d = distance
    else:
        raise ValueError("Distance must be a string or callable.")

    # Calculate distance
    distance_value = d(residuals, np.zeros_like(residuals))

    # Prepare results
    result = {
        'result': {
            'beta': beta,
            'residuals': residuals,
            'distance': distance_value
        },
        'metrics': {
            metric: metric_value
        },
        'params_used': {
            'normalizer': normalizer.__name__ if callable(normalizer) else str(normalizer),
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate the input data."""
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

def _apply_normalization(data: np.ndarray, normalizer: Optional[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    """Apply normalization to the data."""
    if normalizer is None:
        return data
    if callable(normalizer):
        return normalizer(data)
    raise ValueError("Normalizer must be a callable or None.")

def _closed_form_solver(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve the linear regression problem using the closed-form solution."""
    return np.linalg.inv(X.T @ X) @ X.T @ y

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the R-squared metric."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the cosine distance."""
    return 1 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

################################################################################
# hypotheses_independance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def hypotheses_independance_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    distance_metric: str = 'euclidean',
    test_statistic: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.corrcoef(x, y)[0, 1],
    significance_level: float = 0.05,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Test d'hypothèses d'indépendance entre les variables explicatives et la variable cible.

    Parameters
    ----------
    X : np.ndarray
        Matrice des variables explicatives de forme (n_samples, n_features).
    y : np.ndarray
        Vecteur cible de forme (n_samples,).
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Fonction de normalisation à appliquer aux données, par défaut None.
    distance_metric : str, optional
        Métrique de distance à utiliser pour les tests d'indépendance, par défaut 'euclidean'.
    test_statistic : Callable[[np.ndarray, np.ndarray], float], optional
        Statistique de test à utiliser pour évaluer l'indépendance, par défaut np.corrcoef.
    significance_level : float, optional
        Niveau de signification pour le test d'hypothèses, par défaut 0.05.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Métrique personnalisée pour évaluer l'indépendance, par défaut None.

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], str]]
        Dictionnaire contenant les résultats du test d'indépendance.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = hypotheses_independance_fit(X, y)
    """
    # Validation des entrées
    _validate_inputs(X, y)

    # Normalisation des données si spécifiée
    if normalizer is not None:
        X = normalizer(X)
        y = normalizer(y.reshape(-1, 1)).flatten()

    # Calcul des statistiques de test pour chaque variable
    test_results = []
    for i in range(X.shape[1]):
        stat_value = test_statistic(X[:, i], y)
        p_value = _compute_p_value(stat_value, X.shape[0])
        test_results.append({
            'feature': i,
            'statistic': stat_value,
            'p_value': p_value
        })

    # Calcul des métriques personnalisées si spécifiées
    metrics = {}
    if custom_metric is not None:
        for i in range(X.shape[1]):
            metrics[f'feature_{i}'] = custom_metric(X[:, i], y)

    # Détermination des variables indépendantes
    independent_features = [i for i, res in enumerate(test_results) if res['p_value'] > significance_level]

    return {
        'result': {
            'independent_features': independent_features,
            'test_results': test_results
        },
        'metrics': metrics if custom_metric is not None else {},
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer is not None else None,
            'distance_metric': distance_metric,
            'significance_level': significance_level
        },
        'warnings': _check_warnings(X, y)
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validation des entrées pour le test d'indépendance."""
    if X.shape[0] != y.shape[0]:
        raise ValueError("X et y doivent avoir le même nombre d'échantillons.")
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("Les données contiennent des valeurs NaN.")
    if np.isinf(X).any() or np.isinf(y).any():
        raise ValueError("Les données contiennent des valeurs infinies.")

def _compute_p_value(statistic: float, n_samples: int) -> float:
    """Calcul de la p-value pour une statistique de test donnée."""
    # Simplification: utilisation d'une approximation normale
    return 2 * (1 - np.sqrt(n_samples) * abs(statistic))

def _check_warnings(X: np.ndarray, y: np.ndarray) -> list:
    """Vérification des avertissements potentiels."""
    warnings = []
    if X.shape[1] > 20:
        warnings.append("Un grand nombre de variables peut affecter la performance.")
    if np.std(y) < 1e-6:
        warnings.append("La variance de y est très faible, les résultats peuvent être instables.")
    return warnings

def _standard_normalizer(data: np.ndarray) -> np.ndarray:
    """Normalisation standard des données."""
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def _minmax_normalizer(data: np.ndarray) -> np.ndarray:
    """Normalisation Min-Max des données."""
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

def _robust_normalizer(data: np.ndarray) -> np.ndarray:
    """Normalisation robuste des données."""
    median = np.median(data, axis=0)
    iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
    return (data - median) / iqr

################################################################################
# hypotheses_homoscedasticite
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def hypotheses_homoscedasticite_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residuals: Optional[np.ndarray] = None,
    test_type: str = 'breusch_pagan',
    normalize_residuals: bool = True,
    custom_test_func: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Union[float, Dict, str]]:
    """
    Test d'homoscedasticité des résidus d'une régression linéaire.

    Parameters
    ----------
    y_true : np.ndarray
        Valeurs observées.
    y_pred : np.ndarray
        Valeurs prédites par le modèle.
    residuals : Optional[np.ndarray], default=None
        Résidus calculés. Si None, ils sont calculés comme y_true - y_pred.
    test_type : str, default='breusch_pagan'
        Type de test à effectuer. Options: 'breusch_pagan', 'levene', 'bartlett'.
    normalize_residuals : bool, default=True
        Si True, les résidus sont normalisés avant le test.
    custom_test_func : Optional[Callable], default=None
        Fonction personnalisée pour le test d'homoscedasticité.
    **kwargs : dict
        Arguments supplémentaires pour la fonction de test personnalisée.

    Returns
    -------
    Dict[str, Union[float, Dict, str]]
        Dictionnaire contenant les résultats du test:
            - 'statistic': valeur de la statistique de test
            - 'p_value': p-value associée
            - 'test_used': type de test utilisé
            - 'warnings': avertissements éventuels

    Examples
    --------
    >>> y_true = np.array([1, 2, 3, 4, 5])
    >>> y_pred = np.array([1.1, 2.0, 3.2, 4.1, 5.0])
    >>> result = hypotheses_homoscedasticite_fit(y_true, y_pred)
    """
    # Validation des entrées
    _validate_inputs(y_true, y_pred, residuals)

    # Calcul des résidus si nécessaire
    if residuals is None:
        residuals = y_true - y_pred

    # Normalisation des résidus si demandé
    if normalize_residuals:
        residuals = _normalize_residuals(residuals)

    # Choix du test
    if custom_test_func is not None:
        statistic, p_value = _custom_homoscedasticity_test(residuals, custom_test_func, **kwargs)
    else:
        if test_type == 'breusch_pagan':
            statistic, p_value = _breusch_pagan_test(residuals)
        elif test_type == 'levene':
            statistic, p_value = _levene_test(residuals)
        elif test_type == 'bartlett':
            statistic, p_value = _bartlett_test(residuals)
        else:
            raise ValueError(f"Unknown test type: {test_type}")

    # Construction du résultat
    result = {
        'statistic': statistic,
        'p_value': p_value,
        'test_used': test_type if custom_test_func is None else 'custom',
        'warnings': []
    }

    return result

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray, residuals: Optional[np.ndarray]) -> None:
    """Validation des entrées."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("y_true and y_pred must be numpy arrays")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if residuals is not None:
        if len(residuals) != len(y_true):
            raise ValueError("residuals must have the same length as y_true and y_pred")
        if not isinstance(residuals, np.ndarray):
            raise TypeError("residuals must be a numpy array")

def _normalize_residuals(residuals: np.ndarray) -> np.ndarray:
    """Normalisation des résidus."""
    return (residuals - np.mean(residuals)) / np.std(residuals)

def _breusch_pagan_test(residuals: np.ndarray) -> tuple:
    """Test de Breusch-Pagan pour l'homoscedasticité."""
    n = len(residuals)
    residuals_squared = residuals ** 2
    X = np.column_stack([np.ones(n), residuals])
    beta = np.linalg.inv(X.T @ X) @ X.T @ residuals_squared
    statistic = n * beta[1] ** 2 / (1 - beta[1]) ** 2
    p_value = _chi2_pvalue(statistic, df=1)
    return statistic, p_value

def _levene_test(residuals: np.ndarray) -> tuple:
    """Test de Levene pour l'homoscedasticité."""
    # Implémentation simplifiée du test de Levene
    centered_residuals = np.abs(residuals - np.median(residuals))
    statistic, p_value = _f_test(centered_residuals)
    return statistic, p_value

def _bartlett_test(residuals: np.ndarray) -> tuple:
    """Test de Bartlett pour l'homoscedasticité."""
    # Implémentation simplifiée du test de Bartlett
    groups = np.array_split(residuals, 2)
    variances = [np.var(group) for group in groups]
    statistic = _bartlett_statistic(variances, len(groups[0]))
    p_value = _chi2_pvalue(statistic, df=len(groups) - 1)
    return statistic, p_value

def _custom_homoscedasticity_test(
    residuals: np.ndarray,
    test_func: Callable,
    **kwargs
) -> tuple:
    """Test d'homoscedasticité personnalisé."""
    return test_func(residuals, **kwargs)

def _chi2_pvalue(statistic: float, df: int) -> float:
    """Calcul de la p-value pour une statistique du chi-carré."""
    from scipy.stats import chi2
    return 1 - chi2.cdf(statistic, df)

def _f_test(values: np.ndarray) -> tuple:
    """Test F simplifié."""
    from scipy.stats import f
    n = len(values)
    group1, group2 = np.array_split(values, 2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    statistic = var1 / var2
    p_value = 1 - f.cdf(statistic, len(group1)-1, len(group2)-1)
    return statistic, p_value

def _bartlett_statistic(variances: list, n: int) -> float:
    """Calcul de la statistique de Bartlett."""
    k = len(variances)
    C = (1 / (3 * (k - 1))) * (1 / n - 1 / np.sum([1/ni for ni in [n]*k]))
    statistic = (2 * n - k) * np.log(np.mean(variances)) - np.sum([n * np.log(var) for var, ni in zip(variances, [n]*k)])
    return statistic / C

################################################################################
# hypotheses_normalite
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def hypotheses_normalite_fit(
    residuals: np.ndarray,
    normalizations: str = 'standard',
    metrics: Union[str, Callable[[np.ndarray], float]] = 'all',
    distances: str = 'euclidean',
    custom_metric: Optional[Callable[[np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, str]]:
    """
    Fit normality hypotheses tests for regression residuals.

    Parameters
    ----------
    residuals : np.ndarray
        Array of residuals from linear regression.
    normalizations : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust'), default 'standard'.
    metrics : str or callable, optional
        Metrics to compute ('all', 'shapiro', 'kolmogorov', 'anderson'), default 'all'.
    distances : str, optional
        Distance metric ('euclidean', 'manhattan', 'cosine'), default 'euclidean'.
    custom_metric : callable, optional
        Custom metric function.
    custom_distance : callable, optional
        Custom distance function.

    Returns
    -------
    dict
        Dictionary containing results, metrics, parameters used and warnings.

    Examples
    --------
    >>> residuals = np.random.randn(100)
    >>> result = hypotheses_normalite_fit(residuals)
    """
    # Validate inputs
    _validate_inputs(residuals, normalizations, metrics)

    # Normalize residuals
    normalized_residuals = _apply_normalization(residuals, normalizations)

    # Compute metrics
    test_results = _compute_metrics(normalized_residuals, metrics, custom_metric)

    # Compute distances
    distance_results = _compute_distances(normalized_residuals, distances, custom_distance)

    # Prepare output
    return {
        'result': test_results,
        'metrics': distance_results,
        'params_used': {
            'normalization': normalizations,
            'metrics': metrics,
            'distance': distances
        },
        'warnings': _check_warnings(residuals)
    }

def _validate_inputs(
    residuals: np.ndarray,
    normalizations: str,
    metrics: Union[str, Callable[[np.ndarray], float]]
) -> None:
    """Validate input parameters."""
    if not isinstance(residuals, np.ndarray):
        raise TypeError("Residuals must be a numpy array")
    if residuals.ndim != 1:
        raise ValueError("Residuals must be a 1D array")
    if np.isnan(residuals).any():
        raise ValueError("Residuals contain NaN values")
    if np.isinf(residuals).any():
        raise ValueError("Residuals contain infinite values")

    valid_normalizations = ['none', 'standard', 'minmax', 'robust']
    if normalizations not in valid_normalizations:
        raise ValueError(f"Normalization must be one of {valid_normalizations}")

    if isinstance(metrics, str):
        valid_metrics = ['all', 'shapiro', 'kolmogorov', 'anderson']
        if metrics not in valid_metrics:
            raise ValueError(f"Metrics must be one of {valid_metrics}")

def _apply_normalization(
    residuals: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply normalization to residuals."""
    if method == 'none':
        return residuals
    elif method == 'standard':
        return (residuals - np.mean(residuals)) / np.std(residuals)
    elif method == 'minmax':
        return (residuals - np.min(residuals)) / (np.max(residuals) - np.min(residuals))
    elif method == 'robust':
        median = np.median(residuals)
        iqr = np.percentile(residuals, 75) - np.percentile(residuals, 25)
        return (residuals - median) / iqr
    else:
        raise ValueError("Invalid normalization method")

def _compute_metrics(
    residuals: np.ndarray,
    metrics: Union[str, Callable[[np.ndarray], float]],
    custom_metric: Optional[Callable[[np.ndarray], float]]
) -> Dict[str, Union[float, str]]:
    """Compute normality test metrics."""
    results = {}

    if isinstance(metrics, str):
        if metrics == 'all' or metrics == 'shapiro':
            from scipy.stats import shapiro
            stat, p = shapiro(residuals)
            results['shapiro'] = {'statistic': stat, 'pvalue': p}

        if metrics == 'all' or metrics == 'kolmogorov':
            from scipy.stats import kstest
            stat, p = kstest(residuals, 'norm')
            results['kolmogorov'] = {'statistic': stat, 'pvalue': p}

        if metrics == 'all' or metrics == 'anderson':
            from scipy.stats import anderson
            stat = anderson(residuals)
            results['anderson'] = {'statistic': stat.statistic, 'critical_values': stat.critical_values}

    if custom_metric is not None:
        results['custom'] = custom_metric(residuals)

    return results

def _compute_distances(
    residuals: np.ndarray,
    distance_type: str,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, Union[float, str]]:
    """Compute distance metrics for residuals."""
    results = {}

    if custom_distance is not None:
        results['custom'] = custom_distance(residuals, residuals)
    else:
        if distance_type == 'euclidean':
            results['euclidean'] = np.linalg.norm(residuals)
        elif distance_type == 'manhattan':
            results['manhattan'] = np.sum(np.abs(residuals))
        elif distance_type == 'cosine':
            results['cosine'] = 1 - np.dot(residuals, residuals) / (np.linalg.norm(residuals) * np.linalg.norm(residuals))
        else:
            raise ValueError("Invalid distance type")

    return results

def _check_warnings(
    residuals: np.ndarray
) -> list:
    """Check for potential warnings."""
    warnings = []

    if len(residuals) < 20:
        warnings.append("Small sample size may affect test reliability")

    if np.var(residuals) == 0:
        warnings.append("Zero variance in residuals")

    return warnings

################################################################################
# multicolinearite
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def multicolinearite_fit(
    X: np.ndarray,
    normalize: str = 'standard',
    metric: Union[str, Callable] = 'variance_inflation_factor',
    threshold: float = 5.0,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Compute multicollinearity diagnostics for a feature matrix.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, optional
        Multicollinearity metric ('variance_inflation_factor', 'condition_index')
    threshold : float, optional
        Threshold for multicollinearity detection
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': multicollinearity results
        - 'metrics': computed metrics
        - 'params_used': parameters used
        - 'warnings': any warnings

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> result = multicolinearite_fit(X)
    """
    # Validate inputs
    _validate_inputs(X, normalize)

    # Normalize data if needed
    X_normalized = _apply_normalization(X, normalize)

    # Compute multicollinearity metrics
    if metric == 'variance_inflation_factor':
        vifs = _compute_variance_inflation_factors(X_normalized)
        metrics = {'variance_inflation_factor': vifs}
    elif metric == 'condition_index':
        ci = _compute_condition_index(X_normalized)
        metrics = {'condition_index': ci}
    elif callable(metric):
        custom_result = metric(X_normalized)
        metrics = {'custom_metric': custom_result}
    else:
        raise ValueError("Invalid metric specified")

    # Determine multicollinearity results
    if metric == 'variance_inflation_factor':
        result = _detect_multicollinearity_vif(vifs, threshold)
    elif metric == 'condition_index':
        result = _detect_multicollinearity_ci(ci, threshold)
    elif callable(metric):
        result = {'multicollinear_features': []}

    # Prepare output
    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalize': normalize,
            'metric': metric,
            'threshold': threshold
        },
        'warnings': _check_warnings(X, normalize)
    }

def _validate_inputs(
    X: np.ndarray,
    normalize: str
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if np.isnan(X).any():
        raise ValueError("Input contains NaN values")
    if normalize not in ['none', 'standard', 'minmax', 'robust']:
        raise ValueError("Invalid normalization method")

def _apply_normalization(
    X: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply specified normalization to the feature matrix."""
    if method == 'none':
        return X
    elif method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError("Invalid normalization method")

def _compute_variance_inflation_factors(
    X: np.ndarray
) -> np.ndarray:
    """Compute Variance Inflation Factors for each feature."""
    n_features = X.shape[1]
    vifs = np.zeros(n_features)

    for i in range(n_features):
        X_reduced = np.delete(X, i, axis=1)
        if X_reduced.shape[1] == 0:
            vifs[i] = np.nan
            continue

        # Add constant for regression
        X_reduced_with_const = np.column_stack([np.ones(X.shape[0]), X_reduced])
        y = X[:, i]

        # Compute R-squared
        beta = np.linalg.inv(X_reduced_with_const.T @ X_reduced_with_const) @ X_reduced_with_const.T @ y
        residuals = y - X_reduced_with_const @ beta
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)

        if ss_tot == 0:
            vifs[i] = np.nan
        else:
            r_squared = 1 - (ss_res / ss_tot)
            vifs[i] = 1 / (1 - r_squared)

    return vifs

def _compute_condition_index(
    X: np.ndarray
) -> float:
    """Compute condition index of the feature matrix."""
    _, s, _ = np.linalg.svd(X)
    condition_index = max(s) / min(s)
    return condition_index

def _detect_multicollinearity_vif(
    vifs: np.ndarray,
    threshold: float
) -> Dict:
    """Detect multicollinearity using VIF."""
    multicollinear_features = np.where(vifs > threshold)[0]
    return {
        'multicollinear_features': multicollinear_features.tolist(),
        'vif_threshold_exceeded': threshold
    }

def _detect_multicollinearity_ci(
    ci: float,
    threshold: float
) -> Dict:
    """Detect multicollinearity using condition index."""
    return {
        'condition_index': ci,
        'high_conditioning': ci > threshold
    }

def _check_warnings(
    X: np.ndarray,
    normalize: str
) -> list:
    """Check for potential warnings."""
    warnings = []

    if X.shape[1] < 2:
        warnings.append("Warning: Fewer than 2 features - multicollinearity detection may not be meaningful")

    if normalize == 'standard' and np.std(X, axis=0).min() == 0:
        warnings.append("Warning: Zero standard deviation detected - standard normalization may be problematic")

    return warnings

################################################################################
# r_carre
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def r_carre_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalize: str = "standard",
    metric: Union[str, Callable] = "r2",
    solver: str = "closed_form",
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Compute the R-squared (coefficient of determination) for linear regression.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    normalize : str, optional
        Normalization method: "none", "standard", "minmax", or "robust".
    metric : str or callable, optional
        Metric to compute: "mse", "mae", "r2", or custom callable.
    solver : str, optional
        Solver method: "closed_form", "gradient_descent", etc.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function.

    Returns:
    --------
    Dict containing:
        - "result": R-squared value.
        - "metrics": Dictionary of computed metrics.
        - "params_used": Parameters used in the computation.
        - "warnings": List of warnings encountered.

    Example:
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> y = np.array([5, 6])
    >>> result = r_carre_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_normalized = _normalize_data(X, method=normalize)

    # Choose solver and compute coefficients
    if solver == "closed_form":
        beta = _closed_form_solver(X_normalized, y)
    else:
        raise ValueError(f"Solver {solver} not implemented.")

    # Compute predictions
    y_pred = _predict(X_normalized, beta)

    # Compute metrics
    metrics = {}
    if metric == "r2" or (custom_metric is None and metric == "r2"):
        metrics["r2"] = _compute_r_squared(y, y_pred)
    elif callable(metric):
        metrics["custom"] = metric(y, y_pred)
    else:
        raise ValueError(f"Metric {metric} not supported.")

    if custom_metric is not None:
        metrics["custom"] = custom_metric(y, y_pred)

    # Prepare output
    result_dict = {
        "result": metrics.get("r2", None),
        "metrics": metrics,
        "params_used": {
            "normalize": normalize,
            "metric": metric,
            "solver": solver,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    return result_dict

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

def _normalize_data(X: np.ndarray, method: str = "standard") -> np.ndarray:
    """Normalize the feature matrix."""
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
        raise ValueError(f"Normalization method {method} not supported.")

def _closed_form_solver(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute coefficients using closed-form solution."""
    X_tx = np.dot(X.T, X)
    if not np.allclose(np.linalg.det(X_tx), 0):
        beta = np.linalg.solve(X_tx, np.dot(X.T, y))
    else:
        beta = np.linalg.pinv(X_tx).dot(X.T).dot(y)
    return beta

def _predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Compute predictions."""
    return X.dot(beta)

def _compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))

################################################################################
# r_carre_ajuste
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def r_carre_ajuste_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalization: str = 'standard',
    metric: Union[str, Callable] = 'r2',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Calculate the adjusted R-squared for linear regression.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, optional
        Metric to use ('mse', 'mae', 'r2') or custom callable
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent')
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2')
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
    custom_metric : callable, optional
        Custom metric function

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = r_carre_ajuste_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if needed
    X_norm, y_norm = _apply_normalization(X, y, normalization)

    # Choose solver
    if solver == 'closed_form':
        beta = _solve_closed_form(X_norm, y_norm)
    elif solver == 'gradient_descent':
        beta = _solve_gradient_descent(X_norm, y_norm, tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if needed
    if regularization:
        beta = _apply_regularization(beta, X_norm.shape[1], regularization)

    # Calculate predictions
    y_pred = _predict(X_norm, beta)

    # Calculate metrics
    metrics = _calculate_metrics(y_norm, y_pred, metric=metric, custom_metric=custom_metric)

    # Calculate adjusted R-squared
    n = X.shape[0]
    p = X.shape[1]
    r2 = metrics.get('r2', 0)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    return {
        'result': adjusted_r2,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays contain infinite values")

def _apply_normalization(
    X: np.ndarray,
    y: np.ndarray,
    method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply normalization to features and target."""
    if method == 'standard':
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y = (y - np.mean(y)) / np.std(y)
    elif method == 'minmax':
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif method == 'robust':
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        y = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    return X, y

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve linear regression using closed form solution."""
    return np.linalg.inv(X.T @ X) @ X.T @ y

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Solve linear regression using gradient descent."""
    n_features = X.shape[1]
    beta = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradient = -2 * X.T @ (y - X @ beta) / len(y)
        new_beta = beta - learning_rate * gradient

        if np.linalg.norm(new_beta - beta) < tol:
            break
        beta = new_beta

    return beta

def _apply_regularization(
    beta: np.ndarray,
    n_features: int,
    method: str
) -> np.ndarray:
    """Apply regularization to coefficients."""
    if method == 'l1':
        return np.sign(beta) * np.maximum(np.abs(beta) - 0.1, 0)
    elif method == 'l2':
        return beta / (1 + 0.1 * np.sum(beta**2))
    elif method == 'elasticnet':
        beta_l1 = np.sign(beta) * np.maximum(np.abs(beta) - 0.05, 0)
        beta_l2 = beta / (1 + 0.05 * np.sum(beta**2))
        return (beta_l1 + beta_l2) / 2
    return beta

def _predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Make predictions using linear model."""
    return X @ beta

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: Union[str, Callable] = 'r2',
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Calculate regression metrics."""
    metrics = {}

    if metric == 'mse' or custom_metric is None:
        mse = np.mean((y_true - y_pred) ** 2)
        metrics['mse'] = mse

    if metric == 'mae' or custom_metric is None:
        mae = np.mean(np.abs(y_true - y_pred))
        metrics['mae'] = mae

    if metric == 'r2' or custom_metric is None:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot
        metrics['r2'] = r2

    if custom_metric is not None:
        try:
            metrics['custom'] = custom_metric(y_true, y_pred)
        except Exception as e:
            metrics['custom_error'] = str(e)

    return metrics

################################################################################
# ecart_type_residuel
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for linear regression."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def normalize_data(X: np.ndarray, y: np.ndarray,
                  normalization: str = 'standard') -> tuple[np.ndarray, np.ndarray]:
    """Normalize data according to specified method."""
    if normalization == 'none':
        return X, y
    elif normalization == 'standard':
        mean_X = np.mean(X, axis=0)
        std_X = np.std(X, axis=0)
        X_normalized = (X - mean_X) / std_X
        y_normalized = (y - np.mean(y)) / np.std(y)
    elif normalization == 'minmax':
        min_X = np.min(X, axis=0)
        max_X = np.max(X, axis=0)
        X_normalized = (X - min_X) / (max_X - min_X + 1e-8)
        y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-8)
    elif normalization == 'robust':
        median_X = np.median(X, axis=0)
        iqr_X = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_normalized = (X - median_X) / (iqr_X + 1e-8)
        y_normalized = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25) + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")
    return X_normalized, y_normalized

def compute_residuals(X: np.ndarray, y: np.ndarray,
                     coefficients: np.ndarray) -> np.ndarray:
    """Compute residuals from linear regression model."""
    return y - X @ coefficients

def ecart_type_residuel_fit(X: np.ndarray, y: np.ndarray,
                          normalization: str = 'standard',
                          metric: Optional[Callable[[np.ndarray], float]] = None,
                          solver: str = 'closed_form') -> Dict[str, Any]:
    """
    Compute the standard deviation of residuals for linear regression.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : Callable, optional
        Custom metric function that takes residuals and returns a float
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', etc.)

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = ecart_type_residuel_fit(X, y)
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = normalize_data(X, y, normalization)

    # Solve for coefficients (simplified - in practice would use selected solver)
    if solver == 'closed_form':
        coefficients = np.linalg.inv(X_norm.T @ X_norm) @ X_norm.T @ y_norm
    else:
        raise ValueError(f"Solver {solver} not implemented for this function")

    # Compute residuals
    residuals = compute_residuals(X_norm, y_norm, coefficients)

    # Calculate standard deviation of residuals
    std_residuals = np.std(residuals)

    # Prepare metrics dictionary
    metrics = {}
    if metric is not None:
        try:
            custom_metric = metric(residuals)
            metrics['custom'] = custom_metric
        except Exception as e:
            warnings.append(f"Custom metric computation failed: {str(e)}")

    # Return results
    return {
        'result': std_residuals,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'solver': solver
        },
        'warnings': []
    }

################################################################################
# matrice_design
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
    """
    Validate input matrices and vectors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target vector of shape (n_samples,) or None

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise ValueError("y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y is not None and y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or (y is not None and np.any(np.isnan(y))):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or (y is not None and np.any(np.isinf(y))):
        raise ValueError("Input arrays must not contain infinite values")

def normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Normalize the design matrix.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    method : str
        Normalization method ('none', 'standard', 'minmax', 'robust')

    Returns
    -------
    np.ndarray
        Normalized data
    """
    if method == 'none':
        return X
    elif method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1.0
        return (X - mean) / std
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        delta = max_val - min_val
        delta[delta == 0] = 1.0
        return (X - min_val) / delta
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        iqr[iqr == 0] = 1.0
        return (X - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   metrics: Union[str, Callable] = 'mse') -> Dict[str, float]:
    """
    Compute regression metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True values of shape (n_samples,)
    y_pred : np.ndarray
        Predicted values of shape (n_samples,)
    metrics : Union[str, Callable]
        Metric(s) to compute ('mse', 'mae', 'r2', 'logloss') or custom callable

    Returns
    -------
    Dict[str, float]
        Dictionary of computed metrics
    """
    result = {}

    if isinstance(metrics, str):
        metric_list = [m.strip() for m in metrics.split(',')]
    else:
        metric_list = ['custom']

    if 'mse' in metric_list or isinstance(metrics, str) and metrics == 'mse':
        result['mse'] = np.mean((y_true - y_pred) ** 2)

    if 'mae' in metric_list or isinstance(metrics, str) and metrics == 'mae':
        result['mae'] = np.mean(np.abs(y_true - y_pred))

    if 'r2' in metric_list or isinstance(metrics, str) and metrics == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        result['r2'] = 1 - (ss_res / ss_tot)

    if 'logloss' in metric_list or isinstance(metrics, str) and metrics == 'logloss':
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        result['logloss'] = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    if 'custom' in metric_list or isinstance(metrics, Callable):
        if isinstance(metrics, Callable):
            result['custom'] = metrics(y_true, y_pred)
        else:
            raise ValueError("Custom metric must be a callable function")

    return result

def closed_form_solver(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve linear regression using closed form solution.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)

    Returns
    -------
    np.ndarray
        Estimated coefficients of shape (n_features,)
    """
    XtX = np.dot(X.T, X)
    if np.linalg.matrix_rank(XtX) < X.shape[1]:
        raise ValueError("Matrix is singular or nearly singular")
    return np.linalg.solve(XtX, np.dot(X.T, y))

def gradient_descent_solver(X: np.ndarray, y: np.ndarray,
                          learning_rate: float = 0.01,
                          n_iter: int = 1000,
                          tol: float = 1e-4) -> np.ndarray:
    """
    Solve linear regression using gradient descent.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    learning_rate : float
        Learning rate for gradient descent
    n_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for stopping criterion

    Returns
    -------
    np.ndarray
        Estimated coefficients of shape (n_features,)
    """
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)
    prev_loss = float('inf')

    for _ in range(n_iter):
        gradients = -2 * np.dot(X.T, y - np.dot(X, coef)) / n_samples
        coef -= learning_rate * gradients

        current_loss = np.mean((y - np.dot(X, coef)) ** 2)
        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return coef

def matrice_design_fit(X: np.ndarray, y: Optional[np.ndarray] = None,
                      normalization: str = 'standard',
                      solver: str = 'closed_form',
                      metrics: Union[str, Callable] = 'mse',
                      **solver_kwargs) -> Dict:
    """
    Fit a linear regression model using design matrix.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target vector of shape (n_samples,) or None
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    solver : str
        Solver method ('closed_form', 'gradient_descent')
    metrics : Union[str, Callable]
        Metric(s) to compute ('mse', 'mae', 'r2', 'logloss') or custom callable
    **solver_kwargs
        Additional solver-specific parameters

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': estimated coefficients or None if y is None
        - 'metrics': computed metrics or empty dict if y is None
        - 'params_used': parameters used in the computation
        - 'warnings': any warnings generated during computation

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([1, 2, 3])
    >>> result = matrice_design_fit(X, y)
    """
    warnings = []
    params_used = {
        'normalization': normalization,
        'solver': solver,
        'metrics': metrics
    }

    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_normalized = normalize_data(X, normalization)
    if y is not None:
        y_normalized = normalize_data(y.reshape(-1, 1), normalization).flatten()
    else:
        y_normalized = None

    # Add intercept if needed
    if solver == 'closed_form' and np.mean(X_normalized, axis=0)[0] != 1:
        X_with_intercept = np.column_stack([np.ones(X_normalized.shape[0]), X_normalized])
    else:
        X_with_intercept = X_normalized

    # Solve for coefficients
    if y is not None:
        if solver == 'closed_form':
            coef = closed_form_solver(X_with_intercept, y_normalized)
        elif solver == 'gradient_descent':
            coef = gradient_descent_solver(X_with_intercept, y_normalized, **solver_kwargs)
        else:
            raise ValueError(f"Unknown solver: {solver}")

        # Compute predictions and metrics
        y_pred = np.dot(X_with_intercept, coef)
        metrics_result = compute_metrics(y_normalized, y_pred, metrics)

        return {
            'result': coef,
            'metrics': metrics_result,
            'params_used': params_used,
            'warnings': warnings
        }
    else:
        return {
            'result': None,
            'metrics': {},
            'params_used': params_used,
            'warnings': warnings
        }

################################################################################
# gradient_descendant
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for gradient descent."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def normalize_data(X: np.ndarray, y: np.ndarray, method: str = 'standard') -> tuple:
    """Normalize data using specified method."""
    if method == 'none':
        return X, y
    elif method == 'standard':
        mean_X = np.mean(X, axis=0)
        std_X = np.std(X, axis=0)
        X_normalized = (X - mean_X) / std_X
        y_normalized = (y - np.mean(y)) / np.std(y)
    elif method == 'minmax':
        min_X = np.min(X, axis=0)
        max_X = np.max(X, axis=0)
        X_normalized = (X - min_X) / (max_X - min_X + 1e-8)
        y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-8)
    elif method == 'robust':
        median_X = np.median(X, axis=0)
        iqr_X = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_normalized = (X - median_X) / (iqr_X + 1e-8)
        y_normalized = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25) + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return X_normalized, y_normalized

def compute_gradient(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                     metric: Callable[[np.ndarray, np.ndarray], float],
                     penalty: Optional[Callable[[np.ndarray], float]] = None,
                     alpha: float = 0.0) -> np.ndarray:
    """Compute gradient of the loss function."""
    predictions = X @ weights
    error = predictions - y
    gradient = (X.T @ error) / X.shape[0]
    if penalty is not None:
        gradient += alpha * penalty(weights)
    return gradient

def gradient_descent_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                          learning_rate: float, metric: Callable[[np.ndarray, np.ndarray], float],
                          penalty: Optional[Callable[[np.ndarray], float]] = None,
                          alpha: float = 0.0) -> np.ndarray:
    """Perform a single gradient descent step."""
    gradient = compute_gradient(X, y, weights, metric, penalty, alpha)
    return weights - learning_rate * gradient

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]]) -> Dict[str, float]:
    """Compute specified metrics."""
    return {name: metric(y_true, y_pred) for name, metric in metrics.items()}

def gradient_descendant_fit(X: np.ndarray, y: np.ndarray,
                           learning_rate: float = 0.01, max_iter: int = 1000,
                           tol: float = 1e-4, metric: str = 'mse',
                           normalization: str = 'standard',
                           penalty: Optional[str] = None,
                           alpha: float = 0.0) -> Dict:
    """
    Perform gradient descent for linear regression.

    Parameters:
        X: Input features (n_samples, n_features)
        y: Target values (n_samples,)
        learning_rate: Learning rate for gradient descent
        max_iter: Maximum number of iterations
        tol: Tolerance for stopping criterion
        metric: Metric to optimize ('mse', 'mae', 'r2')
        normalization: Normalization method ('none', 'standard', 'minmax', 'robust')
        penalty: Penalty type ('none', 'l1', 'l2', 'elasticnet')
        alpha: Regularization strength

    Returns:
        Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate inputs
    validate_inputs(X, y)

    # Initialize metrics dictionary
    metric_functions = {
        'mse': lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
        'mae': lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)),
        'r2': lambda y_true, y_pred: 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    }

    # Initialize penalty function
    penalty_functions = {
        'none': None,
        'l1': lambda weights: np.sign(weights),
        'l2': lambda weights: 2 * weights,
        'elasticnet': lambda weights: np.where(weights > 0, 1 + alpha, 1) * np.sign(weights)
    }

    # Get selected functions
    metric_func = metric_functions[metric]
    penalty_func = penalty_functions.get(penalty, None)

    # Normalize data
    X_norm, y_norm = normalize_data(X, y, normalization)

    # Initialize weights
    n_features = X_norm.shape[1]
    weights = np.zeros(n_features)

    # Gradient descent
    for i in range(max_iter):
        old_weights = weights.copy()
        weights = gradient_descent_step(X_norm, y_norm, weights,
                                       learning_rate, metric_func,
                                       penalty_func, alpha)

        # Check convergence
        if np.linalg.norm(weights - old_weights) < tol:
            break

    # Compute predictions and metrics
    y_pred = X_norm @ weights
    metrics_result = compute_metrics(y_norm, y_pred, {'metric': metric_func})

    # Return results
    return {
        'result': {
            'weights': weights,
            'iterations': i + 1
        },
        'metrics': metrics_result,
        'params_used': {
            'learning_rate': learning_rate,
            'max_iter': max_iter,
            'tol': tol,
            'metric': metric,
            'normalization': normalization,
            'penalty': penalty,
            'alpha': alpha
        },
        'warnings': []
    }

# Example usage:
"""
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([1, 2, 3])
result = gradient_descendant_fit(X, y)
"""

################################################################################
# algorithme_newton_raphson
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
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def _normalize_data(X: np.ndarray, y: np.ndarray,
                   normalization: str = 'standard',
                   custom_normalization: Optional[Callable] = None) -> tuple:
    """Normalize data according to specified method."""
    if custom_normalization is not None:
        X_norm, y_norm = custom_normalization(X, y)
    elif normalization == 'standard':
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y_norm = (y - np.mean(y)) / np.std(y)
    elif normalization == 'minmax':
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif normalization == 'robust':
        X_norm = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        y_norm = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    else:
        X_norm, y_norm = X, y
    return X_norm, y_norm

def _compute_gradient(X: np.ndarray, y: np.ndarray,
                     beta: np.ndarray) -> np.ndarray:
    """Compute gradient of the loss function."""
    residuals = y - X @ beta
    return -(X.T @ residuals) / len(y)

def _compute_hessian(X: np.ndarray) -> np.ndarray:
    """Compute Hessian matrix."""
    return X.T @ X / len(X)

def _newton_raphson_step(X: np.ndarray, y: np.ndarray,
                        beta: np.ndarray) -> np.ndarray:
    """Perform one Newton-Raphson iteration."""
    gradient = _compute_gradient(X, y, beta)
    hessian = _compute_hessian(X)
    return beta - np.linalg.inv(hessian) @ gradient

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metric: str = 'mse',
                    custom_metric: Optional[Callable] = None) -> Dict[str, float]:
    """Compute specified metrics."""
    if custom_metric is not None:
        return {'custom': custom_metric(y_true, y_pred)}

    metrics = {}
    if metric == 'mse' or 'all' in metric:
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    if metric == 'mae' or 'all' in metric:
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    if metric == 'r2' or 'all' in metric:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)

    return metrics

def algorithme_newton_raphson_fit(X: np.ndarray, y: np.ndarray,
                                normalization: str = 'standard',
                                metric: str = 'mse',
                                tol: float = 1e-6,
                                max_iter: int = 1000,
                                custom_normalization: Optional[Callable] = None,
                                custom_metric: Optional[Callable] = None) -> Dict:
    """
    Fit linear regression model using Newton-Raphson algorithm.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalization : str or callable, optional
        Normalization method ('standard', 'minmax', 'robust') or custom function
    metric : str or callable, optional
        Evaluation metric ('mse', 'mae', 'r2') or custom function
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
    custom_normalization : callable, optional
        Custom normalization function
    custom_metric : callable, optional
        Custom metric function

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': fitted parameters
        - 'metrics': computed metrics
        - 'params_used': parameters used in fitting
        - 'warnings': any warnings generated

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = algorithme_newton_raphson_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Initialize variables
    n_features = X.shape[1]
    beta = np.zeros(n_features)
    warnings = []

    # Normalize data
    X_norm, y_norm = _normalize_data(X, y, normalization, custom_normalization)

    # Newton-Raphson iterations
    for _ in range(max_iter):
        beta_new = _newton_raphson_step(X_norm, y_norm, beta)
        if np.linalg.norm(beta_new - beta) < tol:
            break
        beta = beta_new

    # Compute predictions and metrics
    y_pred = X_norm @ beta
    if normalization != 'none':
        # Denormalize predictions if data was normalized
        y_pred = y_pred * np.std(y) + np.mean(y)

    metrics = _compute_metrics(y, y_pred, metric, custom_metric)

    return {
        'result': beta,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization if custom_normalization is None else 'custom',
            'metric': metric if custom_metric is None else 'custom',
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': warnings
    }

################################################################################
# regression_ridge
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def regression_ridge_fit(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    normalize: str = 'standard',
    solver: str = 'closed_form',
    metric: Union[str, Callable] = 'mse',
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None
) -> Dict:
    """
    Perform Ridge Regression (L2 regularization).

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    alpha : float
        Regularization strength (default=1.0)
    normalize : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    solver : str
        Solver method ('closed_form', 'gradient_descent')
    metric : Union[str, Callable]
        Metric to evaluate ('mse', 'mae', 'r2', or custom callable)
    max_iter : int
        Maximum iterations for iterative solvers (default=1000)
    tol : float
        Tolerance for convergence (default=1e-4)
    random_state : Optional[int]
        Random seed for reproducibility

    Returns:
    --------
    Dict containing:
        - 'result': Dictionary with regression results
        - 'metrics': Dictionary of computed metrics
        - 'params_used': Dictionary of parameters used
        - 'warnings': List of warnings encountered

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = regression_ridge_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize data
    X_norm, y_norm = _normalize_data(X, y, normalize)

    # Initialize parameters
    n_samples, n_features = X_norm.shape

    # Choose solver
    if solver == 'closed_form':
        coef, intercept = _ridge_closed_form(X_norm, y_norm, alpha)
    elif solver == 'gradient_descent':
        coef, intercept = _ridge_gradient_descent(
            X_norm, y_norm, alpha,
            max_iter=max_iter,
            tol=tol
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate predictions
    y_pred = X_norm @ coef + intercept

    # Calculate metrics
    metrics = _calculate_metrics(y_norm, y_pred, metric)

    # Prepare output
    result = {
        'coef': coef,
        'intercept': intercept,
        'alpha': alpha
    }

    params_used = {
        'normalize': normalize,
        'solver': solver,
        'metric': metric if isinstance(metric, str) else 'custom',
        'max_iter': max_iter,
        'tol': tol
    }

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
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

def _normalize_data(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'standard'
) -> tuple:
    """Normalize features and target."""
    X_norm = X.copy()
    y_norm = y.copy()

    if method == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1.0
        X_norm = (X - X_mean) / X_std

        y_mean = np.mean(y)
        y_std = np.std(y)
        if y_std != 0:
            y_norm = (y - y_mean) / y_std

    elif method == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)

        y_min = np.min(y)
        y_max = np.max(y)
        if y_max != y_min:
            y_norm = (y - y_min) / (y_max - y_min)

    elif method == 'robust':
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_iqr[X_iqr == 0] = 1.0
        X_norm = (X - X_median) / X_iqr

    return X_norm, y_norm

def _ridge_closed_form(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float
) -> tuple:
    """Closed-form solution for Ridge Regression."""
    n_samples, n_features = X.shape
    identity = np.eye(n_features)
    coef = np.linalg.inv(X.T @ X + alpha * identity) @ X.T @ y
    intercept = np.mean(y - X @ coef)
    return coef, intercept

def _ridge_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> tuple:
    """Gradient Descent solver for Ridge Regression."""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)
    intercept = 0.0

    for _ in range(max_iter):
        # Compute predictions
        y_pred = X @ coef + intercept

        # Compute gradients
        gradient_coef = (-2/n_samples) * X.T @ (y - y_pred) + 2*alpha*coef
        gradient_intercept = (-2/n_samples) * np.sum(y - y_pred)

        # Update parameters
        coef_new = coef - 0.1 * gradient_coef
        intercept_new = intercept - 0.1 * gradient_intercept

        # Check convergence
        if np.linalg.norm([coef_new - coef, intercept_new - intercept]) < tol:
            break

        coef, intercept = coef_new, intercept_new

    return coef, intercept

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Calculate regression metrics."""
    metrics = {}

    if isinstance(metric, str):
        if metric == 'mse':
            metrics['mse'] = np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        metrics['custom'] = metric(y_true, y_pred)

    return metrics

################################################################################
# regression_lasso
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def regression_lasso_fit(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-4,
    normalize: str = 'standard',
    solver: str = 'coordinate_descent',
    metric: Union[str, Callable] = 'mse',
    penalty: str = 'l1',
    random_state: Optional[int] = None,
    **kwargs
) -> Dict:
    """
    Perform Lasso regression (L1 regularized linear regression).

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    alpha : float, default=1.0
        Regularization strength
    max_iter : int, default=1000
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for optimization
    normalize : str, default='standard'
        Normalization method ('none', 'standard', 'minmax', 'robust')
    solver : str, default='coordinate_descent'
        Solver to use ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    metric : str or callable, default='mse'
        Metric to compute ('mse', 'mae', 'r2', custom callable)
    penalty : str, default='l1'
        Penalty type ('none', 'l1', 'l2')
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    Dict containing:
        - 'result': dict with coefficients and intercept
        - 'metrics': dict of computed metrics
        - 'params_used': dict of parameters used
        - 'warnings': list of warnings

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = regression_lasso_fit(X, y, alpha=0.5)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set random state if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize data
    X_norm, y_norm = _apply_normalization(X, y, normalize)

    # Initialize parameters
    n_features = X_norm.shape[1]
    coef = np.zeros(n_features)
    intercept = 0.0

    # Choose solver
    if solver == 'coordinate_descent':
        coef, intercept = _coordinate_descent_lasso(
            X_norm, y_norm, alpha=alpha,
            max_iter=max_iter, tol=tol
        )
    elif solver == 'gradient_descent':
        coef, intercept = _gradient_descent_lasso(
            X_norm, y_norm, alpha=alpha,
            max_iter=max_iter, tol=tol
        )
    else:
        raise ValueError(f"Unsupported solver: {solver}")

    # Compute metrics
    y_pred = _predict(X_norm, coef, intercept)
    metrics = _compute_metrics(y_norm, y_pred, metric)

    # Prepare output
    result = {
        'coefficients': coef,
        'intercept': intercept
    }

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'alpha': alpha,
            'max_iter': max_iter,
            'tol': tol,
            'normalize': normalize,
            'solver': solver,
            'metric': metric,
            'penalty': penalty
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

def _apply_normalization(
    X: np.ndarray,
    y: np.ndarray,
    method: str
) -> tuple:
    """Apply normalization to features and target."""
    if method == 'standard':
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y_norm = (y - np.mean(y)) / np.std(y)
    elif method == 'minmax':
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif method == 'robust':
        X_norm = (X - np.median(X, axis=0)) / (
            np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
        )
        y_norm = (y - np.median(y)) / (
            np.percentile(y, 75) - np.percentile(y, 25)
        )
    else:
        X_norm = X.copy()
        y_norm = y.copy()

    return X_norm, y_norm

def _coordinate_descent_lasso(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float
) -> tuple:
    """Coordinate descent algorithm for Lasso regression."""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)
    intercept = 0.0

    for _ in range(max_iter):
        old_coef = coef.copy()

        for j in range(n_features):
            # Compute residual without feature j
            r = y - intercept - np.dot(X[:, :j], coef[:j]) - np.dot(X[:, j+1:], coef[j+1:])

            # Compute correlation
            corr = np.dot(X[:, j], r)

            if alpha == 0:
                coef[j] = corr / np.dot(X[:, j], X[:, j])
            else:
                # Soft-thresholding
                if corr < -alpha/2:
                    coef[j] = (corr + alpha/2) / np.dot(X[:, j], X[:, j])
                elif corr > alpha/2:
                    coef[j] = (corr - alpha/2) / np.dot(X[:, j], X[:, j])
                else:
                    coef[j] = 0

        # Update intercept
        intercept = np.mean(y - np.dot(X, coef))

        # Check convergence
        if np.linalg.norm(coef - old_coef) < tol:
            break

    return coef, intercept

def _gradient_descent_lasso(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float
) -> tuple:
    """Gradient descent algorithm for Lasso regression."""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)
    intercept = 0.0

    learning_rate = 0.1
    for _ in range(max_iter):
        old_coef = coef.copy()

        # Compute gradients
        y_pred = _predict(X, coef, intercept)
        error = y_pred - y

        # Gradient for coefficients
        grad_coef = np.dot(X.T, error) / n_samples + alpha * np.sign(coef)

        # Gradient for intercept
        grad_intercept = np.mean(error)

        # Update parameters
        coef -= learning_rate * grad_coef
        intercept -= learning_rate * grad_intercept

        # Check convergence
        if np.linalg.norm(coef - old_coef) < tol:
            break

    return coef, intercept

def _predict(
    X: np.ndarray,
    coef: np.ndarray,
    intercept: float
) -> np.ndarray:
    """Make predictions using linear model."""
    return np.dot(X, coef) + intercept

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Compute metrics for regression."""
    metrics = {}

    if isinstance(metric, str):
        if metric == 'mse':
            metrics['mse'] = np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['r2'] = 1 - ss_res / ss_tot
    else:
        metrics['custom'] = metric(y_true, y_pred)

    return metrics

################################################################################
# elastic_net
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def elastic_net_fit(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    normalize: str = "standard",
    solver: str = "coordinate_descent",
    max_iter: int = 1000,
    tol: float = 1e-4,
    metric: str = "mse",
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, np.ndarray]]:
    """
    Fit an Elastic Net regression model.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    alpha : float
        Regularization strength. Must be >= 0.
    l1_ratio : float
        Mixing parameter between L1 and L2 regularization. Must be in [0, 1].
    normalize : str
        Normalization method: "none", "standard", "minmax", or "robust".
    solver : str
        Solver to use: "coordinate_descent" or "gradient_descent".
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for stopping criteria.
    metric : str or callable
        Metric to evaluate: "mse", "mae", "r2", or custom callable.
    random_state : int, optional
        Random seed for reproducibility.
    custom_metric : callable, optional
        Custom metric function.

    Returns
    -------
    dict
        Dictionary containing:
        - "result": fitted coefficients
        - "metrics": computed metrics
        - "params_used": parameters used for fitting
        - "warnings": any warnings encountered

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = elastic_net_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set random state if provided
    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    # Normalize data
    X_norm, y_norm = _normalize_data(X, y, normalize)

    # Initialize coefficients
    n_features = X_norm.shape[1]
    coef = np.zeros(n_features)

    # Choose solver
    if solver == "coordinate_descent":
        coef = _coordinate_descent(X_norm, y_norm, alpha, l1_ratio, max_iter, tol, rng)
    elif solver == "gradient_descent":
        coef = _gradient_descent(X_norm, y_norm, alpha, l1_ratio, max_iter, tol, rng)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate predictions
    y_pred = X_norm @ coef

    # Calculate metrics
    metrics = _calculate_metrics(y_norm, y_pred, metric, custom_metric)

    # Prepare output
    result = {
        "result": coef,
        "metrics": metrics,
        "params_used": {
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "normalize": normalize,
            "solver": solver,
            "max_iter": max_iter,
            "tol": tol
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
        raise ValueError("X contains NaN or inf values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or inf values")

def _normalize_data(
    X: np.ndarray,
    y: np.ndarray,
    method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize data according to specified method."""
    X_norm = X.copy()
    y_norm = y.copy()

    if method == "standard":
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1  # avoid division by zero
        X_norm = (X - X_mean) / X_std

        y_mean = np.mean(y)
        y_std = np.std(y)
        if y_std != 0:
            y_norm = (y - y_mean) / y_std

    elif method == "minmax":
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)

        y_min = np.min(y)
        y_max = np.max(y)
        if y_max != y_min:
            y_norm = (y - y_min) / (y_max - y_min)

    elif method == "robust":
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_iqr[X_iqr == 0] = 1
        X_norm = (X - X_median) / X_iqr

    return X_norm, y_norm

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tol: float,
    rng: np.random.RandomState
) -> np.ndarray:
    """Coordinate descent solver for Elastic Net."""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)
    l2_penalty = alpha * (1 - l1_ratio)
    l1_penalty = alpha * l1_rati

    for _ in range(max_iter):
        coef_prev = coef.copy()

        for j in range(n_features):
            # Compute residuals without feature j
            r = y - X[:, np.arange(n_features) != j] @ coef[np.arange(n_features) != j]

            # Compute correlation
            rho = X[:, j] @ r

            # Compute L1 penalty term
            if coef[j] > 0:
                sign = 1
            elif coef[j] < 0:
                sign = -1
            else:
                sign = rng.choice([-1, 1])

            # Compute new coefficient
            coef_j_new = np.sign(rho) * np.maximum(
                np.abs(rho) - l1_penalty,
                0
            ) / (X[:, j] @ X[:, j] + l2_penalty)

            # Update coefficient
            coef[j] = coef_j_new

        # Check convergence
        if np.linalg.norm(coef - coef_prev) < tol:
            break

    return coef

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tol: float,
    rng: np.random.RandomState
) -> np.ndarray:
    """Gradient descent solver for Elastic Net."""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)
    l2_penalty = alpha * (1 - l1_ratio)
    l1_penalty = alpha * l1_ratio
    learning_rate = 0.1

    for _ in range(max_iter):
        coef_prev = coef.copy()

        # Compute gradient
        grad = X.T @ (X @ coef - y) / n_samples

        # Add L1 penalty gradient
        grad += l1_penalty * np.sign(coef)

        # Add L2 penalty gradient
        grad += 2 * l2_penalty * coef

        # Update coefficients
        coef -= learning_rate * grad

        # Check convergence
        if np.linalg.norm(coef - coef_prev) < tol:
            break

    return coef

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    metrics = {}

    if metric == "mse" or custom_metric is None:
        mse = np.mean((y_true - y_pred) ** 2)
        metrics["mse"] = mse

    if metric == "mae" or custom_metric is None:
        mae = np.mean(np.abs(y_true - y_pred))
        metrics["mae"] = mae

    if metric == "r2" or custom_metric is None:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        metrics["r2"] = r2

    if custom_metric is not None:
        try:
            metrics["custom"] = custom_metric(y_true, y_pred)
        except Exception as e:
            metrics["custom_error"] = str(e)

    return metrics

################################################################################
# validation_croisee
################################################################################

import numpy as np
from typing import Callable, Dict, List, Optional, Union

def validation_croisee_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    normalisation: str = 'standard',
    metrics: Union[str, List[str], Callable] = ['mse', 'r2'],
    solver: str = 'closed_form',
    reg_type: Optional[str] = None,
    alpha: float = 1.0,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Perform cross-validation for linear regression with configurable options.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    n_splits : int, optional
        Number of cross-validation splits (default: 5)
    normalisation : str, optional
        Type of normalization ('none', 'standard', 'minmax', 'robust') (default: 'standard')
    metrics : Union[str, List[str], Callable], optional
        Metrics to compute ('mse', 'mae', 'r2', 'logloss') or custom callable (default: ['mse', 'r2'])
    solver : str, optional
        Solver to use ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent') (default: 'closed_form')
    reg_type : Optional[str], optional
        Type of regularization ('none', 'l1', 'l2', 'elasticnet') (default: None)
    alpha : float, optional
        Regularization strength (default: 1.0)
    random_state : Optional[int], optional
        Random seed for reproducibility (default: None)
    custom_metric : Optional[Callable], optional
        Custom metric function (default: None)

    Returns:
    --------
    Dict containing:
        - 'result': List of results for each fold
        - 'metrics': Dictionary of average metrics
        - 'params_used': Parameters used in the computation
        - 'warnings': List of warnings encountered

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> results = validation_croisee_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Initialize results dictionary
    results = {
        'result': [],
        'metrics': {},
        'params_used': {
            'n_splits': n_splits,
            'normalisation': normalisation,
            'metrics': metrics,
            'solver': solver,
            'reg_type': reg_type,
            'alpha': alpha
        },
        'warnings': []
    }

    # Normalize data if needed
    X_norm, y_norm = _apply_normalisation(X, y, normalisation)

    # Prepare metrics
    metric_functions = _prepare_metrics(metrics, custom_metric)

    # Perform cross-validation
    for fold in range(n_splits):
        # Split data
        X_train, X_test, y_train, y_test = _kfold_split(X_norm, y_norm, n_splits, fold, random_state)

        # Fit model
        if solver == 'closed_form':
            coefs, intercept = _closed_form_solver(X_train, y_train, reg_type, alpha)
        elif solver == 'gradient_descent':
            coefs, intercept = _gradient_descent_solver(X_train, y_train, reg_type, alpha)
        elif solver == 'newton':
            coefs, intercept = _newton_solver(X_train, y_train, reg_type, alpha)
        elif solver == 'coordinate_descent':
            coefs, intercept = _coordinate_descent_solver(X_train, y_train, reg_type, alpha)
        else:
            raise ValueError(f"Unknown solver: {solver}")

        # Make predictions
        y_pred = _predict(X_test, coefs, intercept)

        # Compute metrics
        fold_metrics = _compute_metrics(y_test, y_pred, metric_functions)

        # Store results
        results['result'].append({
            'fold': fold,
            'coefs': coefs,
            'intercept': intercept,
            'metrics': fold_metrics
        })

    # Compute average metrics
    results['metrics'] = _compute_average_metrics(results['result'], metric_functions)

    return results

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains NaN or infinite values")
    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("y contains NaN or infinite values")

def _apply_normalisation(X: np.ndarray, y: np.ndarray, normalisation: str) -> tuple:
    """Apply specified normalization to data."""
    X_norm = X.copy()
    y_norm = y.copy()

    if normalisation == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_norm = (X - X_mean) / X_std
    elif normalisation == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
    elif normalisation == 'robust':
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - X_median) / (X_iqr + 1e-8)

    return X_norm, y_norm

def _prepare_metrics(metrics: Union[str, List[str], Callable],
                    custom_metric: Optional[Callable]) -> Dict:
    """Prepare metric functions."""
    metric_functions = {}

    if isinstance(metrics, str):
        metrics = [metrics]

    for metric in metrics:
        if callable(metric):
            metric_functions['custom'] = metric
        elif metric == 'mse':
            metric_functions['mse'] = _mean_squared_error
        elif metric == 'mae':
            metric_functions['mae'] = _mean_absolute_error
        elif metric == 'r2':
            metric_functions['r2'] = _r_squared
        elif metric == 'logloss':
            metric_functions['logloss'] = _log_loss
        else:
            raise ValueError(f"Unknown metric: {metric}")

    if custom_metric is not None:
        metric_functions['custom'] = custom_metric

    return metric_functions

def _kfold_split(X: np.ndarray, y: np.ndarray,
                n_splits: int, fold: int,
                random_state: Optional[int]) -> tuple:
    """Split data into train and test sets for k-fold cross-validation."""
    np.random.seed(random_state)
    indices = np.random.permutation(X.shape[0])

    fold_size = X.shape[0] // n_splits
    test_indices = indices[fold * fold_size:(fold + 1) * fold_size]
    train_indices = np.setdiff1d(indices, test_indices)

    return (X[train_indices], X[test_indices],
            y[train_indices], y[test_indices])

def _closed_form_solver(X: np.ndarray, y: np.ndarray,
                       reg_type: Optional[str], alpha: float) -> tuple:
    """Closed form solution for linear regression."""
    n_samples, n_features = X.shape

    # Add regularization if needed
    if reg_type == 'l1':
        penalty = alpha * np.eye(n_features)
        np.fill_diagonal(penalty, 0)  # L1 doesn't penalize intercept
    elif reg_type == 'l2':
        penalty = alpha * np.eye(n_features)
    elif reg_type == 'elasticnet':
        l1_ratio = 0.5
        penalty = alpha * (l1_ratio * np.eye(n_features) +
                          (1 - l1_ratio) * np.eye(n_features))
        np.fill_diagonal(penalty, alpha * l1_ratio)
    else:
        penalty = np.zeros((n_features, n_features))

    # Compute coefficients
    XtX = X.T @ X + penalty
    Xty = X.T @ y

    try:
        coefs = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        coefs = np.linalg.pinv(XtX) @ Xty

    intercept = y.mean() - coefs.T @ np.mean(X, axis=0)

    return coefs, intercept

def _gradient_descent_solver(X: np.ndarray, y: np.ndarray,
                            reg_type: Optional[str], alpha: float) -> tuple:
    """Gradient descent solver for linear regression."""
    n_samples, n_features = X.shape
    coefs = np.zeros(n_features)
    intercept = 0.0

    learning_rate = 0.01
    n_iterations = 1000

    for _ in range(n_iterations):
        # Compute predictions
        y_pred = X @ coefs + intercept

        # Compute gradients
        error = y_pred - y
        grad_coefs = (X.T @ error) / n_samples

        if reg_type == 'l1':
            grad_coefs += alpha * np.sign(coefs)
        elif reg_type == 'l2':
            grad_coefs += 2 * alpha * coefs

        # Update parameters
        coefs -= learning_rate * grad_coefs
        intercept -= learning_rate * np.mean(error)

    return coefs, intercept

def _newton_solver(X: np.ndarray, y: np.ndarray,
                  reg_type: Optional[str], alpha: float) -> tuple:
    """Newton's method solver for linear regression."""
    n_samples, n_features = X.shape
    coefs = np.zeros(n_features)
    intercept = 0.0

    n_iterations = 100
    tol = 1e-4

    for _ in range(n_iterations):
        # Compute predictions
        y_pred = X @ coefs + intercept

        # Compute gradients and Hessian
        error = y_pred - y
        grad_coefs = (X.T @ error) / n_samples

        if reg_type == 'l1':
            grad_coefs += alpha * np.sign(coefs)
        elif reg_type == 'l2':
            grad_coefs += 2 * alpha * coefs

        hessian = (X.T @ X) / n_samples
        if reg_type == 'l2':
            hessian += 2 * alpha * np.eye(n_features)

        # Update parameters
        delta = -np.linalg.solve(hessian, grad_coefs)
        coefs += delta
        intercept -= np.mean(error)

        # Check convergence
        if np.linalg.norm(delta) < tol:
            break

    return coefs, intercept

def _coordinate_descent_solver(X: np.ndarray, y: np.ndarray,
                              reg_type: Optional[str], alpha: float) -> tuple:
    """Coordinate descent solver for linear regression."""
    n_samples, n_features = X.shape
    coefs = np.zeros(n_features)
    intercept = 0.0

    n_iterations = 100
    tol = 1e-4

    for _ in range(n_iterations):
        old_coefs = coefs.copy()

        for j in range(n_features):
            # Compute residual without current feature
            X_j = X[:, j]
            r = y - intercept - (X @ coefs) + coefs[j] * X_j

            # Compute optimal value for current coefficient
            if reg_type == 'l1':
                coefs[j] = _soft_threshold(r @ X_j, alpha * np.linalg.norm(X_j))
            elif reg_type == 'l2':
                coefs[j] = (r @ X_j) / (np.linalg.norm(X_j)**2 + alpha)
            else:
                coefs[j] = (r @ X_j) / np.linalg.norm(X_j)**2

        # Update intercept
        intercept = np.mean(y - X @ coefs)

        # Check convergence
        if np.linalg.norm(coefs - old_coefs) < tol:
            break

    return coefs, intercept

def _soft_threshold(x: float, threshold: float) -> float:
    """Soft-thresholding function for L1 regularization."""
    if x > threshold:
        return x - threshold
    elif x < -threshold:
        return x + threshold
    else:
        return 0.0

def _predict(X: np.ndarray, coefs: np.ndarray, intercept: float) -> np.ndarray:
    """Make predictions using linear regression model."""
    return X @ coefs + intercept

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metric_functions: Dict) -> Dict:
    """Compute specified metrics."""
    metrics = {}

    for name, func in metric_functions.items():
        if name == 'mse':
            metrics['mse'] = _mean_squared_error(y_true, y_pred)
        elif name == 'mae':
            metrics['mae'] = _mean_absolute_error(y_true, y_pred)
        elif name == 'r2':
            metrics['r2'] = _r_squared(y_true, y_pred)
        elif name == 'logloss':
            metrics['logloss'] = _log_loss(y_true, y_pred)
        elif name == 'custom':
            metrics['custom'] = func(y_true, y_pred)

    return metrics

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute log loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _compute_average_metrics(results: List[Dict], metric_functions: Dict) -> Dict:
    """Compute average metrics across all folds."""
    avg_metrics = {}

    for metric_name in metric_functions.keys():
        values = [fold['metrics'][metric_name] for fold in results]
        avg_metrics[metric_name] = np.mean(values)

    return avg_metrics

################################################################################
# splines
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def splines_fit(
    X: np.ndarray,
    y: np.ndarray,
    knots: Optional[np.ndarray] = None,
    degree: int = 3,
    solver: str = 'closed_form',
    metric: Union[str, Callable] = 'mse',
    normalization: str = 'none',
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fit spline regression model to data.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    knots : Optional[np.ndarray]
        Positions of the knots. If None, uses quantiles.
    degree : int
        Degree of the spline basis (default: 3 for cubic)
    solver : str
        Solver to use ('closed_form', 'gradient_descent')
    metric : Union[str, Callable]
        Metric to optimize ('mse', 'mae', 'r2') or custom callable
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    regularization : Optional[str]
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    alpha : float
        Regularization strength (default: 1.0)
    tol : float
        Tolerance for convergence (default: 1e-4)
    max_iter : int
        Maximum iterations (default: 1000)
    custom_metric : Optional[Callable]
        Custom metric function
    custom_distance : Optional[Callable]
        Custom distance function

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 2)
    >>> y = np.random.rand(100)
    >>> result = splines_fit(X, y, knots=np.linspace(0, 1, 5), degree=3)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if requested
    X_norm, y_norm = _apply_normalization(X, y, normalization)

    # Prepare spline basis
    basis = _create_spline_basis(X_norm, knots=knots, degree=degree)

    # Choose solver
    if solver == 'closed_form':
        params = _solve_closed_form(basis, y_norm, regularization, alpha)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(
            basis, y_norm, metric, custom_metric,
            tol, max_iter, regularization, alpha
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    y_pred = basis @ params
    metrics = _calculate_metrics(y_norm, y_pred, metric, custom_metric)

    return {
        'result': {'coefficients': params},
        'metrics': metrics,
        'params_used': {
            'knots': knots,
            'degree': degree,
            'solver': solver,
            'metric': metric,
            'normalization': normalization,
            'regularization': regularization,
            'alpha': alpha
        },
        'warnings': _check_warnings(y_pred)
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input contains NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input contains infinite values")

def _apply_normalization(
    X: np.ndarray,
    y: np.ndarray,
    method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply requested normalization to data."""
    if method == 'none':
        return X, y
    elif method == 'standard':
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y_norm = (y - np.mean(y)) / np.std(y)
    elif method == 'minmax':
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif method == 'robust':
        X_norm = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        y_norm = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return X_norm, y_norm

def _create_spline_basis(
    X: np.ndarray,
    knots: Optional[np.ndarray] = None,
    degree: int = 3
) -> np.ndarray:
    """Create spline basis matrix."""
    n_samples, n_features = X.shape
    if knots is None:
        knots = np.linspace(np.min(X), np.max(X), 5)

    basis = []
    for feature in range(n_features):
        # Create B-spline basis
        bsplines = _calculate_bsplines(X[:, feature], knots, degree)
        basis.append(bsplines)

    return np.hstack(basis)

def _calculate_bsplines(
    x: np.ndarray,
    knots: np.ndarray,
    degree: int
) -> np.ndarray:
    """Calculate B-spline basis functions."""
    # Implementation of B-spline calculation
    pass

def _solve_closed_form(
    basis: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str],
    alpha: float
) -> np.ndarray:
    """Solve using closed form solution."""
    if regularization is None:
        params = np.linalg.pinv(basis) @ y
    elif regularization == 'l2':
        params = _ridge_regression(basis, y, alpha)
    elif regularization == 'l1':
        params = _lasso_regression(basis, y, alpha)
    elif regularization == 'elasticnet':
        params = _elasticnet_regression(basis, y, alpha)
    else:
        raise ValueError(f"Unknown regularization: {regularization}")
    return params

def _ridge_regression(
    basis: np.ndarray,
    y: np.ndarray,
    alpha: float
) -> np.ndarray:
    """Ridge regression solver."""
    return np.linalg.inv(basis.T @ basis + alpha * np.eye(basis.shape[1])) @ basis.T @ y

def _lasso_regression(
    basis: np.ndarray,
    y: np.ndarray,
    alpha: float
) -> np.ndarray:
    """Lasso regression solver (simplified)."""
    # In practice, use coordinate descent or specialized solvers
    return np.linalg.pinv(basis + alpha * np.diag(np.sign(np.random.randn(basis.shape[1])))) @ y

def _elasticnet_regression(
    basis: np.ndarray,
    y: np.ndarray,
    alpha: float
) -> np.ndarray:
    """ElasticNet regression solver (simplified)."""
    # In practice, use coordinate descent or specialized solvers
    return np.linalg.pinv(basis + alpha * (0.5 * np.eye(basis.shape[1]) +
                                          0.5 * np.diag(np.sign(np.random.randn(basis.shape[1]))))) @ y

def _solve_gradient_descent(
    basis: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable],
    tol: float,
    max_iter: int,
    regularization: Optional[str],
    alpha: float
) -> np.ndarray:
    """Solve using gradient descent."""
    n_params = basis.shape[1]
    params = np.zeros(n_params)

    for _ in range(max_iter):
        # Calculate predictions
        y_pred = basis @ params

        # Calculate gradient based on metric
        if custom_metric is not None:
            grad = _custom_gradient(basis, y, y_pred, custom_metric)
        elif metric == 'mse':
            grad = 2 * basis.T @ (y_pred - y) / len(y)
        elif metric == 'mae':
            grad = basis.T @ np.sign(y_pred - y) / len(y)
        elif metric == 'r2':
            grad = 2 * basis.T @ (y_pred - y) / len(y)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Add regularization if needed
        if regularization == 'l2':
            grad += 2 * alpha * params
        elif regularization == 'l1':
            grad += alpha * np.sign(params)
        elif regularization == 'elasticnet':
            grad += 2 * alpha * params + alpha * np.sign(params)

        # Update parameters
        params_new = params - grad

        # Check convergence
        if np.linalg.norm(params_new - params) < tol:
            break

        params = params_new

    return params

def _custom_gradient(
    basis: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable
) -> np.ndarray:
    """Calculate gradient for custom metric."""
    # This is a placeholder - actual implementation depends on the metric
    return basis.T @ (metric_func(y_pred, y_true) - metric_func(y_true, y_true))

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Calculate requested metrics."""
    metrics = {}

    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_pred, y_true)
    else:
        if metric == 'mse' or metric is None:
            metrics['mse'] = np.mean((y_pred - y_true) ** 2)
        if metric == 'mae' or metric is None:
            metrics['mae'] = np.mean(np.abs(y_pred - y_true))
        if metric == 'r2' or metric is None:
            ss_res = np.sum((y_pred - y_true) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['r2'] = 1 - ss_res / ss_tot

    return metrics

def _check_warnings(y_pred: np.ndarray) -> list:
    """Check for potential warnings."""
    warnings = []
    if np.any(np.isnan(y_pred)):
        warnings.append("Predictions contain NaN values")
    if np.any(np.isinf(y_pred)):
        warnings.append("Predictions contain infinite values")
    return warnings

################################################################################
# polynomial_features
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def polynomial_features_fit(
    X: np.ndarray,
    degree: int = 2,
    include_bias: bool = True,
    normalize: str = 'standard',
    custom_normalize: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, Union[np.ndarray, Dict[str, str], list]]:
    """
    Compute polynomial features for linear regression.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    degree : int, optional
        Degree of the polynomial features to generate (default=2).
    include_bias : bool, optional
        Whether to include a bias term (default=True).
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default='standard').
    custom_normalize : callable, optional
        Custom normalization function.

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': Transformed features of shape (n_samples, n_features_poly)
        - 'params_used': Parameters used
        - 'warnings': List of warnings

    Example
    -------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> result = polynomial_features_fit(X)
    """
    # Validate inputs
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X must be a 2D numpy array")
    if degree < 1:
        raise ValueError("degree must be >= 1")

    # Initialize warnings
    warnings = []

    # Apply custom normalization if provided
    if custom_normalize is not None:
        X = custom_normalize(X)
    # Apply built-in normalization
    elif normalize != 'none':
        X = _normalize_features(X, method=normalize)

    # Compute polynomial features
    X_poly = _compute_polynomial_features(X, degree=degree, include_bias=include_bias)

    return {
        'result': X_poly,
        'params_used': {
            'degree': degree,
            'include_bias': include_bias,
            'normalize': normalize if custom_normalize is None else 'custom'
        },
        'warnings': warnings
    }

def _normalize_features(
    X: np.ndarray,
    method: str = 'standard'
) -> np.ndarray:
    """
    Normalize features using specified method.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    method : str
        Normalization method ('standard', 'minmax', 'robust').

    Returns
    -------
    np.ndarray
        Normalized features.
    """
    if method == 'standard':
        X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        X_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        X_normalized = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Handle division by zero
    X_normalized[~np.isfinite(X_normalized)] = 0

    return X_normalized

def _compute_polynomial_features(
    X: np.ndarray,
    degree: int,
    include_bias: bool
) -> np.ndarray:
    """
    Compute polynomial features.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    degree : int
        Degree of the polynomial features.
    include_bias : bool
        Whether to include a bias term.

    Returns
    -------
    np.ndarray
        Polynomial features of shape (n_samples, n_features_poly).
    """
    n_samples, n_features = X.shape
    n_output_features = n_features + (n_features * (n_features - 1)) // 2

    # Initialize output array
    X_poly = np.zeros((n_samples, n_output_features))

    # Fill with polynomial terms
    col = 0
    for i in range(n_features):
        X_poly[:, col] = X[:, i]
        col += 1
    for i in range(n_features):
        for j in range(i + 1, n_features):
            X_poly[:, col] = X[:, i] * X[:, j]
            col += 1

    # Add bias term if requested
    if include_bias:
        X_poly = np.hstack([np.ones((n_samples, 1)), X_poly])

    return X_poly
