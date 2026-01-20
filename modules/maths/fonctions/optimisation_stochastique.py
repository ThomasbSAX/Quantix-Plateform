"""
Quantix – Module optimisation_stochastique
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# algorithmes_gradient
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> None:
    """Validate input data and normalizer."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input data contains NaN values")
    if normalizer is not None:
        try:
            X_normalized = normalizer(X)
        except Exception as e:
            raise ValueError(f"Normalizer function failed: {str(e)}")

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> float:
    """Compute the specified metric between true and predicted values."""
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
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    n_iter: int = 1000,
    tol: float = 1e-4
) -> np.ndarray:
    """Perform gradient descent optimization."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    prev_loss = float('inf')

    for _ in range(n_iter):
        gradients = 2/n_samples * X.T.dot(X.dot(weights) - y)
        weights -= learning_rate * gradients
        current_loss = np.mean((X.dot(weights) - y) ** 2)

        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return weights

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-4
) -> np.ndarray:
    """Perform Newton's method optimization."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)

    while True:
        residuals = y - X.dot(weights)
        gradient = -2 * X.T.dot(residuals) / n_samples
        hessian = 2 * X.T.dot(X) / n_samples

        if np.linalg.cond(hessian) < 1e15:
            delta = np.linalg.solve(hessian, gradient)
        else:
            delta = np.linalg.pinv(hessian).dot(gradient)

        weights += delta

        if np.linalg.norm(delta) < tol:
            break

    return weights

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    n_iter: int = 1000
) -> np.ndarray:
    """Perform coordinate descent optimization."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)

    for _ in range(n_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - (X.dot(weights) - weights[j] * X_j)
            weights[j] = np.sum(X_j * residuals) / np.sum(X_j ** 2)

    return weights

def _standard_normalizer(X: np.ndarray) -> np.ndarray:
    """Standard normalizer (z-score)."""
    return (X - X.mean(axis=0)) / X.std(axis=0)

def _minmax_normalizer(X: np.ndarray) -> np.ndarray:
    """Min-max normalizer."""
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

def _robust_normalizer(X: np.ndarray) -> np.ndarray:
    """Robust normalizer (using median and IQR)."""
    medians = np.median(X, axis=0)
    iqrs = np.subtract(*np.percentile(X, [75, 25], axis=0))
    return (X - medians) / iqrs

def algorithmes_gradient_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'gradient_descent',
    **solver_kwargs
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Perform stochastic gradient optimization.

    Parameters:
    - X: Input features (n_samples, n_features)
    - y: Target values (n_samples,)
    - normalizer: Optional normalizer function
    - metric: Evaluation metric ('mse', 'mae', 'r2') or callable
    - solver: Optimization algorithm ('gradient_descent', 'newton', 'coordinate_descent')
    - solver_kwargs: Additional arguments for the selected solver

    Returns:
    Dictionary containing:
    - result: Optimized parameters
    - metrics: Computed evaluation metrics
    - params_used: Parameters used in the optimization
    - warnings: Any warnings generated during execution

    Example:
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = algorithmes_gradient_fit(X, y, normalizer=_standard_normalizer,
    ...                                  metric='r2', solver='newton')
    """
    _validate_inputs(X, y, normalizer)

    # Apply normalization if specified
    X_normalized = normalizer(X) if normalizer else X

    # Select and run the appropriate solver
    if solver == 'gradient_descent':
        weights = _gradient_descent(X_normalized, y, **solver_kwargs)
    elif solver == 'newton':
        weights = _newton_method(X_normalized, y, **solver_kwargs)
    elif solver == 'coordinate_descent':
        weights = _coordinate_descent(X_normalized, y, **solver_kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute predictions and metrics
    y_pred = X_normalized.dot(weights)
    main_metric = _compute_metric(y, y_pred, metric)

    # Prepare the output dictionary
    result = {
        'result': weights,
        'metrics': {'main_metric': main_metric},
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'solver': solver,
            **solver_kwargs
        },
        'warnings': []
    }

    return result

################################################################################
# recuit_simule
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def recuit_simule_fit(
    objective_function: Callable[[np.ndarray], float],
    initial_solution: np.ndarray,
    temperature: float = 1000.0,
    cooling_rate: float = 0.99,
    max_iterations: int = 1000,
    neighborhood_function: Callable[[np.ndarray], np.ndarray] = None,
    acceptance_criteria: Callable[[float, float, float], bool] = None,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform simulated annealing optimization.

    Parameters:
    -----------
    objective_function : callable
        Function to minimize. Takes a solution vector and returns a scalar value.
    initial_solution : np.ndarray
        Initial solution vector.
    temperature : float, optional
        Initial temperature (default: 1000.0).
    cooling_rate : float, optional
        Rate at which temperature decreases (default: 0.99).
    max_iterations : int, optional
        Maximum number of iterations (default: 1000).
    neighborhood_function : callable, optional
        Function to generate a neighboring solution. Takes a solution vector and returns a modified version.
    acceptance_criteria : callable, optional
        Function to determine if a new solution should be accepted. Takes current energy, new energy, and temperature.
    random_state : int, optional
        Seed for random number generation (default: None).

    Returns:
    --------
    dict
        Dictionary containing the best solution found, final metrics, parameters used, and any warnings.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Default neighborhood function: small random perturbation
    if neighborhood_function is None:
        def neighborhood_function(solution):
            return solution + np.random.normal(0, 1, size=solution.shape)

    # Default acceptance criteria: Metropolis criterion
    if acceptance_criteria is None:
        def acceptance_criteria(current_energy, new_energy, temperature):
            if new_energy < current_energy:
                return True
            else:
                delta = np.exp((current_energy - new_energy) / temperature)
                return np.random.rand() < delta

    current_solution = initial_solution.copy()
    best_solution = current_solution.copy()
    current_energy = objective_function(current_solution)
    best_energy = current_energy

    results = {
        "result": None,
        "metrics": {},
        "params_used": {
            "temperature": temperature,
            "cooling_rate": cooling_rate,
            "max_iterations": max_iterations
        },
        "warnings": []
    }

    for iteration in range(max_iterations):
        temperature *= cooling_rate
        new_solution = neighborhood_function(current_solution)
        new_energy = objective_function(new_solution)

        if acceptance_criteria(current_energy, new_energy, temperature):
            current_solution = new_solution
            current_energy = new_energy

            if new_energy < best_energy:
                best_solution = new_solution
                best_energy = new_energy

    results["result"] = {
        "solution": best_solution,
        "energy": best_energy
    }
    results["metrics"]["final_energy"] = best_energy

    return results

def validate_objective_function(objective_function: Callable[[np.ndarray], float]) -> None:
    """
    Validate the objective function.

    Parameters:
    -----------
    objective_function : callable
        Function to validate.
    """
    test_input = np.array([1.0, 2.0])
    try:
        result = objective_function(test_input)
        if not isinstance(result, (int, float)):
            raise ValueError("Objective function must return a scalar value.")
    except Exception as e:
        raise ValueError(f"Objective function validation failed: {str(e)}")

def validate_neighborhood_function(neighborhood_function: Callable[[np.ndarray], np.ndarray]) -> None:
    """
    Validate the neighborhood function.

    Parameters:
    -----------
    neighborhood_function : callable
        Function to validate.
    """
    test_input = np.array([1.0, 2.0])
    try:
        result = neighborhood_function(test_input)
        if not isinstance(result, np.ndarray):
            raise ValueError("Neighborhood function must return a numpy array.")
        if result.shape != test_input.shape:
            raise ValueError("Neighborhood function must return an array of the same shape as input.")
    except Exception as e:
        raise ValueError(f"Neighborhood function validation failed: {str(e)}")

def validate_acceptance_criteria(acceptance_criteria: Callable[[float, float, float], bool]) -> None:
    """
    Validate the acceptance criteria function.

    Parameters:
    -----------
    acceptance_criteria : callable
        Function to validate.
    """
    test_input = (1.0, 2.0, 3.0)
    try:
        result = acceptance_criteria(*test_input)
        if not isinstance(result, bool):
            raise ValueError("Acceptance criteria function must return a boolean value.")
    except Exception as e:
        raise ValueError(f"Acceptance criteria validation failed: {str(e)}")

# Example usage:
if __name__ == "__main__":
    def example_objective_function(x):
        return np.sum(x**2)

    initial_solution = np.array([5.0, 3.0])
    result = recuit_simule_fit(
        objective_function=example_objective_function,
        initial_solution=initial_solution
    )
    print(result)

################################################################################
# optimisation_bayesienne
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def optimisation_bayesienne_fit(
    objective_function: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_iter: int = 100,
    acquisition_function: Callable[[np.ndarray, np.ndarray, np.ndarray], float] = None,
    kernel_function: Callable[[np.ndarray, np.ndarray], float] = None,
    normalizer: str = 'standard',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Optimisation bayésienne pour minimiser une fonction objectif.

    Parameters:
    -----------
    objective_function : Callable[[np.ndarray], float]
        Fonction objectif à minimiser.
    bounds : np.ndarray
        Bornes des paramètres sous forme [min, max] pour chaque dimension.
    n_iter : int, optional
        Nombre d'itérations (default: 100).
    acquisition_function : Callable, optional
        Fonction d'acquisition (default: None).
    kernel_function : Callable, optional
        Fonction noyau pour le processus gaussien (default: None).
    normalizer : str, optional
        Méthode de normalisation ('none', 'standard', 'minmax', 'robust') (default: 'standard').
    metric : Union[str, Callable], optional
        Métrique d'évaluation ('mse', 'mae', 'r2', 'logloss') ou callable (default: 'mse').
    solver : str, optional
        Solveur ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent') (default: 'closed_form').
    regularization : Optional[str], optional
        Type de régularisation ('none', 'l1', 'l2', 'elasticnet') (default: None).
    tol : float, optional
        Tolérance pour l'arrêt (default: 1e-4).
    random_state : Optional[int], optional
        Graine aléatoire (default: None).

    Returns:
    --------
    Dict[str, Any]
        Dictionnaire contenant les résultats, métriques, paramètres utilisés et avertissements.
    """
    # Initialisation des composants
    if random_state is not None:
        np.random.seed(random_state)

    # Validation des entrées
    _validate_inputs(objective_function, bounds, n_iter, acquisition_function,
                     kernel_function, normalizer, metric, solver, regularization)

    # Normalisation des données
    normalized_bounds = _apply_normalization(bounds, normalizer)

    # Initialisation des paramètres
    initial_params = _initialize_parameters(normalized_bounds, random_state)

    # Optimisation
    best_params, history = _optimize(
        objective_function,
        normalized_bounds,
        initial_params,
        n_iter,
        acquisition_function,
        kernel_function,
        metric,
        solver,
        regularization,
        tol
    )

    # Calcul des métriques
    metrics = _compute_metrics(history, metric)

    # Retour des résultats
    return {
        'result': best_params,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer,
            'metric': metric,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

def _validate_inputs(
    objective_function: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_iter: int,
    acquisition_function: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    kernel_function: Callable[[np.ndarray, np.ndarray], float],
    normalizer: str,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    solver: str,
    regularization: Optional[str]
) -> None:
    """
    Validation des entrées pour l'optimisation bayésienne.
    """
    if not callable(objective_function):
        raise ValueError("objective_function must be a callable.")
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must be a 2D array with shape (n_params, 2).")
    if n_iter <= 0:
        raise ValueError("n_iter must be a positive integer.")
    if acquisition_function is not None and not callable(acquisition_function):
        raise ValueError("acquisition_function must be a callable or None.")
    if kernel_function is not None and not callable(kernel_function):
        raise ValueError("kernel_function must be a callable or None.")
    if normalizer not in ['none', 'standard', 'minmax', 'robust']:
        raise ValueError("normalizer must be one of: 'none', 'standard', 'minmax', 'robust'.")
    if isinstance(metric, str) and metric not in ['mse', 'mae', 'r2', 'logloss']:
        raise ValueError("metric must be one of: 'mse', 'mae', 'r2', 'logloss' or a callable.")
    if solver not in ['closed_form', 'gradient_descent', 'newton', 'coordinate_descent']:
        raise ValueError("solver must be one of: 'closed_form', 'gradient_descent', 'newton', 'coordinate_descent'.")
    if regularization is not None and regularization not in ['none', 'l1', 'l2', 'elasticnet']:
        raise ValueError("regularization must be one of: 'none', 'l1', 'l2', 'elasticnet'.")

def _apply_normalization(
    bounds: np.ndarray,
    normalizer: str
) -> np.ndarray:
    """
    Applique la normalisation spécifiée aux bornes.
    """
    if normalizer == 'none':
        return bounds
    elif normalizer == 'standard':
        mean = np.mean(bounds, axis=1)
        std = np.std(bounds, axis=1)
        return (bounds - mean[:, np.newaxis]) / std[:, np.newaxis]
    elif normalizer == 'minmax':
        min_val = np.min(bounds, axis=1)
        max_val = np.max(bounds, axis=1)
        return (bounds - min_val[:, np.newaxis]) / (max_val - min_val)[:, np.newaxis]
    elif normalizer == 'robust':
        median = np.median(bounds, axis=1)
        iqr = np.subtract(*np.percentile(bounds, [75, 25], axis=1))
        return (bounds - median[:, np.newaxis]) / iqr[:, np.newaxis]
    else:
        raise ValueError("Invalid normalizer specified.")

def _initialize_parameters(
    bounds: np.ndarray,
    random_state: Optional[int]
) -> np.ndarray:
    """
    Initialise les paramètres aléatoires dans les bornes spécifiées.
    """
    if random_state is not None:
        np.random.seed(random_state)
    return np.random.uniform(bounds[:, 0], bounds[:, 1])

def _optimize(
    objective_function: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    initial_params: np.ndarray,
    n_iter: int,
    acquisition_function: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    kernel_function: Callable[[np.ndarray, np.ndarray], float],
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    solver: str,
    regularization: Optional[str],
    tol: float
) -> tuple[np.ndarray, list]:
    """
    Effectue l'optimisation bayésienne.
    """
    history = []
    current_params = initial_params.copy()

    for _ in range(n_iter):
        # Évaluation de la fonction objectif
        current_value = objective_function(current_params)

        # Mise à jour de l'historique
        history.append({'params': current_params.copy(), 'value': current_value})

        # Calcul de la prochaine étape
        next_params = _compute_next_step(
            objective_function,
            bounds,
            current_params,
            acquisition_function,
            kernel_function,
            solver
        )

        # Vérification de la convergence
        if np.linalg.norm(next_params - current_params) < tol:
            break

        current_params = next_params

    return current_params, history

def _compute_next_step(
    objective_function: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    current_params: np.ndarray,
    acquisition_function: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    kernel_function: Callable[[np.ndarray, np.ndarray], float],
    solver: str
) -> np.ndarray:
    """
    Calcule les prochains paramètres à évaluer.
    """
    if solver == 'closed_form':
        return _closed_form_solver(objective_function, bounds)
    elif solver == 'gradient_descent':
        return _gradient_descent_solver(objective_function, current_params)
    elif solver == 'newton':
        return _newton_solver(objective_function, current_params)
    elif solver == 'coordinate_descent':
        return _coordinate_descent_solver(objective_function, current_params)
    else:
        raise ValueError("Invalid solver specified.")

def _closed_form_solver(
    objective_function: Callable[[np.ndarray], float],
    bounds: np.ndarray
) -> np.ndarray:
    """
    Solveur en forme fermée.
    """
    # Implémentation simplifiée pour l'exemple
    return np.mean(bounds, axis=1)

def _gradient_descent_solver(
    objective_function: Callable[[np.ndarray], float],
    current_params: np.ndarray
) -> np.ndarray:
    """
    Solveur par descente de gradient.
    """
    # Implémentation simplifiée pour l'exemple
    learning_rate = 0.01
    gradient = _compute_gradient(objective_function, current_params)
    return current_params - learning_rate * gradient

def _newton_solver(
    objective_function: Callable[[np.ndarray], float],
    current_params: np.ndarray
) -> np.ndarray:
    """
    Solveur par méthode de Newton.
    """
    # Implémentation simplifiée pour l'exemple
    hessian = _compute_hessian(objective_function, current_params)
    gradient = _compute_gradient(objective_function, current_params)
    return current_params - np.linalg.solve(hessian, gradient)

def _coordinate_descent_solver(
    objective_function: Callable[[np.ndarray], float],
    current_params: np.ndarray
) -> np.ndarray:
    """
    Solveur par descente de coordonnées.
    """
    # Implémentation simplifiée pour l'exemple
    new_params = current_params.copy()
    for i in range(len(new_params)):
        # Optimisation le long de chaque coordonnée
        new_params[i] = _optimize_along_coordinate(objective_function, i, current_params)
    return new_params

def _compute_gradient(
    objective_function: Callable[[np.ndarray], float],
    params: np.ndarray
) -> np.ndarray:
    """
    Calcule le gradient de la fonction objectif.
    """
    epsilon = 1e-8
    gradient = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += epsilon
        params_minus = params.copy()
        params_minus[i] -= epsilon
        gradient[i] = (objective_function(params_plus) - objective_function(params_minus)) / (2 * epsilon)
    return gradient

def _compute_hessian(
    objective_function: Callable[[np.ndarray], float],
    params: np.ndarray
) -> np.ndarray:
    """
    Calcule la hessienne de la fonction objectif.
    """
    epsilon = 1e-8
    n_params = len(params)
    hessian = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(n_params):
            params_plus = params.copy()
            params_plus[i] += epsilon
            params_minus = params.copy()
            params_minus[i] -= epsilon

            gradient_plus = _compute_gradient(objective_function, params_plus)
            gradient_minus = _compute_gradient(objective_function, params_minus)

            hessian[i, j] = (gradient_plus[j] - gradient_minus[j]) / (2 * epsilon)
    return hessian

def _optimize_along_coordinate(
    objective_function: Callable[[np.ndarray], float],
    coordinate: int,
    params: np.ndarray
) -> float:
    """
    Optimise la fonction objectif le long d'une coordonnée.
    """
    # Implémentation simplifiée pour l'exemple
    return params[coordinate]

def _compute_metrics(
    history: list,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, Any]:
    """
    Calcule les métriques d'évaluation.
    """
    metrics = {}

    if isinstance(metric, str):
        if metric == 'mse':
            values = np.array([h['value'] for h in history])
            metrics['mse'] = np.mean(values**2)
        elif metric == 'mae':
            values = np.array([h['value'] for h in history])
            metrics['mae'] = np.mean(np.abs(values))
        elif metric == 'r2':
            # Implémentation simplifiée pour l'exemple
            metrics['r2'] = 1.0
        elif metric == 'logloss':
            # Implémentation simplifiée pour l'exemple
            metrics['logloss'] = 0.0
    else:
        # Utilisation d'une métrique personnalisée
        values = np.array([h['value'] for h in history])
        metrics['custom_metric'] = metric(values, values)

    return metrics

################################################################################
# algorithmes_evolutionnistes
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(
    population: np.ndarray,
    fitness_func: Callable[[np.ndarray], Union[float, np.ndarray]],
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.7,
    selection_size: int = 2,
    elitism_count: int = 1
) -> None:
    """
    Validate the inputs for evolutionary algorithms.

    Parameters
    ----------
    population : np.ndarray
        Initial population of candidates.
    fitness_func : callable
        Function to evaluate the fitness of each candidate.
    mutation_rate : float, optional
        Probability of mutation for each gene (default: 0.1).
    crossover_rate : float, optional
        Probability of crossover between parents (default: 0.7).
    selection_size : int, optional
        Number of candidates to select for reproduction (default: 2).
    elitism_count : int, optional
        Number of best candidates to carry over to the next generation (default: 1).

    Raises
    ------
    ValueError
        If any input is invalid.
    """
    if not isinstance(population, np.ndarray):
        raise ValueError("Population must be a numpy array.")
    if population.ndim != 2:
        raise ValueError("Population must be a 2D array.")
    if not callable(fitness_func):
        raise ValueError("Fitness function must be callable.")
    if not 0 <= mutation_rate <= 1:
        raise ValueError("Mutation rate must be between 0 and 1.")
    if not 0 <= crossover_rate <= 1:
        raise ValueError("Crossover rate must be between 0 and 1.")
    if selection_size < 2:
        raise ValueError("Selection size must be at least 2.")
    if elitism_count < 0:
        raise ValueError("Elitism count must be non-negative.")
    if elitism_count > population.shape[0]:
        raise ValueError("Elitism count must be less than or equal to population size.")

def _evaluate_fitness(
    population: np.ndarray,
    fitness_func: Callable[[np.ndarray], Union[float, np.ndarray]]
) -> np.ndarray:
    """
    Evaluate the fitness of each candidate in the population.

    Parameters
    ----------
    population : np.ndarray
        Population of candidates.
    fitness_func : callable
        Function to evaluate the fitness of each candidate.

    Returns
    -------
    np.ndarray
        Array of fitness values for each candidate.
    """
    return np.array([fitness_func(ind) for ind in population])

def _select_parents(
    population: np.ndarray,
    fitness: np.ndarray,
    selection_size: int = 2
) -> np.ndarray:
    """
    Select parents for reproduction using tournament selection.

    Parameters
    ----------
    population : np.ndarray
        Population of candidates.
    fitness : np.ndarray
        Fitness values for each candidate.
    selection_size : int, optional
        Number of candidates to select for reproduction (default: 2).

    Returns
    -------
    np.ndarray
        Selected parents.
    """
    indices = np.argsort(fitness)[-selection_size:]
    return population[indices]

def _crossover(
    parents: np.ndarray,
    crossover_rate: float = 0.7
) -> np.ndarray:
    """
    Perform crossover between parents to produce offspring.

    Parameters
    ----------
    parents : np.ndarray
        Array of parent candidates.
    crossover_rate : float, optional
        Probability of crossover between parents (default: 0.7).

    Returns
    -------
    np.ndarray
        Offspring produced by crossover.
    """
    if len(parents) < 2:
        return parents.copy()

    offspring = []
    for i in range(0, len(parents), 2):
        parent1, parent2 = parents[i], parents[i+1]
        if np.random.rand() < crossover_rate:
            crossover_point = np.random.randint(1, parent1.shape[0])
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            offspring.extend([child1, child2])
        else:
            offspring.extend([parent1.copy(), parent2.copy()])

    return np.array(offspring)

def _mutate(
    offspring: np.ndarray,
    mutation_rate: float = 0.1
) -> np.ndarray:
    """
    Mutate the offspring by randomly altering genes.

    Parameters
    ----------
    offspring : np.ndarray
        Offspring produced by crossover.
    mutation_rate : float, optional
        Probability of mutation for each gene (default: 0.1).

    Returns
    -------
    np.ndarray
        Mutated offspring.
    """
    for i in range(offspring.shape[0]):
        for j in range(offspring.shape[1]):
            if np.random.rand() < mutation_rate:
                offspring[i, j] += np.random.normal(0, 1)
    return offspring

def _apply_elitism(
    population: np.ndarray,
    fitness: np.ndarray,
    offspring: np.ndarray,
    elitism_count: int = 1
) -> np.ndarray:
    """
    Apply elitism by carrying over the best candidates to the next generation.

    Parameters
    ----------
    population : np.ndarray
        Current population of candidates.
    fitness : np.ndarray
        Fitness values for each candidate.
    offspring : np.ndarray
        Offspring produced by crossover and mutation.
    elitism_count : int, optional
        Number of best candidates to carry over (default: 1).

    Returns
    -------
    np.ndarray
        New population including elite candidates.
    """
    elite_indices = np.argsort(fitness)[:elitism_count]
    elite_candidates = population[elite_indices]

    if offspring.shape[0] + elite_candidates.shape[0] > population.shape[0]:
        return offspring[:population.shape[0] - elite_candidates.shape[0]]

    new_population = np.vstack([offspring, elite_candidates])
    return new_population

def algorithmes_evolutionnistes_fit(
    population: np.ndarray,
    fitness_func: Callable[[np.ndarray], Union[float, np.ndarray]],
    generations: int = 100,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.7,
    selection_size: int = 2,
    elitism_count: int = 1,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run evolutionary algorithms to optimize a given fitness function.

    Parameters
    ----------
    population : np.ndarray
        Initial population of candidates.
    fitness_func : callable
        Function to evaluate the fitness of each candidate.
    generations : int, optional
        Number of generations to run (default: 100).
    mutation_rate : float, optional
        Probability of mutation for each gene (default: 0.1).
    crossover_rate : float, optional
        Probability of crossover between parents (default: 0.7).
    selection_size : int, optional
        Number of candidates to select for reproduction (default: 2).
    elitism_count : int, optional
        Number of best candidates to carry over to the next generation (default: 1).
    verbose : bool, optional
        Whether to print progress information (default: False).

    Returns
    -------
    dict
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    _validate_inputs(population, fitness_func, mutation_rate, crossover_rate, selection_size, elitism_count)

    best_fitness_history = []
    avg_fitness_history = []

    for generation in range(generations):
        fitness = _evaluate_fitness(population, fitness_func)
        best_fitness = np.max(fitness)
        avg_fitness = np.mean(fitness)

        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)

        if verbose:
            print(f"Generation {generation + 1}: Best Fitness = {best_fitness}, Avg Fitness = {avg_fitness}")

        parents = _select_parents(population, fitness, selection_size)
        offspring = _crossover(parents, crossover_rate)
        offspring = _mutate(offspring, mutation_rate)
        population = _apply_elitism(population, fitness, offspring, elitism_count)

    final_fitness = _evaluate_fitness(population, fitness_func)
    best_candidate_idx = np.argmax(final_fitness)
    best_candidate = population[best_candidate_idx]

    return {
        "result": {
            "best_candidate": best_candidate,
            "best_fitness": final_fitness[best_candidate_idx]
        },
        "metrics": {
            "best_fitness_history": best_fitness_history,
            "avg_fitness_history": avg_fitness_history
        },
        "params_used": {
            "generations": generations,
            "mutation_rate": mutation_rate,
            "crossover_rate": crossover_rate,
            "selection_size": selection_size,
            "elitism_count": elitism_count
        },
        "warnings": []
    }

################################################################################
# recherche_harmonique
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

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
        _ = normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalizer function failed: {str(e)}")

    # Test metric
    try:
        _ = metric(y, y)
    except Exception as e:
        raise ValueError(f"Metric function failed: {str(e)}")

def _harmonic_search(
    X: np.ndarray,
    y: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    solver: str,
    max_iter: int = 1000,
    tol: float = 1e-4,
    **solver_kwargs
) -> Dict[str, Any]:
    """Perform harmonic search optimization."""
    n_samples, n_features = X.shape

    # Initialize parameters
    params = np.zeros(n_features)

    for _ in range(max_iter):
        # Update parameters based on solver
        if solver == 'gradient_descent':
            grad = _compute_gradient(X, y, params)
            params -= solver_kwargs.get('learning_rate', 0.01) * grad
        elif solver == 'newton':
            hess = _compute_hessian(X, y, params)
            grad = _compute_gradient(X, y, params)
            params -= np.linalg.solve(hess, grad)
        elif solver == 'coordinate_descent':
            for i in range(n_features):
                params[i] = _coordinate_descent_step(X, y, params, i)
        else:
            raise ValueError(f"Unknown solver: {solver}")

        # Check convergence
        if _check_convergence(params, tol):
            break

    return {'params': params}

def _compute_gradient(X: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Compute gradient of the objective function."""
    residuals = y - X @ params
    return -(2/n) * X.T @ residuals

def _compute_hessian(X: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Compute Hessian matrix of the objective function."""
    return (2/n) * X.T @ X

def _coordinate_descent_step(X: np.ndarray, y: np.ndarray, params: np.ndarray, i: int) -> float:
    """Perform a single coordinate descent step."""
    X_i = X[:, i]
    residuals = y - (X @ params - X_i * params[i])
    numerator = X_i.T @ residuals
    denominator = X_i.T @ X_i
    return numerator / denominator

def _check_convergence(params: np.ndarray, tol: float) -> bool:
    """Check if parameters have converged."""
    return np.linalg.norm(params) < tol

def recherche_harmonique_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: Callable[[np.ndarray, np.ndarray], float] = lambda y_true, y_pred: np.mean((y_true - y_pred)**2),
    solver: str = 'gradient_descent',
    max_iter: int = 1000,
    tol: float = 1e-4,
    **solver_kwargs
) -> Dict[str, Any]:
    """
    Perform harmonic search optimization.

    Parameters:
    -----------
    X : np.ndarray
        Input features (n_samples, n_features)
    y : np.ndarray
        Target values (n_samples,)
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize input data
    metric : Callable[[np.ndarray, np.ndarray], float]
        Function to evaluate model performance
    solver : str
        Optimization algorithm ('gradient_descent', 'newton', 'coordinate_descent')
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for convergence
    **solver_kwargs :
        Additional solver-specific parameters

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate inputs
    _validate_inputs(X, y, normalizer, metric)

    # Normalize data
    X_norm = normalizer(X)
    y_norm = normalizer(y.reshape(-1, 1)).flatten()

    # Perform harmonic search
    results = _harmonic_search(X_norm, y_norm, metric, solver, max_iter, tol, **solver_kwargs)

    # Compute metrics
    y_pred = X_norm @ results['params']
    metrics = {
        'metric': metric(y_norm, y_pred),
        'r2': 1 - np.sum((y_norm - y_pred)**2) / np.sum((y_norm - np.mean(y_norm))**2)
    }

    return {
        'result': results['params'],
        'metrics': metrics,
        'params_used': {
            'solver': solver,
            'max_iter': max_iter,
            'tol': tol,
            **solver_kwargs
        },
        'warnings': []
    }

# Example usage:
"""
X = np.random.rand(100, 5)
y = X @ np.array([1.2, -0.8, 0.5, 1.0, -0.3]) + np.random.normal(0, 0.1, 100)

result = recherche_harmonique_fit(
    X,
    y,
    normalizer=lambda x: (x - np.mean(x, axis=0)) / np.std(x, axis=0),
    solver='gradient_descent',
    learning_rate=0.01
)
"""

################################################################################
# optimisation_particules_essaim
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def optimisation_particules_essaim_fit(
    objective_function: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_particles: int = 30,
    max_iter: int = 100,
    w: float = 0.5,
    c1: float = 1.0,
    c2: float = 1.0,
    tol: float = 1e-6,
    verbose: bool = False
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Optimisation par essaim de particules.

    Parameters
    ----------
    objective_function : Callable[[np.ndarray], float]
        Fonction objectif à minimiser.
    bounds : np.ndarray
        Matrice de dimensions (n_dimensions, 2) contenant les bornes inférieures et supérieures.
    n_particles : int, optional
        Nombre de particules dans l'essaim (default: 30).
    max_iter : int, optional
        Nombre maximal d'itérations (default: 100).
    w : float, optional
        Poids d'inertie (default: 0.5).
    c1 : float, optional
        Coefficient cognitif (default: 1.0).
    c2 : float, optional
        Coefficient social (default: 1.0).
    tol : float, optional
        Tolérance pour l'arrêt (default: 1e-6).
    verbose : bool, optional
        Afficher les logs d'optimisation (default: False).

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionnaire contenant les résultats de l'optimisation.
    """
    # Validation des entrées
    _validate_inputs(objective_function, bounds, n_particles, max_iter)

    # Initialisation des particules
    particles = _initialize_particles(bounds, n_particles)
    velocities = np.zeros_like(particles)

    # Initialisation des meilleures positions
    personal_best_positions = particles.copy()
    personal_best_scores = np.array([objective_function(p) for p in particles])

    # Meilleure position globale
    global_best_index = np.argmin(personal_best_scores)
    global_best_position = personal_best_positions[global_best_index].copy()
    global_best_score = personal_best_scores[global_best_index]

    # Optimisation
    for iteration in range(max_iter):
        r1, r2 = np.random.rand(2)

        # Mise à jour des vitesses et positions
        velocities = w * velocities + c1 * r1 * (personal_best_positions - particles) + c2 * r2 * (global_best_position - particles)
        particles = np.clip(particles + velocities, bounds[:, 0], bounds[:, 1])

        # Évaluation des nouvelles positions
        current_scores = np.array([objective_function(p) for p in particles])

        # Mise à jour des meilleures positions personnelles
        improved_indices = current_scores < personal_best_scores
        personal_best_positions[improved_indices] = particles[improved_indices]
        personal_best_scores[improved_indices] = current_scores[improved_indices]

        # Mise à jour de la meilleure position globale
        current_best_index = np.argmin(current_scores)
        if current_scores[current_best_index] < global_best_score:
            global_best_position = particles[current_best_index].copy()
            global_best_score = current_scores[current_best_index]

        # Critère d'arrêt
        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}: Best score = {global_best_score}")

        if np.abs(global_best_score - personal_best_scores.min()) < tol:
            break

    # Résultats
    result = {
        "result": global_best_position,
        "metrics": {"best_score": global_best_score},
        "params_used": {
            "n_particles": n_particles,
            "max_iter": iteration + 1,
            "w": w,
            "c1": c1,
            "c2": c2
        },
        "warnings": []
    }

    return result

def _validate_inputs(
    objective_function: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_particles: int,
    max_iter: int
) -> None:
    """Validation des entrées."""
    if not callable(objective_function):
        raise ValueError("objective_function must be a callable.")
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must be a 2D array of shape (n_dimensions, 2).")
    if n_particles <= 0:
        raise ValueError("n_particles must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

def _initialize_particles(
    bounds: np.ndarray,
    n_particles: int
) -> np.ndarray:
    """Initialisation des particules."""
    dimensions = bounds.shape[0]
    return np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(n_particles, dimensions))

################################################################################
# algorithmes_colonie_fourmis
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(X: np.ndarray,
                    y: Optional[np.ndarray] = None,
                    distance_metric: str = 'euclidean',
                    custom_distance: Optional[Callable] = None) -> None:
    """
    Validate input data and parameters for ant colony optimization.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values if available
    distance_metric : str
        Distance metric to use
    custom_distance : Optional[Callable]
        Custom distance function if provided

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X must be a 2D numpy array")
    if y is not None and (not isinstance(y, np.ndarray) or len(y.shape) != 1):
        raise ValueError("y must be a 1D numpy array if provided")
    if y is not None and X.shape[0] != len(y):
        raise ValueError("X and y must have the same number of samples")

    if custom_distance is not None:
        return

    valid_metrics = ['euclidean', 'manhattan', 'cosine', 'minkowski']
    if distance_metric not in valid_metrics:
        raise ValueError(f"distance_metric must be one of {valid_metrics}")

def _calculate_distance(X: np.ndarray,
                       distance_metric: str = 'euclidean',
                       p: float = 2.0,
                       custom_distance: Optional[Callable] = None) -> np.ndarray:
    """
    Calculate distance matrix between samples.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix of shape (n_samples, n_features)
    distance_metric : str
        Distance metric to use
    p : float
        Power parameter for Minkowski distance (default=2.0)
    custom_distance : Optional[Callable]
        Custom distance function if provided

    Returns
    ------
    np.ndarray
        Distance matrix of shape (n_samples, n_samples)
    """
    if custom_distance is not None:
        return np.array([[custom_distance(x1, x2) for x2 in X] for x1 in X])

    n_samples = X.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    if distance_metric == 'euclidean':
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance = np.linalg.norm(X[i] - X[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

    elif distance_metric == 'manhattan':
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance = np.sum(np.abs(X[i] - X[j]))
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

    elif distance_metric == 'cosine':
        for i in range(n_samples):
            for j in range(i, n_samples):
                dot_product = np.dot(X[i], X[j])
                norm_i = np.linalg.norm(X[i])
                norm_j = np.linalg.norm(X[j])
                distance = 1 - (dot_product / (norm_i * norm_j))
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

    elif distance_metric == 'minkowski':
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance = np.sum(np.abs(X[i] - X[j])**p)**(1/p)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

    return distance_matrix

def _update_pheromone(trails: np.ndarray,
                     distance_matrix: np.ndarray,
                     evaporation_rate: float = 0.5,
                     q: float = 1.0) -> np.ndarray:
    """
    Update pheromone trails based on distance matrix.

    Parameters
    ----------
    trails : np.ndarray
        Current pheromone trail matrix
    distance_matrix : np.ndarray
        Distance matrix between samples
    evaporation_rate : float
        Pheromone evaporation rate (default=0.5)
    q : float
        Pheromone deposit constant (default=1.0)

    Returns
    ------
    np.ndarray
        Updated pheromone trail matrix
    """
    # Evaporation
    trails = trails * (1 - evaporation_rate)

    # Deposit new pheromone
    for i in range(distance_matrix.shape[0]):
        for j in range(i+1, distance_matrix.shape[1]):
            trails[i, j] += q / distance_matrix[i, j]
            trails[j, i] = trails[i, j]

    return trails

def _select_path(pheromone: np.ndarray,
                 distance_matrix: np.ndarray,
                 alpha: float = 1.0,
                 beta: float = 2.0) -> np.ndarray:
    """
    Select paths based on pheromone and distance.

    Parameters
    ----------
    pheromone : np.ndarray
        Pheromone trail matrix
    distance_matrix : np.ndarray
        Distance matrix between samples
    alpha : float
        Pheromone importance (default=1.0)
    beta : float
        Distance importance (default=2.0)

    Returns
    ------
    np.ndarray
        Probability matrix for path selection
    """
    n = pheromone.shape[0]
    probabilities = np.zeros((n, n))

    for i in range(n):
        # Calculate denominator
        denominator = np.sum(pheromone[i] ** alpha * (1 / distance_matrix[i]) ** beta)

        # Calculate probabilities
        for j in range(n):
            if i != j:
                probabilities[i, j] = (pheromone[i, j] ** alpha * (1 / distance_matrix[i, j]) ** beta) / denominator

    return probabilities

def _calculate_metrics(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      metric: str = 'mse',
                      custom_metric: Optional[Callable] = None) -> Dict[str, float]:
    """
    Calculate optimization metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
    metric : str
        Metric to calculate (default='mse')
    custom_metric : Optional[Callable]
        Custom metric function if provided

    Returns
    ------
    Dict[str, float]
        Dictionary of calculated metrics
    """
    results = {}

    if custom_metric is not None:
        results['custom'] = custom_metric(y_true, y_pred)
        return results

    if metric == 'mse':
        mse = np.mean((y_true - y_pred) ** 2)
        results['mse'] = mse

    elif metric == 'mae':
        mae = np.mean(np.abs(y_true - y_pred))
        results['mae'] = mae

    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        results['r2'] = r2

    elif metric == 'logloss':
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        logloss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        results['logloss'] = logloss

    return results

def algorithmes_colonie_fourmis_fit(X: np.ndarray,
                                   y: Optional[np.ndarray] = None,
                                   n_ants: int = 10,
                                   n_iterations: int = 100,
                                   distance_metric: str = 'euclidean',
                                   p: float = 2.0,
                                   alpha: float = 1.0,
                                   beta: float = 2.0,
                                   evaporation_rate: float = 0.5,
                                   q: float = 1.0,
                                   metric: str = 'mse',
                                   custom_metric: Optional[Callable] = None,
                                   custom_distance: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Ant Colony Optimization algorithm for optimization problems.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values if available
    n_ants : int
        Number of ants in the colony (default=10)
    n_iterations : int
        Number of iterations to run (default=100)
    distance_metric : str
        Distance metric to use (default='euclidean')
    p : float
        Power parameter for Minkowski distance (default=2.0)
    alpha : float
        Pheromone importance (default=1.0)
    beta : float
        Distance importance (default=2.0)
    evaporation_rate : float
        Pheromone evaporation rate (default=0.5)
    q : float
        Pheromone deposit constant (default=1.0)
    metric : str
        Metric to optimize (default='mse')
    custom_metric : Optional[Callable]
        Custom metric function if provided
    custom_distance : Optional[Callable]
        Custom distance function if provided

    Returns
    ------
    Dict[str, Any]
        Dictionary containing:
        - 'result': Optimization result
        - 'metrics': Calculated metrics
        - 'params_used': Parameters used
        - 'warnings': Any warnings generated

    Example
    -------
    >>> X = np.random.rand(10, 5)
    >>> y = np.random.rand(10)
    >>> result = algorithmes_colonie_fourmis_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y, distance_metric, custom_distance)

    # Initialize parameters
    n_samples = X.shape[0]
    trails = np.ones((n_samples, n_samples))
    best_solution = None
    best_metric = float('inf') if metric != 'r2' else float('-inf')

    # Calculate distance matrix
    distance_matrix = _calculate_distance(X, distance_metric, p, custom_distance)

    # Main optimization loop
    for _ in range(n_iterations):
        # Update pheromone trails
        trails = _update_pheromone(trails, distance_matrix, evaporation_rate, q)

        # Select paths based on pheromone and distance
        probabilities = _select_path(trails, distance_matrix, alpha, beta)

        # Here you would implement the actual path selection and solution construction
        # For this example, we'll just use a placeholder

        # Calculate metrics for the current solution
        if y is not None:
            # In a real implementation, you would calculate y_pred based on the current solution
            y_pred = np.random.rand(len(y))  # Placeholder

            metrics = _calculate_metrics(y, y_pred, metric, custom_metric)

            current_metric = next(iter(metrics.values()))
            if (metric == 'r2' and current_metric > best_metric) or \
               (metric != 'r2' and current_metric < best_metric):
                best_metric = current_metric
                best_solution = y_pred.copy()
        else:
            metrics = {}

    # Prepare results
    result_dict = {
        'result': best_solution,
        'metrics': metrics,
        'params_used': {
            'n_ants': n_ants,
            'n_iterations': n_iterations,
            'distance_metric': distance_metric,
            'p': p,
            'alpha': alpha,
            'beta': beta,
            'evaporation_rate': evaporation_rate,
            'q': q,
            'metric': metric
        },
        'warnings': []
    }

    return result_dict

################################################################################
# optimisation_swarm_intelligence
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def optimisation_swarm_intelligence_fit(
    objective_function: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_particles: int = 30,
    max_iter: int = 100,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    normalization: str = 'none',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    tolerance: float = 1e-6,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Optimisation par essaim de particules.

    Parameters:
    -----------
    objective_function : Callable[[np.ndarray], float]
        Fonction objectif à minimiser.
    bounds : np.ndarray
        Matrice de bornes [min, max] pour chaque dimension.
    n_particles : int, optional
        Nombre de particules dans l'essaim (default: 30).
    max_iter : int, optional
        Nombre maximal d'itérations (default: 100).
    w : float, optional
        Poids d'inertie (default: 0.7).
    c1 : float, optional
        Coefficient cognitif (default: 1.5).
    c2 : float, optional
        Coefficient social (default: 1.5).
    normalization : str, optional
        Type de normalisation ('none', 'standard', 'minmax', 'robust') (default: 'none').
    metric : Union[str, Callable], optional
        Métrique d'évaluation ('mse', 'mae', 'r2', 'logloss') ou callable (default: 'mse').
    distance : Union[str, Callable], optional
        Distance utilisée ('euclidean', 'manhattan', 'cosine', 'minkowski') ou callable (default: 'euclidean').
    tolerance : float, optional
        Tolérance pour l'arrêt (default: 1e-6).
    verbose : bool, optional
        Afficher les logs (default: False).

    Returns:
    --------
    Dict[str, Any]
        Dictionnaire contenant les résultats, métriques, paramètres utilisés et avertissements.
    """
    # Validation des entrées
    _validate_inputs(objective_function, bounds, n_particles, max_iter)

    # Initialisation des particules
    particles = _initialize_particles(bounds, n_particles)
    velocities = np.zeros_like(particles)

    # Meilleure position personnelle et globale
    personal_best_positions = particles.copy()
    personal_best_scores = np.array([objective_function(p) for p in particles])
    global_best_index = np.argmin(personal_best_scores)
    global_best_position = personal_best_positions[global_best_index]
    global_best_score = personal_best_scores[global_best_index]

    # Normalisation
    if normalization != 'none':
        particles, velocities = _apply_normalization(particles, velocities, normalization)

    # Optimisation
    for iteration in range(max_iter):
        r1, r2 = np.random.rand(2)
        for i in range(n_particles):
            # Mise à jour de la vitesse
            velocities[i] = (w * velocities[i] +
                            c1 * r1 * (personal_best_positions[i] - particles[i]) +
                            c2 * r2 * (global_best_position - particles[i]))

            # Mise à jour de la position
            particles[i] += velocities[i]

            # Bornes
            particles[i] = np.clip(particles[i], bounds[:, 0], bounds[:, 1])

            # Évaluation
            current_score = objective_function(particles[i])

            # Mise à jour des meilleures positions
            if current_score < personal_best_scores[i]:
                personal_best_positions[i] = particles[i]
                personal_best_scores[i] = current_score

                if current_score < global_best_score:
                    global_best_position = particles[i]
                    global_best_score = current_score

        # Critère d'arrêt
        if np.abs(global_best_score - personal_best_scores[global_best_index]) < tolerance:
            break

        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}, Best Score: {global_best_score}")

    # Calcul des métriques
    metrics = _compute_metrics(global_best_position, metric)

    return {
        "result": global_best_position,
        "metrics": metrics,
        "params_used": {
            "n_particles": n_particles,
            "max_iter": max_iter,
            "w": w,
            "c1": c1,
            "c2": c2,
            "normalization": normalization,
            "metric": metric,
            "distance": distance
        },
        "warnings": []
    }

def _validate_inputs(
    objective_function: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_particles: int,
    max_iter: int
) -> None:
    """Validation des entrées."""
    if not callable(objective_function):
        raise ValueError("objective_function must be a callable.")
    if bounds.shape[1] != 2:
        raise ValueError("bounds must be a 2D array with shape (n_features, 2).")
    if n_particles <= 0:
        raise ValueError("n_particles must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

def _initialize_particles(
    bounds: np.ndarray,
    n_particles: int
) -> np.ndarray:
    """Initialisation des particules."""
    return np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_particles, bounds.shape[0]))

def _apply_normalization(
    particles: np.ndarray,
    velocities: np.ndarray,
    normalization: str
) -> tuple[np.ndarray, np.ndarray]:
    """Application de la normalisation."""
    if normalization == 'standard':
        mean = np.mean(particles, axis=0)
        std = np.std(particles, axis=0)
        particles = (particles - mean) / std
    elif normalization == 'minmax':
        min_val = np.min(particles, axis=0)
        max_val = np.max(particles, axis=0)
        particles = (particles - min_val) / (max_val - min_val)
    elif normalization == 'robust':
        median = np.median(particles, axis=0)
        iqr = np.subtract(*np.percentile(particles, [75, 25], axis=0))
        particles = (particles - median) / iqr
    return particles, velocities

def _compute_metrics(
    position: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Calcul des métriques."""
    metrics = {}
    if callable(metric):
        # Utiliser la métrique personnalisée
        pass  # Implémentation spécifique nécessaire
    elif metric == 'mse':
        metrics['mse'] = 0.0  # Implémentation spécifique nécessaire
    elif metric == 'mae':
        metrics['mae'] = 0.0  # Implémentation spécifique nécessaire
    elif metric == 'r2':
        metrics['r2'] = 0.0   # Implémentation spécifique nécessaire
    elif metric == 'logloss':
        metrics['logloss'] = 0.0  # Implémentation spécifique nécessaire
    return metrics

# Exemple minimal :
# def objective_function(x):
#     return np.sum(x**2)
#
# bounds = np.array([[0, 1], [0, 1]])
# result = optimisation_swarm_intelligence_fit(objective_function, bounds)

################################################################################
# algorithmes_renforcement
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> None:
    """Validate input data and normalizer."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or inf values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or inf values")
    if normalizer is not None:
        try:
            X_normalized = normalizer(X)
        except Exception as e:
            raise ValueError(f"Normalizer failed: {str(e)}")

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute the specified metric between true and predicted values."""
    metrics = {}
    if isinstance(metric, str):
        if metric == "mse":
            metrics["mse"] = np.mean((y_true - y_pred) ** 2)
        elif metric == "mae":
            metrics["mae"] = np.mean(np.abs(y_true - y_pred))
        elif metric == "r2":
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics["r2"] = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        elif metric == "logloss":
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            metrics["logloss"] = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        metrics["custom"] = metric(y_true, y_pred)
    else:
        raise ValueError("Metric must be a string or callable")
    return metrics

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    n_iter: int = 1000,
    tol: float = 1e-4
) -> np.ndarray:
    """Perform gradient descent optimization."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    for _ in range(n_iter):
        gradients = 2 / n_samples * X.T @ (X @ weights - y)
        new_weights = weights - learning_rate * gradients
        if np.linalg.norm(new_weights - weights) < tol:
            break
        weights = new_weights
    return weights

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-4
) -> np.ndarray:
    """Perform Newton's method optimization."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    while True:
        residuals = X @ weights - y
        gradient = 2 / n_samples * X.T @ residuals
        hessian = 2 / n_samples * X.T @ X
        if np.linalg.norm(gradient) < tol:
            break
        weights = weights - np.linalg.solve(hessian, gradient)
    return weights

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    n_iter: int = 1000
) -> np.ndarray:
    """Perform coordinate descent optimization."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    for _ in range(n_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - (X @ weights - X_j * weights[j])
            weights[j] = np.sum(X_j * residuals) / np.sum(X_j ** 2)
    return weights

def _apply_regularization(
    weights: np.ndarray,
    reg_type: str = "none",
    alpha: float = 1.0
) -> np.ndarray:
    """Apply regularization to the weights."""
    if reg_type == "l1":
        return np.sign(weights) * np.maximum(np.abs(weights) - alpha, 0)
    elif reg_type == "l2":
        return weights / (1 + alpha * np.linalg.norm(weights))
    elif reg_type == "elasticnet":
        l1_weights = np.sign(weights) * np.maximum(np.abs(weights) - alpha, 0)
        l2_weights = weights / (1 + alpha * np.linalg.norm(weights))
        return 0.5 * l1_weights + 0.5 * l2_weights
    elif reg_type == "none":
        return weights
    else:
        raise ValueError(f"Unknown regularization type: {reg_type}")

def algorithmes_renforcement_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    solver: str = "gradient_descent",
    reg_type: str = "none",
    alpha: float = 1.0,
    **solver_kwargs
) -> Dict[str, Any]:
    """
    Fit reinforcement learning algorithms.

    Parameters:
    - X: Input features (n_samples, n_features)
    - y: Target values (n_samples,)
    - normalizer: Optional callable for data normalization
    - metric: Metric to evaluate performance ("mse", "mae", "r2", "logloss") or callable
    - solver: Optimization solver ("gradient_descent", "newton", "coordinate_descent")
    - reg_type: Regularization type ("none", "l1", "l2", "elasticnet")
    - alpha: Regularization strength
    - solver_kwargs: Additional arguments for the solver

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate inputs
    _validate_inputs(X, y, normalizer)

    # Normalize data if specified
    X_normalized = normalizer(X) if normalizer is not None else X

    # Choose solver
    if solver == "gradient_descent":
        weights = _gradient_descent(X_normalized, y, **solver_kwargs)
    elif solver == "newton":
        weights = _newton_method(X_normalized, y, **solver_kwargs)
    elif solver == "coordinate_descent":
        weights = _coordinate_descent(X_normalized, y, **solver_kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization
    weights = _apply_regularization(weights, reg_type, alpha)

    # Make predictions
    y_pred = X_normalized @ weights

    # Compute metrics
    metrics = _compute_metric(y, y_pred, metric)

    return {
        "result": {"weights": weights},
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric": metric,
            "solver": solver,
            "reg_type": reg_type,
            "alpha": alpha
        },
        "warnings": []
    }

################################################################################
# optimisation_basée_population
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def optimisation_basée_population_fit(
    objective_function: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    population_size: int = 100,
    generations: int = 50,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.7,
    selection_function: Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
    mutation_function: Callable[[np.ndarray, float], np.ndarray] = None,
    crossover_function: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = None,
    normalization: str = 'none',
    metric: Callable[[np.ndarray, np.ndarray], float] = None,
    tolerance: float = 1e-6,
    max_iterations: int = 1000
) -> Dict[str, Any]:
    """
    Optimisation basée sur une population stochastique.

    Parameters:
    - objective_function: Fonction objectif à minimiser.
    - bounds: Matrice de bornes [min, max] pour chaque paramètre.
    - population_size: Taille de la population.
    - generations: Nombre de générations.
    - mutation_rate: Taux de mutation.
    - crossover_rate: Taux de croisement.
    - selection_function: Fonction de sélection (par défaut: tournoi).
    - mutation_function: Fonction de mutation (par défaut: gaussienne).
    - crossover_function: Fonction de croisement (par défaut: uniforme).
    - normalization: Type de normalisation ('none', 'standard', 'minmax').
    - metric: Métrique pour évaluer la performance.
    - tolerance: Tolérance pour l'arrêt.
    - max_iterations: Nombre maximal d'itérations.

    Returns:
    - Dict contenant les résultats, métriques et paramètres utilisés.
    """
    # Validation des entrées
    _validate_inputs(objective_function, bounds, population_size)

    # Initialisation de la population
    population = _initialize_population(bounds, population_size)

    # Normalisation de la population
    if normalization != 'none':
        population = _normalize_population(population, normalization)

    # Boucle d'optimisation
    best_solution = None
    best_fitness = float('inf')
    warnings = []

    for generation in range(generations):
        # Évaluation de la population
        fitness = np.array([objective_function(ind) for ind in population])

        # Mise à jour de la meilleure solution
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_solution = population[current_best_idx].copy()

        # Sélection
        selected_population = _selection(population, fitness, selection_function)

        # Croisement
        offspring_population = _crossover(selected_population, crossover_rate, crossover_function)

        # Mutation
        offspring_population = _mutation(offspring_population, mutation_rate, mutation_function)

        # Remplacement
        population = _replacement(population, offspring_population, fitness)

        # Critère d'arrêt
        if best_fitness < tolerance:
            warnings.append(f"Convergence atteinte à la génération {generation}")
            break

    # Calcul des métriques
    metrics = {}
    if metric is not None:
        metrics['final_metric'] = metric(best_solution, np.array([best_fitness]))

    return {
        'result': best_solution,
        'metrics': metrics,
        'params_used': {
            'population_size': population_size,
            'generations': generations,
            'mutation_rate': mutation_rate,
            'crossover_rate': crossover_rate
        },
        'warnings': warnings
    }

def _validate_inputs(
    objective_function: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    population_size: int
) -> None:
    """Validation des entrées."""
    if not callable(objective_function):
        raise ValueError("objective_function doit être un callable")
    if bounds.shape[1] != 2:
        raise ValueError("bounds doit être une matrice [min, max]")
    if population_size <= 0:
        raise ValueError("population_size doit être positif")

def _initialize_population(
    bounds: np.ndarray,
    population_size: int
) -> np.ndarray:
    """Initialisation de la population."""
    n_params = bounds.shape[0]
    return np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(population_size, n_params))

def _normalize_population(
    population: np.ndarray,
    normalization: str
) -> np.ndarray:
    """Normalisation de la population."""
    if normalization == 'standard':
        mean = np.mean(population, axis=0)
        std = np.std(population, axis=0)
        return (population - mean) / (std + 1e-8)
    elif normalization == 'minmax':
        min_vals = np.min(population, axis=0)
        max_vals = np.max(population, axis=0)
        return (population - min_vals) / (max_vals - min_vals + 1e-8)
    return population

def _selection(
    population: np.ndarray,
    fitness: np.ndarray,
    selection_function: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Sélection des individus."""
    if selection_function is not None:
        return selection_function(population, fitness)
    # Sélection par tournoi par défaut
    selected_indices = []
    n_individuals = population.shape[0]
    for _ in range(n_individuals):
        tournament_size = min(3, n_individuals)
        candidates = np.random.choice(n_individuals, tournament_size, replace=False)
        winner = candidates[np.argmin(fitness[candidates])]
        selected_indices.append(winner)
    return population[selected_indices]

def _crossover(
    population: np.ndarray,
    crossover_rate: float,
    crossover_function: Optional[Callable[[np.ndarray, np.ndarray, float], np.ndarray]] = None
) -> np.ndarray:
    """Croisement des individus."""
    if crossover_function is not None:
        return crossover_function(population, population, crossover_rate)
    # Croisement uniforme par défaut
    n_individuals = population.shape[0]
    offspring = np.zeros_like(population)
    for i in range(0, n_individuals, 2):
        parent1 = population[i]
        parent2 = population[i + 1] if i + 1 < n_individuals else population[0]
        mask = np.random.rand(*parent1.shape) < crossover_rate
        offspring[i] = np.where(mask, parent1, parent2)
        if i + 1 < n_individuals:
            offspring[i + 1] = np.where(mask, parent2, parent1)
    return offspring

def _mutation(
    population: np.ndarray,
    mutation_rate: float,
    mutation_function: Optional[Callable[[np.ndarray, float], np.ndarray]] = None
) -> np.ndarray:
    """Mutation des individus."""
    if mutation_function is not None:
        return mutation_function(population, mutation_rate)
    # Mutation gaussienne par défaut
    mutated_population = population.copy()
    mask = np.random.rand(*population.shape) < mutation_rate
    noise = np.random.normal(0, 0.1, size=population.shape)
    mutated_population[mask] += noise[mask]
    return mutated_population

def _replacement(
    population: np.ndarray,
    offspring_population: np.ndarray,
    fitness: np.ndarray
) -> np.ndarray:
    """Remplacement des individus."""
    combined_population = np.vstack((population, offspring_population))
    combined_fitness = np.hstack((fitness, [objective_function(ind) for ind in offspring_population]))
    sorted_indices = np.argsort(combined_fitness)
    return combined_population[sorted_indices[:population.shape[0]]]

################################################################################
# méthodes_métaphoriques
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def méthodes_métaphoriques_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalisation: str = 'standard',
    métrique: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solveur: str = 'gradient_descent',
    régularisation: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Fonction principale pour l'optimisation stochastique avec méthodes métaphoriques.

    Parameters
    ----------
    X : np.ndarray
        Matrice des caractéristiques (n_samples, n_features).
    y : np.ndarray
        Vecteur cible (n_samples,).
    normalisation : str, optional
        Type de normalisation ('none', 'standard', 'minmax', 'robust').
    métrique : str or Callable, optional
        Métrique d'évaluation ('mse', 'mae', 'r2', 'logloss') ou fonction personnalisée.
    distance : str or Callable, optional
        Distance utilisée ('euclidean', 'manhattan', 'cosine', 'minkowski') ou fonction personnalisée.
    solveur : str, optional
        Solveur à utiliser ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    régularisation : str, optional
        Type de régularisation ('none', 'l1', 'l2', 'elasticnet').
    tol : float, optional
        Tolérance pour la convergence.
    max_iter : int, optional
        Nombre maximal d'itérations.
    learning_rate : float, optional
        Taux d'apprentissage pour les solveurs itératifs.
    custom_metric : Callable, optional
        Fonction personnalisée pour la métrique.
    custom_distance : Callable, optional
        Fonction personnalisée pour la distance.

    Returns
    -------
    Dict[str, Any]
        Dictionnaire contenant les résultats, métriques, paramètres utilisés et avertissements.
    """
    # Validation des entrées
    _validate_inputs(X, y)

    # Normalisation des données
    X_normalized = _apply_normalization(X, normalisation)

    # Initialisation des paramètres
    params = _initialize_params(X_normalized.shape[1])

    # Choix du solveur
    if solveur == 'closed_form':
        result = _closed_form_solution(X_normalized, y)
    elif solveur == 'gradient_descent':
        result = _gradient_descent(
            X_normalized, y,
            tol=tol,
            max_iter=max_iter,
            learning_rate=learning_rate
        )
    elif solveur == 'newton':
        result = _newton_method(
            X_normalized, y,
            tol=tol,
            max_iter=max_iter
        )
    elif solveur == 'coordinate_descent':
        result = _coordinate_descent(
            X_normalized, y,
            tol=tol,
            max_iter=max_iter
        )
    else:
        raise ValueError("Solveur non reconnu.")

    # Calcul des métriques
    metrics = _compute_metrics(
        y, result['prediction'],
        métrique=métrique,
        custom_metric=custom_metric
    )

    # Application de la régularisation si nécessaire
    if régularisation is not None:
        result['params'] = _apply_regularization(
            result['params'],
            régularisation,
            alpha=0.1  # Valeur par défaut, pourrait être paramétrée
        )

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalisation': normalisation,
            'métrique': métrique,
            'distance': distance,
            'solveur': solveur,
            'régularisation': régularisation
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validation des entrées."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X doit être une matrice (n_samples, n_features) et y un vecteur (n_samples,).")
    if X.shape[0] != y.shape[0]:
        raise ValueError("Le nombre d'échantillons dans X et y doit être identique.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contient des valeurs NaN ou inf.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contient des valeurs NaN ou inf.")

def _apply_normalization(X: np.ndarray, normalisation: str) -> np.ndarray:
    """Application de la normalisation."""
    if normalisation == 'none':
        return X
    elif normalisation == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif normalisation == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif normalisation == 'robust':
        median = np.median(X, axis=0)
        iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError("Normalisation non reconnue.")

def _initialize_params(n_features: int) -> np.ndarray:
    """Initialisation des paramètres."""
    return np.zeros(n_features)

def _closed_form_solution(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Solution en forme fermée."""
    XTX = np.dot(X.T, X)
    if np.linalg.det(XTX) == 0:
        raise ValueError("Matrice non inversible.")
    params = np.linalg.solve(XTX, np.dot(X.T, y))
    prediction = np.dot(X, params)
    return {'params': params, 'prediction': prediction}

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01
) -> Dict[str, Any]:
    """Descente de gradient stochastique."""
    n_samples, n_features = X.shape
    params = np.zeros(n_features)
    prev_loss = float('inf')

    for _ in range(max_iter):
        for i in range(n_samples):
            xi = X[i].reshape(1, -1)
            yi = y[i]
            gradient = 2 * xi.T.dot(xi.dot(params) - yi)
            params -= learning_rate * gradient

        current_loss = np.mean((X.dot(params) - y) ** 2)
        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    prediction = X.dot(params)
    return {'params': params, 'prediction': prediction}

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> Dict[str, Any]:
    """Méthode de Newton."""
    n_samples, n_features = X.shape
    params = np.zeros(n_features)
    prev_loss = float('inf')

    for _ in range(max_iter):
        gradient = 2 * X.T.dot(X.dot(params) - y)
        hessian = 2 * X.T.dot(X)

        if np.linalg.det(hessian) == 0:
            raise ValueError("Hessienne non inversible.")

        params -= np.linalg.solve(hessian, gradient)
        current_loss = np.mean((X.dot(params) - y) ** 2)

        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    prediction = X.dot(params)
    return {'params': params, 'prediction': prediction}

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> Dict[str, Any]:
    """Descente de coordonnées."""
    n_samples, n_features = X.shape
    params = np.zeros(n_features)
    prev_loss = float('inf')

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - X.dot(params) + params[j] * X_j
            params[j] = np.sum(X_j * residuals) / np.sum(X_j ** 2)

        current_loss = np.mean((X.dot(params) - y) ** 2)
        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    prediction = X.dot(params)
    return {'params': params, 'prediction': prediction}

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    métrique: Union[str, Callable] = 'mse',
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """Calcul des métriques."""
    metrics = {}

    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)

    if métrique == 'mse':
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    elif métrique == 'mae':
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    elif métrique == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)
    elif métrique == 'logloss':
        metrics['logloss'] = -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
    else:
        raise ValueError("Métrique non reconnue.")

    return metrics

def _apply_regularization(
    params: np.ndarray,
    régularisation: str,
    alpha: float = 0.1
) -> np.ndarray:
    """Application de la régularisation."""
    if régularisation == 'l1':
        return np.sign(params) * np.maximum(np.abs(params) - alpha, 0)
    elif régularisation == 'l2':
        return params / (1 + alpha * np.abs(params))
    elif régularisation == 'elasticnet':
        l1 = np.sign(params) * np.maximum(np.abs(params) - alpha, 0)
        l2 = params / (1 + alpha * np.abs(params))
        return (l1 + l2) / 2
    else:
        raise ValueError("Régularisation non reconnue.")

################################################################################
# optimisation_stochastique_adaptive
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def optimisation_stochastique_adaptive_fit(
    objective_func: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    bounds: Optional[tuple] = None,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric_func: Callable[[float, float], float] = lambda y_true, y_pred: (y_true - y_pred)**2,
    solver: str = 'gradient_descent',
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6,
    regularization: Optional[str] = None,
    reg_param: float = 1.0,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Optimisation stochastique adaptative avec choix paramétrables.

    Parameters:
    -----------
    objective_func : callable
        Fonction objectif à minimiser.
    initial_params : np.ndarray
        Paramètres initiaux pour l'optimisation.
    bounds : tuple, optional
        Bornes des paramètres (min, max).
    normalizer : callable, optional
        Fonction de normalisation des paramètres.
    metric_func : callable, optional
        Métrique pour évaluer la performance.
    solver : str, optional
        Solveur à utiliser ('gradient_descent', 'newton', etc.).
    learning_rate : float, optional
        Taux d'apprentissage pour les solveurs itératifs.
    max_iter : int, optional
        Nombre maximal d'itérations.
    tol : float, optional
        Tolérance pour l'arrêt des itérations.
    regularization : str, optional
        Type de régularisation ('l1', 'l2', 'elasticnet').
    reg_param : float, optional
        Paramètre de régularisation.
    random_state : int, optional
        Graine pour la reproductibilité.

    Returns:
    --------
    dict
        Dictionnaire contenant les résultats, métriques et paramètres utilisés.
    """
    # Validation des entrées
    _validate_inputs(objective_func, initial_params, bounds)

    # Initialisation
    params = normalizer(initial_params.copy())
    best_params = params.copy()
    best_value = objective_func(params)
    history = {'params': [params], 'values': [best_value]}

    # Optimisation
    for _ in range(max_iter):
        params_new = _optimize_step(
            objective_func=objective_func,
            current_params=params,
            solver=solver,
            learning_rate=learning_rate,
            bounds=bounds,
            regularization=regularization,
            reg_param=reg_param
        )

        value_new = objective_func(params_new)

        # Mise à jour des meilleurs paramètres
        if metric_func(best_value, value_new) > 0:
            best_params = params_new.copy()
            best_value = value_new

        # Critère d'arrêt
        if np.linalg.norm(params_new - params) < tol:
            break

        params = params_new
        history['params'].append(params)
        history['values'].append(value_new)

    # Calcul des métriques
    metrics = _compute_metrics(objective_func, best_params)

    # Retour des résultats
    return {
        'result': best_params,
        'metrics': metrics,
        'params_used': {
            'solver': solver,
            'learning_rate': learning_rate,
            'max_iter': max_iter,
            'tol': tol,
            'regularization': regularization,
            'reg_param': reg_param
        },
        'warnings': _check_warnings(history)
    }

def _validate_inputs(
    objective_func: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    bounds: Optional[tuple]
) -> None:
    """Validation des entrées."""
    if not callable(objective_func):
        raise ValueError("objective_func doit être une fonction callable.")
    if not isinstance(initial_params, np.ndarray):
        raise ValueError("initial_params doit être un tableau NumPy.")
    if bounds is not None and len(bounds) != 2:
        raise ValueError("bounds doit être un tuple de deux valeurs (min, max).")

def _optimize_step(
    objective_func: Callable[[np.ndarray], float],
    current_params: np.ndarray,
    solver: str,
    learning_rate: float,
    bounds: Optional[tuple],
    regularization: Optional[str],
    reg_param: float
) -> np.ndarray:
    """Étape d'optimisation unique."""
    if solver == 'gradient_descent':
        return _gradient_descent_step(
            objective_func=objective_func,
            current_params=current_params,
            learning_rate=learning_rate,
            bounds=bounds
        )
    elif solver == 'newton':
        return _newton_step(
            objective_func=objective_func,
            current_params=current_params
        )
    else:
        raise ValueError(f"Solveur {solver} non pris en charge.")

def _gradient_descent_step(
    objective_func: Callable[[np.ndarray], float],
    current_params: np.ndarray,
    learning_rate: float,
    bounds: Optional[tuple]
) -> np.ndarray:
    """Étape de descente de gradient."""
    # Calcul du gradient (approximation numérique)
    epsilon = 1e-8
    grad = np.zeros_like(current_params)
    for i in range(len(current_params)):
        params_plus = current_params.copy()
        params_plus[i] += epsilon
        params_minus = current_params.copy()
        params_minus[i] -= epsilon

        grad[i] = (objective_func(params_plus) - objective_func(params_minus)) / (2 * epsilon)

    # Mise à jour des paramètres
    new_params = current_params - learning_rate * grad

    # Application des bornes si spécifiées
    if bounds is not None:
        new_params = np.clip(new_params, bounds[0], bounds[1])

    return new_params

def _newton_step(
    objective_func: Callable[[np.ndarray], float],
    current_params: np.ndarray
) -> np.ndarray:
    """Étape de méthode de Newton."""
    # Approximation numérique de la hessienne
    epsilon = 1e-8
    n = len(current_params)
    hessian = np.zeros((n, n))
    for i in range(n):
        params_plus_i = current_params.copy()
        params_plus_i[i] += epsilon
        for j in range(n):
            params_plus_j = current_params.copy()
            params_plus_j[j] += epsilon
            hessian[i, j] = (objective_func(params_plus_i) - objective_func(params_plus_j)) / (epsilon ** 2)

    # Résolution du système linéaire
    grad = _compute_gradient(objective_func, current_params)
    new_params = current_params - np.linalg.solve(hessian, grad)

    return new_params

def _compute_gradient(
    objective_func: Callable[[np.ndarray], float],
    params: np.ndarray
) -> np.ndarray:
    """Calcul du gradient par différences finies."""
    epsilon = 1e-8
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += epsilon
        params_minus = params.copy()
        params_minus[i] -= epsilon

        grad[i] = (objective_func(params_plus) - objective_func(params_minus)) / (2 * epsilon)

    return grad

def _compute_metrics(
    objective_func: Callable[[np.ndarray], float],
    params: np.ndarray
) -> Dict[str, float]:
    """Calcul des métriques."""
    value = objective_func(params)
    return {
        'objective_value': float(value),
        'gradient_norm': float(np.linalg.norm(_compute_gradient(objective_func, params)))
    }

def _check_warnings(
    history: Dict[str, list]
) -> Dict[str, str]:
    """Vérification des avertissements."""
    warnings = []
    if len(history['values']) == history['params'][0]:
        warnings.append("Aucune amélioration pendant l'optimisation.")
    return {'warnings': warnings} if warnings else {}

################################################################################
# algorithmes_heuristiques
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    distance_metric: str = 'euclidean',
    custom_distance: Optional[Callable] = None,
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if custom_distance is not None:
        if not callable(custom_distance):
            raise TypeError("custom_distance must be a callable")
    else:
        valid_metrics = ['euclidean', 'manhattan', 'cosine', 'minkowski']
        if distance_metric not in valid_metrics:
            raise ValueError(f"distance_metric must be one of {valid_metrics}")

def _normalize_data(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalization: str = 'none',
) -> tuple:
    """Normalize data according to specified method."""
    if normalization == 'none':
        return X, y
    elif normalization == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_normalized = (X - mean) / std
    elif normalization == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_normalized = (X - min_val) / (max_val - min_val + 1e-8)
    elif normalization == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_normalized = (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    if y is not None:
        return X_normalized, y
    else:
        return X_normalized

def _compute_distance(
    X: np.ndarray,
    distance_metric: str = 'euclidean',
    custom_distance: Optional[Callable] = None,
) -> np.ndarray:
    """Compute distance matrix using specified metric."""
    n_samples = X.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    if custom_distance is not None:
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance_matrix[i, j] = custom_distance(X[i], X[j])
                distance_matrix[j, i] = distance_matrix[i, j]
    else:
        if distance_metric == 'euclidean':
            for i in range(n_samples):
                for j in range(i, n_samples):
                    distance_matrix[i, j] = np.linalg.norm(X[i] - X[j])
                    distance_matrix[j, i] = distance_matrix[i, j]
        elif distance_metric == 'manhattan':
            for i in range(n_samples):
                for j in range(i, n_samples):
                    distance_matrix[i, j] = np.sum(np.abs(X[i] - X[j]))
                    distance_matrix[j, i] = distance_matrix[i, j]
        elif distance_metric == 'cosine':
            for i in range(n_samples):
                for j in range(i, n_samples):
                    dot_product = np.dot(X[i], X[j])
                    norm_i = np.linalg.norm(X[i])
                    norm_j = np.linalg.norm(X[j])
                    distance_matrix[i, j] = 1 - (dot_product / (norm_i * norm_j + 1e-8))
                    distance_matrix[j, i] = distance_matrix[i, j]
        elif distance_metric == 'minkowski':
            for i in range(n_samples):
                for j in range(i, n_samples):
                    distance_matrix[i, j] = np.sum(np.abs(X[i] - X[j]) ** 3) ** (1/3)
                    distance_matrix[j, i] = distance_matrix[i, j]

    return distance_matrix

def _optimize_heuristic(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    metric: str = 'mse',
    custom_metric: Optional[Callable] = None,
    solver: str = 'gradient_descent',
    max_iter: int = 100,
    tol: float = 1e-4,
) -> Dict[str, Any]:
    """Optimize using specified heuristic method."""
    if solver == 'gradient_descent':
        return _gradient_descent(X, y, metric, custom_metric, max_iter, tol)
    elif solver == 'newton':
        return _newton_method(X, y, metric, custom_metric, max_iter, tol)
    elif solver == 'coordinate_descent':
        return _coordinate_descent(X, y, metric, custom_metric, max_iter, tol)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _gradient_descent(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    metric: str = 'mse',
    custom_metric: Optional[Callable] = None,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> Dict[str, Any]:
    """Gradient descent optimization."""
    # Placeholder for gradient descent implementation
    return {
        'result': None,
        'metrics': {},
        'params_used': {'solver': 'gradient_descent'},
        'warnings': []
    }

def _newton_method(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    metric: str = 'mse',
    custom_metric: Optional[Callable] = None,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> Dict[str, Any]:
    """Newton method optimization."""
    # Placeholder for Newton method implementation
    return {
        'result': None,
        'metrics': {},
        'params_used': {'solver': 'newton'},
        'warnings': []
    }

def _coordinate_descent(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    metric: str = 'mse',
    custom_metric: Optional[Callable] = None,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> Dict[str, Any]:
    """Coordinate descent optimization."""
    # Placeholder for coordinate descent implementation
    return {
        'result': None,
        'metrics': {},
        'params_used': {'solver': 'coordinate_descent'},
        'warnings': []
    }

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = 'mse',
    custom_metric: Optional[Callable] = None,
) -> Dict[str, float]:
    """Compute specified metrics."""
    metrics = {}

    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)
    else:
        if metric == 'mse':
            metrics['mse'] = np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['r2'] = 1 - (ss_res / (ss_tot + 1e-8))
        elif metric == 'logloss':
            y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
            metrics['logloss'] = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    return metrics

def algorithmes_heuristiques_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    distance_metric: str = 'euclidean',
    custom_distance: Optional[Callable] = None,
    normalization: str = 'none',
    metric: str = 'mse',
    custom_metric: Optional[Callable] = None,
    solver: str = 'gradient_descent',
    max_iter: int = 100,
    tol: float = 1e-4,
) -> Dict[str, Any]:
    """
    Main function for heuristic optimization algorithms.

    Parameters:
    - X: Input data (n_samples, n_features)
    - y: Target values (optional) (n_samples,)
    - distance_metric: Distance metric to use
    - custom_distance: Custom distance function (optional)
    - normalization: Data normalization method
    - metric: Performance metric to optimize
    - custom_metric: Custom performance metric (optional)
    - solver: Optimization algorithm to use
    - max_iter: Maximum number of iterations
    - tol: Tolerance for convergence

    Returns:
    Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate inputs
    _validate_inputs(X, y, distance_metric, custom_distance)

    # Normalize data
    X_normalized = _normalize_data(X, y, normalization)[0]
    if y is not None:
        _, y_normalized = _normalize_data(X, y, normalization)
    else:
        y_normalized = None

    # Compute distance matrix
    distance_matrix = _compute_distance(X_normalized, distance_metric, custom_distance)

    # Optimize using specified heuristic
    result = _optimize_heuristic(
        distance_matrix,
        y_normalized,
        metric,
        custom_metric,
        solver,
        max_iter,
        tol
    )

    # Compute final metrics if y is provided
    if y is not None and result['result'] is not None:
        metrics = _compute_metrics(y, result['result'], metric, custom_metric)
    else:
        metrics = {}

    return {
        'result': result['result'],
        'metrics': metrics,
        'params_used': {
            'distance_metric': distance_metric,
            'normalization': normalization,
            'metric': metric,
            'solver': solver,
            'max_iter': max_iter,
            'tol': tol
        },
        'warnings': result['warnings']
    }

################################################################################
# méthodes_monte_carlo
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def méthodes_monte_carlo_fit(
    objective_function: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    n_iterations: int = 1000,
    step_size: float = 0.01,
    normalize: str = 'none',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict[str, float], str]]:
    """
    Optimisation stochastique par méthodes de Monte Carlo.

    Parameters
    ----------
    objective_function : Callable[[np.ndarray], float]
        Fonction objectif à minimiser.
    initial_params : np.ndarray
        Paramètres initiaux pour l'optimisation.
    n_iterations : int, optional
        Nombre d'itérations, par défaut 1000.
    step_size : float, optional
        Taille du pas pour les méthodes de gradient, par défaut 0.01.
    normalize : str, optional
        Méthode de normalisation ('none', 'standard', 'minmax', 'robust'), par défaut 'none'.
    metric : Union[str, Callable], optional
        Métrique d'évaluation ('mse', 'mae', 'r2', 'logloss') ou fonction personnalisée, par défaut 'mse'.
    solver : str, optional
        Solveur à utiliser ('gradient_descent', 'newton', 'coordinate_descent'), par défaut 'gradient_descent'.
    regularization : Optional[str], optional
        Type de régularisation ('none', 'l1', 'l2', 'elasticnet'), par défaut None.
    tol : float, optional
        Tolérance pour l'arrêt, par défaut 1e-6.
    random_state : Optional[int], optional
        Graine pour la reproductibilité, par défaut None.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict[str, float], str]]
        Dictionnaire contenant les résultats de l'optimisation.

    Examples
    --------
    >>> def objective(x):
    ...     return x[0]**2 + x[1]**2
    >>> result = méthodes_monte_carlo_fit(objective, np.array([1.0, 2.0]))
    """
    # Initialisation
    params = initial_params.copy()
    rng = np.random.RandomState(random_state)

    # Validation des entrées
    _validate_inputs(objective_function, initial_params)

    # Normalisation
    if normalize != 'none':
        params = _normalize(params, method=normalize)

    # Initialisation des métriques
    metrics = _initialize_metrics(metric, objective_function)

    # Optimisation
    for i in range(n_iterations):
        if solver == 'gradient_descent':
            params = _gradient_descent_step(objective_function, params, step_size, rng)
        elif solver == 'newton':
            params = _newton_step(objective_function, params)
        elif solver == 'coordinate_descent':
            params = _coordinate_descent_step(objective_function, params)

        # Régularisation
        if regularization is not None:
            params = _apply_regularization(params, method=regularization)

        # Vérification de convergence
        if _check_convergence(params, tol):
            break

    # Calcul des métriques finales
    final_metrics = _compute_metrics(params, metric, objective_function)

    # Retour des résultats
    return {
        'result': params,
        'metrics': final_metrics,
        'params_used': {
            'n_iterations': i + 1,
            'step_size': step_size,
            'normalize': normalize,
            'metric': metric if isinstance(metric, str) else 'custom',
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

def _validate_inputs(objective_function: Callable, params: np.ndarray) -> None:
    """Validation des entrées."""
    if not callable(objective_function):
        raise ValueError("objective_function must be a callable.")
    if not isinstance(params, np.ndarray):
        raise ValueError("params must be a numpy array.")
    if np.any(np.isnan(params)) or np.any(np.isinf(params)):
        raise ValueError("params must not contain NaN or inf values.")

def _normalize(params: np.ndarray, method: str) -> np.ndarray:
    """Normalisation des paramètres."""
    if method == 'standard':
        return (params - np.mean(params)) / np.std(params)
    elif method == 'minmax':
        return (params - np.min(params)) / (np.max(params) - np.min(params))
    elif method == 'robust':
        return (params - np.median(params)) / (np.percentile(params, 75) - np.percentile(params, 25))
    return params

def _initialize_metrics(metric: Union[str, Callable], objective_function: Callable) -> Dict[str, float]:
    """Initialisation des métriques."""
    return {'objective': objective_function(np.zeros_like(initial_params))}

def _gradient_descent_step(
    objective_function: Callable,
    params: np.ndarray,
    step_size: float,
    rng: np.random.RandomState
) -> np.ndarray:
    """Pas de descente de gradient."""
    epsilon = 1e-8
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += epsilon
        params_minus[i] -= epsilon
        grad[i] = (objective_function(params_plus) - objective_function(params_minus)) / (2 * epsilon)
    return params - step_size * grad

def _newton_step(objective_function: Callable, params: np.ndarray) -> np.ndarray:
    """Pas de méthode de Newton."""
    # Simplification: utilisation d'une approximation diagonale pour la hessienne
    epsilon = 1e-8
    hessian = np.zeros((len(params), len(params)))
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += epsilon
        params_minus[i] -= epsilon
        hessian[i, i] = (objective_function(params_plus) - 2 * objective_function(params) + objective_function(params_minus)) / (epsilon ** 2)
    grad = _compute_gradient(objective_function, params)
    return params - np.linalg.solve(hessian, grad)

def _coordinate_descent_step(objective_function: Callable, params: np.ndarray) -> np.ndarray:
    """Pas de descente de coordonnées."""
    for i in range(len(params)):
        params_minus = params.copy()
        params_minus[i] -= params[i]
        params_plus = params.copy()
        params_plus[i] += params[i]
        if objective_function(params_minus) < objective_function(params):
            params[i] = -params[i]
        elif objective_function(params_plus) < objective_function(params):
            params[i] = +params[i]
    return params

def _apply_regularization(params: np.ndarray, method: str) -> np.ndarray:
    """Application de la régularisation."""
    if method == 'l1':
        return np.sign(params) * np.maximum(np.abs(params) - 0.1, 0)
    elif method == 'l2':
        return params / (1 + 0.1 * np.linalg.norm(params))
    elif method == 'elasticnet':
        return _apply_regularization(params, 'l1') * 0.5 + _apply_regularization(params, 'l2') * 0.5
    return params

def _check_convergence(params: np.ndarray, tol: float) -> bool:
    """Vérification de la convergence."""
    return np.linalg.norm(params) < tol

def _compute_metrics(
    params: np.ndarray,
    metric: Union[str, Callable],
    objective_function: Callable
) -> Dict[str, float]:
    """Calcul des métriques."""
    metrics = {'objective': objective_function(params)}
    if isinstance(metric, str):
        if metric == 'mse':
            metrics['mse'] = _compute_mse(params)
        elif metric == 'mae':
            metrics['mae'] = _compute_mae(params)
        elif metric == 'r2':
            metrics['r2'] = _compute_r2(params)
        elif metric == 'logloss':
            metrics['logloss'] = _compute_logloss(params)
    else:
        metrics['custom_metric'] = metric(params)
    return metrics

def _compute_gradient(objective_function: Callable, params: np.ndarray) -> np.ndarray:
    """Calcul du gradient."""
    epsilon = 1e-8
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += epsilon
        params_minus[i] -= epsilon
        grad[i] = (objective_function(params_plus) - objective_function(params_minus)) / (2 * epsilon)
    return grad

def _compute_mse(params: np.ndarray) -> float:
    """Calcul de l'erreur quadratique moyenne."""
    return np.mean(params ** 2)

def _compute_mae(params: np.ndarray) -> float:
    """Calcul de l'erreur absolue moyenne."""
    return np.mean(np.abs(params))

def _compute_r2(params: np.ndarray) -> float:
    """Calcul du coefficient de détermination."""
    return 1 - np.sum(params ** 2) / (len(params) * np.var(params))

def _compute_logloss(params: np.ndarray) -> float:
    """Calcul de la log-vraisemblance."""
    return -np.mean(np.log(1 + np.exp(-params)))

################################################################################
# optimisation_basée_gradient_stochastique
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def optimisation_basée_gradient_stochastique_fit(
    loss_func: Callable[[np.ndarray, np.ndarray], float],
    gradient_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    initial_params: np.ndarray,
    learning_rate: float = 0.01,
    n_iterations: int = 1000,
    batch_size: Optional[int] = None,
    tol: float = 1e-4,
    normalization: str = 'none',
    metric_func: Callable[[np.ndarray, np.ndarray], float] = None,
    regularization: str = 'none',
    l1_penalty: float = 0.0,
    l2_penalty: float = 0.0,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Optimisation basée sur le gradient stochastique.

    Parameters:
    - loss_func: Fonction de perte à minimiser.
    - gradient_func: Fonction de calcul du gradient.
    - X: Matrice des caractéristiques (n_samples, n_features).
    - y: Vecteur des cibles (n_samples,).
    - initial_params: Paramètres initiaux.
    - learning_rate: Taux d'apprentissage.
    - n_iterations: Nombre d'itérations.
    - batch_size: Taille du batch pour le gradient stochastique. Si None, utilise tout le dataset.
    - tol: Tolérance pour l'arrêt.
    - normalization: Type de normalisation ('none', 'standard', 'minmax', 'robust').
    - metric_func: Fonction de métrique pour évaluer la performance.
    - regularization: Type de régularisation ('none', 'l1', 'l2', 'elasticnet').
    - l1_penalty: Coefficient de pénalité L1.
    - l2_penalty: Coefficient de pénalité L2.
    - random_state: Graine aléatoire pour la reproductibilité.

    Returns:
    - Dict contenant les résultats, métriques, paramètres utilisés et avertissements.
    """
    # Initialisation
    params = initial_params.copy()
    n_samples, n_features = X.shape

    if batch_size is None:
        batch_size = n_samples

    if random_state is not None:
        np.random.seed(random_state)

    # Normalisation
    X_normalized = _apply_normalization(X, normalization)
    y_normalized = _normalize_target(y, normalization)

    # Initialisation des métriques
    metrics = {}
    if metric_func is not None:
        initial_metric = metric_func(y, _predict(X_normalized, params))
        metrics['initial_metric'] = initial_metric

    # Boucle d'optimisation
    for i in range(n_iterations):
        # Sélection du batch
        indices = np.random.choice(n_samples, batch_size, replace=False)
        X_batch = X_normalized[indices]
        y_batch = y_normalized[indices]

        # Calcul du gradient
        gradient = _compute_gradient(
            loss_func,
            gradient_func,
            X_batch,
            y_batch,
            params,
            regularization,
            l1_penalty,
            l2_penalty
        )

        # Mise à jour des paramètres
        params -= learning_rate * gradient

        # Évaluation de la métrique
        if metric_func is not None and i % 10 == 0:
            current_metric = metric_func(y, _predict(X_normalized, params))
            metrics[f'iteration_{i}'] = current_metric

        # Critère d'arrêt
        if i > 0 and np.linalg.norm(gradient) < tol:
            break

    # Résultats finaux
    result = {
        'params': params,
        'metrics': metrics,
        'params_used': {
            'learning_rate': learning_rate,
            'n_iterations': i + 1,
            'batch_size': batch_size,
            'normalization': normalization,
            'regularization': regularization,
            'l1_penalty': l1_penalty,
            'l2_penalty': l2_penalty
        },
        'warnings': []
    }

    return result

def _apply_normalization(X: np.ndarray, normalization: str) -> np.ndarray:
    """Applique la normalisation spécifiée."""
    if normalization == 'standard':
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalization == 'minmax':
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalization == 'robust':
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    return X

def _normalize_target(y: np.ndarray, normalization: str) -> np.ndarray:
    """Normalise la cible selon le type de normalisation spécifié."""
    if normalization == 'standard':
        y = (y - np.mean(y)) / np.std(y)
    elif normalization == 'minmax':
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif normalization == 'robust':
        y = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    return y

def _predict(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Prédiction linéaire."""
    return X @ params

def _compute_gradient(
    loss_func: Callable[[np.ndarray, np.ndarray], float],
    gradient_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    X_batch: np.ndarray,
    y_batch: np.ndarray,
    params: np.ndarray,
    regularization: str,
    l1_penalty: float,
    l2_penalty: float
) -> np.ndarray:
    """Calcule le gradient avec régularisation."""
    gradient = gradient_func(X_batch, y_batch)
    if regularization == 'l1':
        gradient += l1_penalty * np.sign(params)
    elif regularization == 'l2':
        gradient += 2 * l2_penalty * params
    elif regularization == 'elasticnet':
        gradient += l1_penalty * np.sign(params) + 2 * l2_penalty * params
    return gradient

# Exemple minimal
if __name__ == "__main__":
    def mse_loss(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def mse_gradient(X, y):
        return (-2 / X.shape[0]) * (X.T @ (y - _predict(X, np.zeros(X.shape[1]))))

    X = np.random.rand(100, 5)
    y = np.random.rand(100)

    result = optimisation_basée_gradient_stochastique_fit(
        loss_func=mse_loss,
        gradient_func=mse_gradient,
        X=X,
        y=y,
        initial_params=np.zeros(X.shape[1]),
        learning_rate=0.01,
        n_iterations=1000
    )
