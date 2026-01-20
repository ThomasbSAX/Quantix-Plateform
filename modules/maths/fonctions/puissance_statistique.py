"""
Quantix – Module puissance_statistique
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# taille_echantillon
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.8,
    tail: str = 'two-sided',
) -> None:
    """Validate input parameters for sample size calculation."""
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be between 0 and 1")
    if not (0 < power <= 1):
        raise ValueError("Power must be between 0 and 1")
    if tail not in ['one-sided', 'two-sided']:
        raise ValueError("Tail must be either 'one-sided' or 'two-sided'")
    if effect_size <= 0:
        raise ValueError("Effect size must be positive")

def _calculate_z_score(
    alpha: float,
    power: float,
    tail: str = 'two-sided',
) -> float:
    """Calculate the z-score based on significance level and power."""
    from scipy.stats import norm
    if tail == 'two-sided':
        alpha /= 2
    z_alpha = norm.ppf(1 - alpha)
    z_beta = norm.ppf(power)
    return z_alpha + z_beta

def _calculate_sample_size(
    effect_size: float,
    z_score: float,
    sigma: Optional[float] = None,
) -> int:
    """Calculate the required sample size."""
    if sigma is not None and sigma <= 0:
        raise ValueError("Standard deviation must be positive")
    if sigma is None:
        # For proportion tests, effect_size is the difference in proportions
        n = (z_score / effect_size) ** 2
    else:
        # For mean tests, effect_size is the standardized effect size (Cohen's d)
        n = ((z_score / effect_size) ** 2 * sigma ** 2)
    return int(np.ceil(n))

def taille_echantillon_fit(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.8,
    tail: str = 'two-sided',
    sigma: Optional[float] = None,
    test_type: str = 'z_test',
) -> Dict[str, Union[int, float, Dict]]:
    """
    Calculate the required sample size for a statistical test.

    Parameters
    ----------
    effect_size : float
        The expected effect size (Cohen's d for mean tests, difference in proportions for proportion tests).
    alpha : float, optional
        Significance level (Type I error probability), by default 0.05.
    power : float, optional
        Desired statistical power (1 - Type II error probability), by default 0.8.
    tail : str, optional
        Test tail ('one-sided' or 'two-sided'), by default 'two-sided'.
    sigma : float, optional
        Population standard deviation (for mean tests), by default None.
    test_type : str, optional
        Type of statistical test ('z_test' or 't_test'), by default 'z_test'.

    Returns
    -------
    Dict[str, Union[int, float, Dict]]
        Dictionary containing:
        - 'result': calculated sample size
        - 'metrics': dictionary of intermediate metrics
        - 'params_used': dictionary of parameters used
        - 'warnings': list of warnings (if any)

    Examples
    --------
    >>> taille_echantillon_fit(effect_size=0.5, alpha=0.05, power=0.8)
    {
        'result': 64,
        'metrics': {'z_score': 2.807, 'n_calculated': 63.5},
        'params_used': {'effect_size': 0.5, 'alpha': 0.05, 'power': 0.8, 'tail': 'two-sided', 'sigma': None},
        'warnings': []
    }
    """
    _validate_inputs(effect_size, alpha, power, tail)

    z_score = _calculate_z_score(alpha, power, tail)
    n = _calculate_sample_size(effect_size, z_score, sigma)

    metrics = {
        'z_score': z_score,
        'n_calculated': n - 1 if n > 0 else 0
    }

    params_used = {
        'effect_size': effect_size,
        'alpha': alpha,
        'power': power,
        'tail': tail,
        'sigma': sigma,
        'test_type': test_type
    }

    warnings = []

    return {
        'result': n,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

################################################################################
# niveau_confiance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    sample_size: int,
    effect_size: float,
    significance_level: float = 0.05,
    power_target: float = 0.8
) -> None:
    """
    Validate input parameters for power analysis.

    Parameters
    ----------
    sample_size : int
        Sample size.
    effect_size : float
        Expected effect size.
    significance_level : float, optional
        Significance level (alpha), by default 0.05.
    power_target : float, optional
        Target statistical power, by default 0.8.

    Raises
    ------
    ValueError
        If any input parameter is invalid.
    """
    if sample_size <= 0:
        raise ValueError("Sample size must be positive.")
    if not (0 < significance_level < 1):
        raise ValueError("Significance level must be between 0 and 1.")
    if not (0 < power_target < 1):
        raise ValueError("Power target must be between 0 and 1.")
    if effect_size <= 0:
        raise ValueError("Effect size must be positive.")

def _calculate_z_scores(
    significance_level: float,
    power_target: float
) -> Dict[str, float]:
    """
    Calculate z-scores for given significance level and power target.

    Parameters
    ----------
    significance_level : float
        Significance level (alpha).
    power_target : float
        Target statistical power.

    Returns
    ------
    Dict[str, float]
        Dictionary containing z-scores for alpha and power.
    """
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - significance_level / 2)
    z_power = norm.ppf(power_target)
    return {"z_alpha": z_alpha, "z_power": z_power}

def _calculate_required_sample_size(
    effect_size: float,
    significance_level: float,
    power_target: float
) -> int:
    """
    Calculate required sample size for given parameters.

    Parameters
    ----------
    effect_size : float
        Expected effect size.
    significance_level : float
        Significance level (alpha).
    power_target : float
        Target statistical power.

    Returns
    ------
    int
        Required sample size.
    """
    z_scores = _calculate_z_scores(significance_level, power_target)
    numerator = (z_scores["z_alpha"] + z_scores["z_power"]) ** 2
    denominator = effect_size ** 2
    sample_size = numerator / denominator
    return int(np.ceil(sample_size))

def _calculate_statistical_power(
    sample_size: int,
    effect_size: float,
    significance_level: float
) -> float:
    """
    Calculate statistical power for given parameters.

    Parameters
    ----------
    sample_size : int
        Sample size.
    effect_size : float
        Expected effect size.
    significance_level : float
        Significance level (alpha).

    Returns
    ------
    float
        Calculated statistical power.
    """
    z_scores = _calculate_z_scores(significance_level, 0.5)  # Dummy power value
    numerator = (effect_size * np.sqrt(sample_size)) - z_scores["z_alpha"]
    denominator = np.sqrt(2)
    z_power = numerator / denominator
    power = norm.cdf(z_power)
    return power

def niveau_confiance_fit(
    sample_size: int,
    effect_size: float,
    significance_level: float = 0.05,
    power_target: Optional[float] = None,
    calculate_power: bool = False
) -> Dict[str, Union[Dict, float]]:
    """
    Calculate confidence level or statistical power based on input parameters.

    Parameters
    ----------
    sample_size : int
        Sample size.
    effect_size : float
        Expected effect size.
    significance_level : float, optional
        Significance level (alpha), by default 0.05.
    power_target : float, optional
        Target statistical power, by default None.
    calculate_power : bool, optional
        Whether to calculate power instead of sample size, by default False.

    Returns
    ------
    Dict[str, Union[Dict, float]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> niveau_confiance_fit(sample_size=100, effect_size=0.5)
    {
        'result': {'required_sample_size': 128},
        'metrics': {},
        'params_used': {
            'sample_size': 100,
            'effect_size': 0.5,
            'significance_level': 0.05
        },
        'warnings': []
    }
    """
    _validate_inputs(sample_size, effect_size, significance_level, power_target or 0.8)

    params_used = {
        "sample_size": sample_size,
        "effect_size": effect_size,
        "significance_level": significance_level
    }

    if calculate_power:
        power = _calculate_statistical_power(sample_size, effect_size, significance_level)
        result = {"statistical_power": power}
    else:
        required_sample_size = _calculate_required_sample_size(
            effect_size, significance_level, power_target or 0.8
        )
        result = {"required_sample_size": required_sample_size}

    return {
        "result": result,
        "metrics": {},
        "params_used": params_used,
        "warnings": []
    }

################################################################################
# marge_erreur
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def marge_erreur_fit(
    data: np.ndarray,
    alpha: float = 0.05,
    method: str = 'normal',
    confidence_interval: Optional[Callable] = None,
    normalize: bool = True
) -> Dict[str, Union[float, Dict]]:
    """
    Calculate the margin of error for a given dataset.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    alpha : float, optional
        Significance level (default is 0.05).
    method : str, optional
        Method to calculate margin of error ('normal', 't', or 'custom') (default is 'normal').
    confidence_interval : Optional[Callable], optional
        Custom callable for confidence interval calculation (default is None).
    normalize : bool, optional
        Whether to normalize the data (default is True).

    Returns
    -------
    Dict[str, Union[float, Dict]]
        Dictionary containing the result, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.array([1.2, 2.3, 3.4, 4.5])
    >>> result = marge_erreur_fit(data)
    """
    # Validate inputs
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha must be between 0 and 1.")
    if method not in ['normal', 't', 'custom']:
        raise ValueError("Method must be 'normal', 't', or 'custom'.")

    # Normalize data if required
    if normalize:
        data = (data - np.mean(data)) / np.std(data)

    # Calculate mean and standard error
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(len(data))

    # Calculate margin of error based on method
    if method == 'normal':
        z_score = _calculate_z_score(alpha)
        margin_of_error = z_score * std_err
    elif method == 't':
        t_score = _calculate_t_score(alpha, len(data) - 1)
        margin_of_error = t_score * std_err
    elif method == 'custom' and confidence_interval is not None:
        margin_of_error = confidence_interval(data, alpha)
    else:
        raise ValueError("Custom method requires a custom callable.")

    # Prepare the result dictionary
    result = {
        "result": margin_of_error,
        "metrics": {
            "mean": mean,
            "std_err": std_err
        },
        "params_used": {
            "alpha": alpha,
            "method": method,
            "normalize": normalize
        },
        "warnings": []
    }

    return result

def _calculate_z_score(alpha: float) -> float:
    """
    Calculate the z-score for a given significance level.

    Parameters
    ----------
    alpha : float
        Significance level.

    Returns
    -------
    float
        Z-score.
    """
    from scipy.stats import norm
    return norm.ppf(1 - alpha / 2)

def _calculate_t_score(alpha: float, df: int) -> float:
    """
    Calculate the t-score for a given significance level and degrees of freedom.

    Parameters
    ----------
    alpha : float
        Significance level.
    df : int
        Degrees of freedom.

    Returns
    -------
    float
        T-score.
    """
    from scipy.stats import t
    return t.ppf(1 - alpha / 2, df)

################################################################################
# effet_taille
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def effet_taille_fit(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.8,
    tail: str = 'two-sided',
    normalization: Optional[str] = None,
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    custom_metric: Optional[Callable] = None,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict:
    """
    Calculate the sample size required to achieve a desired statistical power for a given effect size.

    Parameters:
    -----------
    effect_size : float
        The expected effect size.
    alpha : float, optional
        Significance level (default is 0.05).
    power : float, optional
        Desired statistical power (default is 0.8).
    tail : str, optional
        Type of test ('two-sided', 'one-sided') (default is 'two-sided').
    normalization : str, optional
        Type of normalization ('none', 'standard', 'minmax', 'robust') (default is None).
    metric : Union[str, Callable], optional
        Metric to use ('mse', 'mae', 'r2', 'logloss') or custom callable (default is 'mse').
    solver : str, optional
        Solver to use ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent') (default is 'closed_form').
    custom_metric : Callable, optional
        Custom metric function (default is None).
    tol : float, optional
        Tolerance for convergence (default is 1e-6).
    max_iter : int, optional
        Maximum number of iterations (default is 1000).

    Returns:
    --------
    Dict
        A dictionary containing the result, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(effect_size, alpha, power, tail)

    # Normalize the effect size if specified
    normalized_effect_size = _normalize(effect_size, normalization)

    # Calculate the required sample size
    if solver == 'closed_form':
        result = _closed_form_solution(normalized_effect_size, alpha, power, tail)
    else:
        result = _iterative_solution(normalized_effect_size, alpha, power, tail, metric, custom_metric, solver, tol, max_iter)

    # Calculate metrics
    metrics = _calculate_metrics(result, normalized_effect_size, metric, custom_metric)

    # Prepare the output dictionary
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'effect_size': effect_size,
            'alpha': alpha,
            'power': power,
            'tail': tail,
            'normalization': normalization,
            'metric': metric if isinstance(metric, str) else 'custom',
            'solver': solver,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': _check_warnings(result, normalized_effect_size)
    }

    return output

def _validate_inputs(effect_size: float, alpha: float, power: float, tail: str) -> None:
    """
    Validate the input parameters.

    Parameters:
    -----------
    effect_size : float
        The expected effect size.
    alpha : float
        Significance level.
    power : float
        Desired statistical power.
    tail : str
        Type of test.

    Raises:
    -------
    ValueError
        If any input is invalid.
    """
    if not isinstance(effect_size, (int, float)) or effect_size <= 0:
        raise ValueError("Effect size must be a positive number.")
    if not isinstance(alpha, (int, float)) or alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha must be between 0 and 1.")
    if not isinstance(power, (int, float)) or power <= 0 or power >= 1:
        raise ValueError("Power must be between 0 and 1.")
    if tail not in ['two-sided', 'one-sided']:
        raise ValueError("Tail must be either 'two-sided' or 'one-sided'.")

def _normalize(effect_size: float, normalization: Optional[str]) -> float:
    """
    Normalize the effect size based on the specified method.

    Parameters:
    -----------
    effect_size : float
        The expected effect size.
    normalization : str, optional
        Type of normalization.

    Returns:
    --------
    float
        The normalized effect size.
    """
    if normalization is None:
        return effect_size
    elif normalization == 'standard':
        return (effect_size - np.mean([0, 1])) / np.std([0, 1])
    elif normalization == 'minmax':
        return (effect_size - np.min([0, 1])) / (np.max([0, 1]) - np.min([0, 1]))
    elif normalization == 'robust':
        return (effect_size - np.median([0, 1])) / (np.percentile([0, 1], 75) - np.percentile([0, 1], 25))
    else:
        raise ValueError("Invalid normalization method.")

def _closed_form_solution(effect_size: float, alpha: float, power: float, tail: str) -> float:
    """
    Calculate the sample size using a closed-form solution.

    Parameters:
    -----------
    effect_size : float
        The expected effect size.
    alpha : float
        Significance level.
    power : float
        Desired statistical power.
    tail : str
        Type of test.

    Returns:
    --------
    float
        The required sample size.
    """
    z_alpha = _get_z_score(alpha, tail)
    z_power = _get_z_score(power, 'one-sided')
    sample_size = (z_alpha + z_power) ** 2 / effect_size ** 2
    return sample_size

def _iterative_solution(
    effect_size: float,
    alpha: float,
    power: float,
    tail: str,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable],
    solver: str,
    tol: float,
    max_iter: int
) -> float:
    """
    Calculate the sample size using an iterative solution.

    Parameters:
    -----------
    effect_size : float
        The expected effect size.
    alpha : float
        Significance level.
    power : float
        Desired statistical power.
    tail : str
        Type of test.
    metric : Union[str, Callable]
        Metric to use.
    custom_metric : Callable, optional
        Custom metric function.
    solver : str
        Solver to use.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.

    Returns:
    --------
    float
        The required sample size.
    """
    if solver == 'gradient_descent':
        return _gradient_descent(effect_size, alpha, power, tail, metric, custom_metric, tol, max_iter)
    elif solver == 'newton':
        return _newton_method(effect_size, alpha, power, tail, metric, custom_metric, tol, max_iter)
    elif solver == 'coordinate_descent':
        return _coordinate_descent(effect_size, alpha, power, tail, metric, custom_metric, tol, max_iter)
    else:
        raise ValueError("Invalid solver method.")

def _gradient_descent(
    effect_size: float,
    alpha: float,
    power: float,
    tail: str,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable],
    tol: float,
    max_iter: int
) -> float:
    """
    Calculate the sample size using gradient descent.

    Parameters:
    -----------
    effect_size : float
        The expected effect size.
    alpha : float
        Significance level.
    power : float
        Desired statistical power.
    tail : str
        Type of test.
    metric : Union[str, Callable]
        Metric to use.
    custom_metric : Callable, optional
        Custom metric function.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.

    Returns:
    --------
    float
        The required sample size.
    """
    # Implementation of gradient descent algorithm
    pass

def _newton_method(
    effect_size: float,
    alpha: float,
    power: float,
    tail: str,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable],
    tol: float,
    max_iter: int
) -> float:
    """
    Calculate the sample size using Newton's method.

    Parameters:
    -----------
    effect_size : float
        The expected effect size.
    alpha : float
        Significance level.
    power : float
        Desired statistical power.
    tail : str
        Type of test.
    metric : Union[str, Callable]
        Metric to use.
    custom_metric : Callable, optional
        Custom metric function.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.

    Returns:
    --------
    float
        The required sample size.
    """
    # Implementation of Newton's method algorithm
    pass

def _coordinate_descent(
    effect_size: float,
    alpha: float,
    power: float,
    tail: str,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable],
    tol: float,
    max_iter: int
) -> float:
    """
    Calculate the sample size using coordinate descent.

    Parameters:
    -----------
    effect_size : float
        The expected effect size.
    alpha : float
        Significance level.
    power : float
        Desired statistical power.
    tail : str
        Type of test.
    metric : Union[str, Callable]
        Metric to use.
    custom_metric : Callable, optional
        Custom metric function.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.

    Returns:
    --------
    float
        The required sample size.
    """
    # Implementation of coordinate descent algorithm
    pass

def _get_z_score(p: float, tail: str) -> float:
    """
    Get the z-score for a given probability and tail type.

    Parameters:
    -----------
    p : float
        Probability.
    tail : str
        Type of test.

    Returns:
    --------
    float
        The z-score.
    """
    if tail == 'two-sided':
        return np.abs(np.percentile(np.random.normal(0, 1, 10000), (1 - p) * 50))
    else:
        return np.percentile(np.random.normal(0, 1, 10000), (1 - p) * 100)

def _calculate_metrics(
    sample_size: float,
    effect_size: float,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """
    Calculate the metrics for the given sample size and effect size.

    Parameters:
    -----------
    sample_size : float
        The required sample size.
    effect_size : float
        The expected effect size.
    metric : Union[str, Callable]
        Metric to use.
    custom_metric : Callable, optional
        Custom metric function.

    Returns:
    --------
    Dict
        A dictionary containing the calculated metrics.
    """
    if isinstance(metric, str):
        if metric == 'mse':
            return {'mse': sample_size * effect_size ** 2}
        elif metric == 'mae':
            return {'mae': sample_size * effect_size}
        elif metric == 'r2':
            return {'r2': 1 - (1 / sample_size)}
        elif metric == 'logloss':
            return {'logloss': -np.log(effect_size)}
        else:
            raise ValueError("Invalid metric.")
    else:
        if custom_metric is None:
            raise ValueError("Custom metric function must be provided.")
        return {'custom_metric': custom_metric(sample_size, effect_size)}

def _check_warnings(sample_size: float, effect_size: float) -> list:
    """
    Check for warnings based on the sample size and effect size.

    Parameters:
    -----------
    sample_size : float
        The required sample size.
    effect_size : float
        The expected effect size.

    Returns:
    --------
    list
        A list of warning messages.
    """
    warnings = []
    if sample_size > 1000:
        warnings.append("Large sample size may be required.")
    if effect_size < 0.2:
        warnings.append("Small effect size may lead to high sample size requirements.")
    return warnings

################################################################################
# variabilite_donnees
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def _validate_inputs(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalize: str = "standard",
    distance_metric: Union[str, Callable] = "euclidean"
) -> Dict:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y is not None and X.shape[0] != len(y):
        raise ValueError("X and y must have the same number of samples")

    warnings = []
    if np.any(np.isnan(X)):
        warnings.append("NaN values found in X and removed")
        X = np.nan_to_num(X)
    if y is not None and np.any(np.isnan(y)):
        warnings.append("NaN values found in y and removed")
        y = np.nan_to_num(y)

    return {"X": X, "y": y, "warnings": warnings}

def _normalize_data(
    X: np.ndarray,
    method: str = "standard"
) -> np.ndarray:
    """Normalize data using specified method."""
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

def _compute_distance(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    metric: Union[str, Callable] = "euclidean"
) -> np.ndarray:
    """Compute distance matrix using specified metric."""
    if callable(metric):
        return metric(X, y)
    elif metric == "euclidean":
        return np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
    elif metric == "manhattan":
        return np.sum(np.abs(X[:, np.newaxis] - X), axis=2)
    elif metric == "cosine":
        dot_products = np.dot(X, X.T)
        norms = np.sqrt(np.sum(X ** 2, axis=1))[:, np.newaxis]
        return 1 - dot_products / (norms * norms.T + 1e-8)
    elif metric == "minkowski":
        return np.sum(np.abs(X[:, np.newaxis] - X) ** 3, axis=2) ** (1/3)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Union[str, List[str], Callable] = "mse"
) -> Dict:
    """Compute specified metrics between true and predicted values."""
    results = {}
    if callable(metrics):
        results["custom"] = metrics(y_true, y_pred)
    else:
        if isinstance(metrics, str):
            metrics = [metrics]
        for metric in metrics:
            if metric == "mse":
                results["mse"] = np.mean((y_true - y_pred) ** 2)
            elif metric == "mae":
                results["mae"] = np.mean(np.abs(y_true - y_pred))
            elif metric == "r2":
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                results["r2"] = 1 - (ss_res / (ss_tot + 1e-8))
            elif metric == "logloss":
                y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
                results["logloss"] = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            else:
                raise ValueError(f"Unknown metric: {metric}")
    return results

def variabilite_donnees_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalize: str = "standard",
    distance_metric: Union[str, Callable] = "euclidean",
    metrics: Union[str, List[str], Callable] = "mse"
) -> Dict:
    """
    Compute data variability statistics.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values if available
    normalize : str
        Normalization method: "none", "standard", "minmax", or "robust"
    distance_metric : Union[str, Callable]
        Distance metric to use: "euclidean", "manhattan", "cosine", "minkowski", or custom callable
    metrics : Union[str, List[str], Callable]
        Metrics to compute: "mse", "mae", "r2", "logloss", or custom callable

    Returns:
    --------
    Dict containing:
        - "result": computed statistics
        - "metrics": calculated metrics
        - "params_used": parameters used
        - "warnings": any warnings generated

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = variabilite_donnees_fit(X, y)
    """
    # Validate inputs
    validated = _validate_inputs(X, y, normalize, distance_metric)
    X_validated = validated["X"]
    y_validated = validated["y"]
    warnings = validated["warnings"]

    # Normalize data
    X_normalized = _normalize_data(X_validated, normalize)

    # Compute distance matrix
    distance_matrix = _compute_distance(X_normalized, y_validated, distance_metric)

    # Compute metrics if y is provided
    metrics_results = {}
    if y_validated is not None:
        # For demonstration, we'll use the mean of each feature as prediction
        y_pred = np.mean(X_normalized, axis=1)
        metrics_results = _compute_metrics(y_validated, y_pred, metrics)

    # Prepare results
    result = {
        "result": {
            "distance_matrix": distance_matrix,
            "normalized_data": X_normalized
        },
        "metrics": metrics_results if y_validated is not None else {},
        "params_used": {
            "normalize": normalize,
            "distance_metric": distance_metric,
            "metrics": metrics
        },
        "warnings": warnings
    }

    return result

################################################################################
# puissance_test
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def puissance_test_fit(
    effect_size: float,
    alpha: float = 0.05,
    power_target: float = 0.8,
    test_type: str = 'z',
    normalization: Optional[str] = None,
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    custom_function: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Calculate the statistical power of a test given various parameters.

    Parameters
    ----------
    effect_size : float
        The expected effect size.
    alpha : float, optional
        Significance level (Type I error probability), by default 0.05.
    power_target : float, optional
        Target statistical power (1 - Type II error probability), by default 0.8.
    test_type : str, optional
        Type of statistical test ('z', 't'), by default 'z'.
    normalization : str, optional
        Type of normalization ('none', 'standard', 'minmax', 'robust'), by default None.
    metric : Union[str, Callable], optional
        Metric to use ('mse', 'mae', 'r2', custom callable), by default 'mse'.
    solver : str, optional
        Solver to use ('closed_form', 'gradient_descent', 'newton'), by default 'closed_form'.
    custom_function : Callable, optional
        Custom function to compute power, by default None.
    **kwargs :
        Additional keyword arguments for specific solvers or metrics.

    Returns
    -------
    Dict
        A dictionary containing the results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> puissance_test_fit(effect_size=0.5, alpha=0.05)
    {
        'result': 0.8,
        'metrics': {'mse': 0.1},
        'params_used': {'effect_size': 0.5, 'alpha': 0.05},
        'warnings': []
    }
    """
    # Validate inputs
    _validate_inputs(effect_size, alpha, power_target)

    # Normalize data if specified
    normalized_effect = _apply_normalization(effect_size, normalization)

    # Choose the appropriate solver
    if custom_function is not None:
        power = _custom_power_solver(normalized_effect, alpha, custom_function)
    else:
        power = _solve_power(
            normalized_effect,
            alpha,
            test_type=test_type,
            solver=solver,
            **kwargs
        )

    # Calculate metrics
    metrics = _calculate_metrics(power, metric)

    return {
        'result': power,
        'metrics': metrics,
        'params_used': {
            'effect_size': effect_size,
            'alpha': alpha,
            'power_target': power_target,
            'test_type': test_type,
            'normalization': normalization,
            'metric': metric,
            'solver': solver
        },
        'warnings': _check_warnings(power, power_target)
    }

def _validate_inputs(effect_size: float, alpha: float, power_target: float) -> None:
    """Validate the input parameters."""
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be between 0 and 1.")
    if not (0 < power_target <= 1):
        raise ValueError("Power target must be between 0 and 1.")
    if effect_size <= 0:
        raise ValueError("Effect size must be positive.")

def _apply_normalization(effect_size: float, normalization: Optional[str]) -> float:
    """Apply the specified normalization to the effect size."""
    if normalization is None:
        return effect_size
    elif normalization == 'standard':
        return (effect_size - np.mean([0, effect_size])) / np.std([0, effect_size])
    elif normalization == 'minmax':
        return (effect_size - 0) / (1 - 0)
    elif normalization == 'robust':
        return (effect_size - np.median([0, effect_size])) / (np.percentile([0, effect_size], 75) - np.percentile([0, effect_size], 25))
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _solve_power(
    effect_size: float,
    alpha: float,
    test_type: str = 'z',
    solver: str = 'closed_form',
    **kwargs
) -> float:
    """Solve for the statistical power using the specified solver."""
    if test_type == 'z':
        return _solve_z_power(effect_size, alpha, solver, **kwargs)
    elif test_type == 't':
        return _solve_t_power(effect_size, alpha, solver, **kwargs)
    else:
        raise ValueError(f"Unknown test type: {test_type}")

def _solve_z_power(
    effect_size: float,
    alpha: float,
    solver: str = 'closed_form',
    **kwargs
) -> float:
    """Solve for the statistical power using a z-test."""
    if solver == 'closed_form':
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - alpha / 2)
        power = 1 - norm.cdf(z_alpha - effect_size)
        return power
    else:
        raise ValueError(f"Unknown solver for z-test: {solver}")

def _solve_t_power(
    effect_size: float,
    alpha: float,
    solver: str = 'closed_form',
    **kwargs
) -> float:
    """Solve for the statistical power using a t-test."""
    if solver == 'closed_form':
        from scipy.stats import t
        df = kwargs.get('df', 30)  # Default degrees of freedom
        t_alpha = t.ppf(1 - alpha / 2, df)
        power = 1 - t.cdf(t_alpha - effect_size, df)
        return power
    else:
        raise ValueError(f"Unknown solver for t-test: {solver}")

def _custom_power_solver(
    effect_size: float,
    alpha: float,
    custom_function: Callable
) -> float:
    """Solve for the statistical power using a custom function."""
    return custom_function(effect_size, alpha)

def _calculate_metrics(
    power: float,
    metric: Union[str, Callable]
) -> Dict:
    """Calculate the specified metrics."""
    metrics = {}
    if metric == 'mse':
        metrics['mse'] = (1 - power) ** 2
    elif metric == 'mae':
        metrics['mae'] = abs(1 - power)
    elif metric == 'r2':
        metrics['r2'] = power ** 2
    elif callable(metric):
        metrics['custom'] = metric(power)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics

def _check_warnings(
    power: float,
    power_target: float
) -> list:
    """Check for any warnings to return."""
    warnings = []
    if power < power_target:
        warnings.append(f"Power ({power}) is below the target ({power_target}).")
    return warnings

################################################################################
# risque_alpha
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    sample_size: int,
    effect_size: float,
    alpha_level: float,
    power_target: float
) -> None:
    """Validate input parameters for risk_alpha computation."""
    if sample_size <= 0:
        raise ValueError("Sample size must be positive.")
    if not (0 < alpha_level < 1):
        raise ValueError("Alpha level must be between 0 and 1.")
    if not (0 < power_target < 1):
        raise ValueError("Power target must be between 0 and 1.")
    if effect_size <= 0:
        raise ValueError("Effect size must be positive.")

def _compute_z_scores(
    effect_size: float,
    sample_size: int
) -> float:
    """Compute z-scores for given effect size and sample size."""
    return effect_size * np.sqrt(sample_size)

def _find_critical_value(
    alpha_level: float,
    tail: str = 'two'
) -> float:
    """Find critical z-value for given alpha level and tail."""
    from scipy.stats import norm
    if tail == 'two':
        return norm.ppf(1 - alpha_level / 2)
    elif tail == 'one':
        return norm.ppf(1 - alpha_level)
    else:
        raise ValueError("Tail must be 'one' or 'two'.")

def _compute_power(
    effect_size: float,
    sample_size: int,
    alpha_level: float,
    tail: str = 'two'
) -> float:
    """Compute statistical power for given parameters."""
    z_critical = _find_critical_value(alpha_level, tail)
    z_score = _compute_z_scores(effect_size, sample_size)
    power = norm.cdf(z_score - z_critical) if tail == 'two' else norm.cdf(z_score)
    return power

def _find_sample_size(
    effect_size: float,
    alpha_level: float,
    power_target: float,
    tail: str = 'two'
) -> int:
    """Find required sample size for given parameters."""
    z_alpha = _find_critical_value(alpha_level, tail)
    z_beta = norm.ppf(power_target)

    if tail == 'two':
        required_z = z_alpha + z_beta
    else:
        required_z = z_alpha - z_beta

    sample_size = (required_z / effect_size) ** 2
    return int(np.ceil(sample_size))

def risque_alpha_fit(
    effect_size: float,
    alpha_level: float = 0.05,
    power_target: float = 0.8,
    tail: str = 'two',
    method: str = 'exact'
) -> Dict[str, Union[float, int, Dict]]:
    """
    Compute risk alpha and related statistical power metrics.

    Parameters:
    -----------
    effect_size : float
        Expected effect size (Cohen's d)
    alpha_level : float, optional
        Significance level (default: 0.05)
    power_target : float, optional
        Desired statistical power (default: 0.8)
    tail : str, optional
        Test tail ('one' or 'two', default: 'two')
    method : str, optional
        Calculation method ('exact' or 'approximate', default: 'exact')

    Returns:
    --------
    dict
        Dictionary containing computation results, metrics and parameters used.

    Example:
    --------
    >>> risque_alpha_fit(effect_size=0.5, alpha_level=0.05)
    {
        'result': {'power': 0.8, 'sample_size': 64},
        'metrics': {'z_score': 4.0, 'critical_value': 1.96},
        'params_used': {'effect_size': 0.5, 'alpha_level': 0.05},
        'warnings': []
    }
    """
    _validate_inputs(1, effect_size, alpha_level, power_target)

    if method == 'exact':
        sample_size = _find_sample_size(effect_size, alpha_level, power_target, tail)
        power = _compute_power(effect_size, sample_size, alpha_level, tail)
    else:
        raise ValueError("Only 'exact' method is currently supported.")

    z_score = _compute_z_scores(effect_size, sample_size)
    critical_value = _find_critical_value(alpha_level, tail)

    return {
        'result': {
            'power': power,
            'sample_size': sample_size
        },
        'metrics': {
            'z_score': z_score,
            'critical_value': critical_value
        },
        'params_used': {
            'effect_size': effect_size,
            'alpha_level': alpha_level,
            'power_target': power_target,
            'tail': tail
        },
        'warnings': []
    }

################################################################################
# risque_beta
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def risque_beta_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalisation: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'gradient_descent',
    tol: float = 1e-4,
    max_iter: int = 1000,
    alpha: float = 0.5,
    beta: float = 1.0,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Estimate the beta risk for statistical power analysis.

    Parameters:
    -----------
    X : np.ndarray
        Input features matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    normalisation : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to evaluate ('mse', 'mae', 'r2', 'logloss') or custom callable.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton').
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    alpha : float, optional
        Regularization parameter for L1 penalty.
    beta : float, optional
        Regularization parameter for L2 penalty.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    **kwargs :
        Additional solver-specific parameters.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, used parameters and warnings.

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = risque_beta_fit(X, y, normalisation='standard', metric='mse')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm = _normalize_data(X, normalisation)
    y_norm = _normalize_target(y, normalisation)

    # Choose solver
    if solver == 'closed_form':
        params = _solve_closed_form(X_norm, y_norm)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(X_norm, y_norm, tol, max_iter, alpha, beta, **kwargs)
    elif solver == 'newton':
        params = _solve_newton(X_norm, y_norm, tol, max_iter, **kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_norm, y_norm, params, metric, custom_metric)

    return {
        'result': metrics,
        'metrics': metrics,
        'params_used': {
            'normalisation': normalisation,
            'metric': metric,
            'solver': solver,
            'tol': tol,
            'max_iter': max_iter,
            'alpha': alpha,
            'beta': beta
        },
        'warnings': _check_warnings(X_norm, y_norm)
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or Inf values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or Inf values")

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize the input features."""
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

def _normalize_target(y: np.ndarray, method: str) -> np.ndarray:
    """Normalize the target values."""
    if method == 'none':
        return y
    elif method in ['standard', 'minmax', 'robust']:
        mean = np.mean(y)
        std = np.std(y)
        return (y - mean) / (std + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve using closed-form solution."""
    XtX = np.dot(X.T, X)
    if not np.allclose(np.linalg.det(XtX), 0):
        XtX_inv = np.linalg.inv(XtX)
        return np.dot(np.dot(XtX_inv, X.T), y)
    else:
        raise ValueError("Matrix is singular")

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int,
    alpha: float,
    beta: float,
    **kwargs
) -> np.ndarray:
    """Solve using gradient descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    learning_rate = kwargs.get('learning_rate', 0.01)

    for _ in range(max_iter):
        gradient = np.dot(X.T, (np.dot(X, params) - y)) / len(y)
        gradient += alpha * np.sign(params) + beta * params
        new_params = params - learning_rate * gradient

        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params

    return params

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int,
    **kwargs
) -> np.ndarray:
    """Solve using Newton's method."""
    n_features = X.shape[1]
    params = np.zeros(n_features)

    for _ in range(max_iter):
        residuals = np.dot(X, params) - y
        gradient = np.dot(X.T, residuals) / len(y)
        hessian = np.dot(X.T, X) / len(y)

        if not np.allclose(np.linalg.det(hessian), 0):
            delta = np.linalg.solve(hessian, -gradient)
            new_params = params + delta

            if np.linalg.norm(new_params - params) < tol:
                break
            params = new_params
        else:
            raise ValueError("Hessian matrix is singular")

    return params

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Compute evaluation metrics."""
    predictions = np.dot(X, params)
    metrics_dict = {}

    if metric == 'mse':
        metrics_dict['mse'] = np.mean((predictions - y) ** 2)
    elif metric == 'mae':
        metrics_dict['mae'] = np.mean(np.abs(predictions - y))
    elif metric == 'r2':
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics_dict['r2'] = 1 - (ss_res / ss_tot)
    elif metric == 'logloss':
        metrics_dict['logloss'] = -np.mean(y * np.log(predictions + 1e-8) + (1 - y) * np.log(1 - predictions + 1e-8))
    elif callable(metric):
        metrics_dict['custom'] = metric(y, predictions)
    elif custom_metric is not None:
        metrics_dict['custom'] = custom_metric(y, predictions)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics_dict

def _check_warnings(X: np.ndarray, y: np.ndarray) -> list:
    """Check for potential warnings."""
    warnings = []
    if X.shape[1] > X.shape[0]:
        warnings.append("Warning: Number of features exceeds number of samples")
    if np.any(np.isclose(X.std(axis=0), 0)):
        warnings.append("Warning: Zero-variance features detected")
    return warnings

################################################################################
# taille_effet
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def taille_effet_fit(
    data: np.ndarray,
    effect_size_func: Callable[[np.ndarray], float],
    normalization: str = 'standard',
    metric: str = 'mse',
    solver: str = 'closed_form',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Calculate the effect size with configurable parameters.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    effect_size_func : Callable[[np.ndarray], float]
        Function to compute the effect size.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust'), by default 'standard'.
    metric : str, optional
        Metric to evaluate ('mse', 'mae', 'r2', 'logloss'), by default 'mse'.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent'), by default 'closed_form'.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function, by default None.
    tol : float, optional
        Tolerance for convergence, by default 1e-6.
    max_iter : int, optional
        Maximum number of iterations, by default 1000.

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], str]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.random.randn(100, 2)
    >>> def effect_size_func(x): return np.mean(x[:, 1] - x[:, 0])
    >>> result = taille_effet_fit(data, effect_size_func)
    """
    # Validate inputs
    _validate_inputs(data)

    # Normalize data
    normalized_data = _normalize_data(data, normalization)

    # Compute effect size
    effect_size = effect_size_func(normalized_data)

    # Choose metric function
    if custom_metric is not None:
        metric_func = custom_metric
    else:
        metric_func = _get_metric_function(metric)

    # Solve for parameters
    params = _solve_parameters(normalized_data, effect_size_func, solver, tol, max_iter)

    # Compute metrics
    metrics = _compute_metrics(normalized_data, effect_size_func, metric_func)

    # Prepare output
    result = {
        'result': effect_size,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'solver': solver
        },
        'warnings': []
    }

    return result

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values.")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values.")

def _normalize_data(
    data: np.ndarray,
    method: str = 'standard'
) -> np.ndarray:
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

def _get_metric_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
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
    return 1 - (ss_res / (ss_tot + 1e-8))

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute log loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _solve_parameters(
    data: np.ndarray,
    effect_size_func: Callable[[np.ndarray], float],
    solver: str = 'closed_form',
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, float]:
    """Solve for parameters using specified solver."""
    if solver == 'closed_form':
        return _solve_closed_form(data, effect_size_func)
    elif solver == 'gradient_descent':
        return _solve_gradient_descent(data, effect_size_func, tol, max_iter)
    elif solver == 'newton':
        return _solve_newton(data, effect_size_func, tol, max_iter)
    elif solver == 'coordinate_descent':
        return _solve_coordinate_descent(data, effect_size_func, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _solve_closed_form(data: np.ndarray, effect_size_func: Callable[[np.ndarray], float]) -> Dict[str, float]:
    """Solve parameters using closed form solution."""
    # Placeholder for actual implementation
    return {'param1': 0.5, 'param2': 0.5}

def _solve_gradient_descent(
    data: np.ndarray,
    effect_size_func: Callable[[np.ndarray], float],
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, float]:
    """Solve parameters using gradient descent."""
    # Placeholder for actual implementation
    return {'param1': 0.5, 'param2': 0.5}

def _solve_newton(
    data: np.ndarray,
    effect_size_func: Callable[[np.ndarray], float],
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, float]:
    """Solve parameters using Newton's method."""
    # Placeholder for actual implementation
    return {'param1': 0.5, 'param2': 0.5}

def _solve_coordinate_descent(
    data: np.ndarray,
    effect_size_func: Callable[[np.ndarray], float],
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, float]:
    """Solve parameters using coordinate descent."""
    # Placeholder for actual implementation
    return {'param1': 0.5, 'param2': 0.5}

def _compute_metrics(
    data: np.ndarray,
    effect_size_func: Callable[[np.ndarray], float],
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Compute metrics for the effect size calculation."""
    y_pred = np.array([effect_size_func(data) for _ in range(len(data))])
    return {'metric_value': metric_func(data[:, 0], y_pred)}

################################################################################
# analyse_puissance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def analyse_puissance_fit(
    effect_size: float,
    alpha: float = 0.05,
    power_target: float = 0.8,
    sample_size_range: Optional[tuple] = None,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: str = 't_test',
    solver: str = 'closed_form',
    custom_metric: Optional[Callable] = None,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, Union[Dict, float]]:
    """
    Calculate statistical power analysis for given parameters.

    Parameters
    ----------
    effect_size : float
        The expected effect size.
    alpha : float, optional
        Significance level (Type I error probability), by default 0.05.
    power_target : float, optional
        Target statistical power (1 - Type II error probability), by default 0.8.
    sample_size_range : tuple, optional
        Range of sample sizes to consider (min, max), by default None.
    normalizer : Callable[[np.ndarray], np.ndarray], optional
        Normalization function, by default identity.
    metric : str, optional
        Metric to use for power calculation ('t_test', 'z_test'), by default 't_test'.
    solver : str, optional
        Solver method ('closed_form', 'iterative'), by default 'closed_form'.
    custom_metric : Callable, optional
        Custom metric function if needed.
    tol : float, optional
        Tolerance for convergence, by default 1e-6.
    max_iter : int, optional
        Maximum iterations for iterative solver, by default 1000.

    Returns
    -------
    Dict[str, Union[Dict, float]]
        Dictionary containing results, metrics, parameters used and warnings.

    Examples
    --------
    >>> analyse_puissance_fit(effect_size=0.5, alpha=0.05)
    """
    # Validate inputs
    _validate_inputs(effect_size, alpha, power_target)

    # Normalize effect size if needed
    normalized_effect = normalizer(np.array([effect_size]))[0]

    # Choose solver based on parameters
    if solver == 'closed_form':
        result = _solve_closed_form(normalized_effect, alpha, power_target)
    elif solver == 'iterative':
        result = _solve_iterative(normalized_effect, alpha, power_target,
                                tol=tol, max_iter=max_iter)
    else:
        raise ValueError("Invalid solver specified")

    # Calculate metrics
    metrics = _calculate_metrics(result, metric, custom_metric)

    return {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "effect_size": effect_size,
            "alpha": alpha,
            "power_target": power_target,
            "normalizer": normalizer.__name__ if hasattr(normalizer, '__name__') else "custom",
            "metric": metric,
            "solver": solver
        },
        "warnings": _check_warnings(result)
    }

def _validate_inputs(effect_size: float, alpha: float, power_target: float) -> None:
    """Validate input parameters."""
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be between 0 and 1")
    if not (0 < power_target <= 1):
        raise ValueError("Power target must be between 0 and 1")
    if effect_size <= 0:
        raise ValueError("Effect size must be positive")

def _solve_closed_form(effect_size: float, alpha: float, power_target: float) -> Dict[str, float]:
    """Closed form solution for power analysis."""
    # This is a simplified example - actual implementation would use proper statistical formulas
    z_alpha = _z_score(alpha / 2)
    z_beta = _z_score(1 - power_target)

    sample_size = ((z_alpha + z_beta) / effect_size) ** 2
    power_achieved = _calculate_power(effect_size, sample_size, alpha)

    return {
        "sample_size": sample_size,
        "power_achieved": power_achieved
    }

def _solve_iterative(effect_size: float, alpha: float, power_target: float,
                   tol: float = 1e-6, max_iter: int = 1000) -> Dict[str, float]:
    """Iterative solution for power analysis."""
    sample_size = 10
    for _ in range(max_iter):
        power_achieved = _calculate_power(effect_size, sample_size, alpha)
        if abs(power_achieved - power_target) < tol:
            break
        sample_size += 1

    return {
        "sample_size": sample_size,
        "power_achieved": power_achieved
    }

def _calculate_power(effect_size: float, sample_size: float, alpha: float) -> float:
    """Calculate achieved power for given parameters."""
    # Simplified example - actual implementation would use proper statistical formulas
    z_alpha = _z_score(alpha / 2)
    non_centrality = effect_size * np.sqrt(sample_size)
    power = _z_to_pvalue(z_alpha + non_centrality, lower_tail=False)
    return power

def _z_score(p: float) -> float:
    """Convert p-value to z-score."""
    from scipy.stats import norm
    return norm.ppf(p)

def _z_to_pvalue(z: float, lower_tail: bool = True) -> float:
    """Convert z-score to p-value."""
    from scipy.stats import norm
    return norm.cdf(z) if lower_tail else 1 - norm.cdf(z)

def _calculate_metrics(result: Dict[str, float], metric: str,
                     custom_metric: Optional[Callable] = None) -> Dict[str, float]:
    """Calculate various metrics based on results."""
    metrics = {}

    if metric == 't_test':
        # Example t-test metrics
        metrics['t_statistic'] = result['sample_size'] * 0.1  # Simplified
        metrics['p_value'] = _z_to_pvalue(metrics['t_statistic'])
    elif metric == 'z_test':
        # Example z-test metrics
        metrics['z_statistic'] = result['sample_size'] * 0.1  # Simplified
        metrics['p_value'] = _z_to_pvalue(metrics['z_statistic'])
    elif custom_metric is not None:
        metrics.update(custom_metric(result))

    return metrics

def _check_warnings(result: Dict[str, float]) -> list:
    """Check for any warnings in the results."""
    warnings = []

    if result['power_achieved'] < 0.7:
        warnings.append("Achieved power is below recommended threshold of 0.7")

    if result['sample_size'] > 1000:
        warnings.append("Large sample size may indicate issues with effect size or parameters")

    return warnings

################################################################################
# calcul_puissance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def calcul_puissance_fit(
    effect_size: float,
    alpha: float = 0.05,
    power_target: float = 0.8,
    sample_size_range: Optional[tuple] = None,
    method: str = 'exact',
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: str = 't_test',
    solver: str = 'closed_form',
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, Union[float, Dict]]:
    """
    Calculate statistical power for a given effect size and sample size.

    Parameters
    ----------
    effect_size : float
        The expected effect size.
    alpha : float, optional
        Significance level (Type I error probability), by default 0.05.
    power_target : float, optional
        Target statistical power (1 - Type II error probability), by default 0.8.
    sample_size_range : tuple, optional
        Range of sample sizes to consider (n_min, n_max), by default None.
    method : str, optional
        Method for power calculation ('exact', 'approximate'), by default 'exact'.
    normalizer : Callable, optional
        Function to normalize data, by default identity function.
    metric : str, optional
        Metric for power calculation ('t_test', 'z_test'), by default 't_test'.
    solver : str, optional
        Solver method ('closed_form', 'newton'), by default 'closed_form'.
    tol : float, optional
        Tolerance for convergence, by default 1e-6.
    max_iter : int, optional
        Maximum number of iterations, by default 1000.

    Returns
    -------
    Dict[str, Union[float, Dict]]
        Dictionary containing:
        - 'result': Calculated power or required sample size
        - 'metrics': Dictionary of performance metrics
        - 'params_used': Parameters used in the calculation
        - 'warnings': Any warnings generated during computation

    Example
    -------
    >>> calcul_puissance_fit(effect_size=0.5, alpha=0.05)
    {
        'result': 0.78,
        'metrics': {'convergence_iterations': 5, 'final_error': 1e-7},
        'params_used': {'alpha': 0.05, 'method': 'exact'},
        'warnings': []
    }
    """
    # Validate inputs
    _validate_inputs(effect_size, alpha, power_target)

    # Normalize effect size if needed
    normalized_effect = normalizer(np.array([effect_size]))[0]

    # Choose appropriate calculation method
    if sample_size_range is not None:
        result = _calculate_power_for_sample_range(
            normalized_effect, alpha, power_target,
            sample_size_range, method, metric, solver, tol, max_iter
        )
    else:
        result = _calculate_power(
            normalized_effect, alpha, power_target,
            method, metric, solver, tol, max_iter
        )

    # Prepare output dictionary
    output = {
        'result': result,
        'metrics': {'convergence_iterations': 0, 'final_error': 0.0},
        'params_used': {
            'effect_size': effect_size,
            'alpha': alpha,
            'power_target': power_target,
            'method': method,
            'metric': metric,
            'solver': solver
        },
        'warnings': []
    }

    return output

def _validate_inputs(effect_size: float, alpha: float, power_target: float) -> None:
    """Validate input parameters."""
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be between 0 and 1")
    if not (0 < power_target <= 1):
        raise ValueError("Power target must be between 0 and 1")
    if effect_size <= 0:
        raise ValueError("Effect size must be positive")

def _calculate_power(
    effect_size: float,
    alpha: float,
    power_target: float,
    method: str,
    metric: str,
    solver: str,
    tol: float,
    max_iter: int
) -> float:
    """Calculate statistical power for given parameters."""
    if method == 'exact':
        return _exact_power_calculation(effect_size, alpha)
    elif method == 'approximate':
        return _approximate_power_calculation(effect_size, alpha)
    else:
        raise ValueError(f"Unknown method: {method}")

def _calculate_power_for_sample_range(
    effect_size: float,
    alpha: float,
    power_target: float,
    sample_range: tuple,
    method: str,
    metric: str,
    solver: str,
    tol: float,
    max_iter: int
) -> Dict[str, Union[float, list]]:
    """Calculate power for a range of sample sizes."""
    n_min, n_max = sample_range
    sample_sizes = np.arange(n_min, n_max + 1)
    powers = []

    for n in sample_sizes:
        power = _calculate_power(effect_size, alpha, power_target,
                                method, metric, solver, tol, max_iter)
        powers.append(power)

    return {
        'sample_sizes': sample_sizes.tolist(),
        'powers': powers,
        'required_sample_size': _find_required_sample_size(powers, power_target)
    }

def _exact_power_calculation(effect_size: float, alpha: float) -> float:
    """Exact calculation of statistical power."""
    # Implementation of exact power calculation
    return 0.8  # Placeholder

def _approximate_power_calculation(effect_size: float, alpha: float) -> float:
    """Approximate calculation of statistical power."""
    # Implementation of approximate power calculation
    return 0.75  # Placeholder

def _find_required_sample_size(powers: list, power_target: float) -> int:
    """Find the smallest sample size that meets the power target."""
    for i, power in enumerate(powers):
        if power >= power_target:
            return i + 1
    raise ValueError("No sample size meets the power target")

################################################################################
# estimation_parametres
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def estimation_parametres_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalisation: str = "standard",
    metric: Union[str, Callable] = "mse",
    distance: Optional[Union[str, Callable]] = None,
    solver: str = "closed_form",
    regularisation: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Estimate statistical parameters with configurable options.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    normalisation : str, optional
        Normalization method: "none", "standard", "minmax", or "robust".
    metric : str or callable, optional
        Metric to optimize: "mse", "mae", "r2", or custom callable.
    distance : str or callable, optional
        Distance metric for some solvers: "euclidean", "manhattan", etc.
    solver : str, optional
        Solver method: "closed_form", "gradient_descent", etc.
    regularisation : str, optional
        Regularization type: "none", "l1", "l2", or "elasticnet".
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    custom_distance : callable, optional
        Custom distance function if not using built-in distances.

    Returns:
    --------
    Dict containing:
        - "result": Estimated parameters.
        - "metrics": Computed metrics.
        - "params_used": Parameters used in the estimation.
        - "warnings": Any warnings encountered.

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = estimation_parametres_fit(X, y, normalisation="standard", metric="mse")
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_norm = _apply_normalisation(X, normalisation)

    # Select solver and compute parameters
    if solver == "closed_form":
        params = _solve_closed_form(X_norm, y)
    elif solver == "gradient_descent":
        params = _solve_gradient_descent(
            X_norm, y,
            metric=metric,
            distance=distance,
            tol=tol,
            max_iter=max_iter
        )
    else:
        raise ValueError(f"Unsupported solver: {solver}")

    # Apply regularisation if required
    if regularisation:
        params = _apply_regularisation(params, X_norm, y, regularisation)

    # Compute metrics
    metrics = _compute_metrics(X_norm, y, params, metric=metric)

    # Prepare output
    return {
        "result": params,
        "metrics": metrics,
        "params_used": {
            "normalisation": normalisation,
            "metric": metric,
            "solver": solver,
            "regularisation": regularisation
        },
        "warnings": []
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

def _apply_normalisation(X: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalisation to input data."""
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
        raise ValueError(f"Unsupported normalisation method: {method}")

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve parameters using closed-form solution."""
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    metric: Union[str, Callable] = "mse",
    distance: Optional[Union[str, Callable]] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Solve parameters using gradient descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    prev_loss = float('inf')

    for _ in range(max_iter):
        gradient = _compute_gradient(X, y, params, metric=metric)
        params -= 0.01 * gradient

        current_loss = _compute_metric(X, y, params, metric=metric)
        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return params

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable] = "mse"
) -> np.ndarray:
    """Compute gradient for given metric."""
    if callable(metric):
        return _compute_custom_gradient(X, y, params, metric)
    elif metric == "mse":
        return 2 * X.T @ (X @ params - y) / len(y)
    else:
        raise ValueError(f"Unsupported metric for gradient: {metric}")

def _compute_metric(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable] = "mse"
) -> float:
    """Compute specified metric."""
    if callable(metric):
        return _compute_custom_metric(X, y, params, metric)
    elif metric == "mse":
        return np.mean((X @ params - y) ** 2)
    elif metric == "mae":
        return np.mean(np.abs(X @ params - y))
    elif metric == "r2":
        ss_res = np.sum((y - X @ params) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / (ss_tot + 1e-8)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable] = "mse"
) -> Dict:
    """Compute all relevant metrics."""
    if callable(metric):
        custom_value = _compute_custom_metric(X, y, params, metric)
    else:
        custom_value = None

    return {
        "mse": np.mean((X @ params - y) ** 2),
        "mae": np.mean(np.abs(X @ params - y)),
        "r2": 1 - (np.sum((y - X @ params) ** 2) / np.sum((y - np.mean(y)) ** 2)),
        "custom": custom_value
    }

def _apply_regularisation(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply specified regularisation."""
    if method == "l1":
        return _apply_l1_regularisation(params, X, y)
    elif method == "l2":
        return _apply_l2_regularisation(params, X, y)
    elif method == "elasticnet":
        return _apply_elasticnet_regularisation(params, X, y)
    else:
        raise ValueError(f"Unsupported regularisation method: {method}")

def _apply_l1_regularisation(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """Apply L1 regularisation."""
    # Simplified implementation - in practice would use coordinate descent
    return np.sign(params) * np.maximum(np.abs(params) - 0.1, 0)

def _apply_l2_regularisation(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """Apply L2 regularisation."""
    # Simplified implementation - in practice would use ridge regression
    return np.linalg.pinv(X.T @ X + 0.1 * np.eye(X.shape[1])) @ X.T @ y

def _apply_elasticnet_regularisation(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """Apply elastic net regularisation."""
    # Simplified implementation - in practice would combine L1 and L2
    return np.linalg.pinv(X.T @ X + 0.1 * np.eye(X.shape[1])) @ (X.T @ y - 0.5 * params)

def _compute_custom_metric(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric_func: Callable
) -> float:
    """Compute custom metric."""
    return metric_func(X @ params, y)

def _compute_custom_gradient(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric_func: Callable
) -> np.ndarray:
    """Compute gradient for custom metric."""
    # This is a simplified implementation
    epsilon = 1e-8
    gradient = np.zeros_like(params)

    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += epsilon
        loss_plus = metric_func(X @ params_plus, y)

        params_minus = params.copy()
        params_minus[i] -= epsilon
        loss_minus = metric_func(X @ params_minus, y)

        gradient[i] = (loss_plus - loss_minus) / (2 * epsilon)

    return gradient

################################################################################
# hypotheses_statistiques
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def hypotheses_statistiques_fit(
    data: np.ndarray,
    hypothesis_func: Callable[[np.ndarray], float],
    null_hypothesis_value: float,
    alternative: str = 'two_sided',
    normalization: Optional[str] = None,
    metric: str = 'p_value',
    solver: str = 'z_test',
    custom_metric: Optional[Callable[[np.ndarray, float], float]] = None,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, Any]:
    """
    Compute statistical hypothesis testing results.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    hypothesis_func : Callable[[np.ndarray], float]
        Function to compute the test statistic from data.
    null_hypothesis_value : float
        Value under the null hypothesis.
    alternative : str, optional
        Type of alternative hypothesis ('two_sided', 'less', 'greater').
    normalization : str, optional
        Type of normalization ('none', 'standard', 'minmax').
    metric : str, optional
        Metric to compute ('p_value', 'statistic').
    solver : str, optional
        Solver method ('z_test', 't_test').
    custom_metric : Callable, optional
        Custom metric function.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, hypothesis_func, null_hypothesis_value)

    # Normalize data if specified
    normalized_data = _normalize_data(data, normalization)

    # Compute test statistic
    statistic = hypothesis_func(normalized_data)
    params_used = {
        'normalization': normalization,
        'metric': metric,
        'solver': solver,
        'tol': tol,
        'max_iter': max_iter
    }

    # Compute results based on solver and alternative
    if solver == 'z_test':
        result = _compute_z_test(statistic, null_hypothesis_value, alternative)
    elif solver == 't_test':
        result = _compute_t_test(statistic, null_hypothesis_value, alternative)
    else:
        raise ValueError("Unsupported solver method.")

    # Compute metrics
    metrics = _compute_metrics(result, statistic, null_hypothesis_value, custom_metric)

    # Check for warnings
    warnings = _check_warnings(data, normalized_data, statistic)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

def _validate_inputs(
    data: np.ndarray,
    hypothesis_func: Callable[[np.ndarray], float],
    null_hypothesis_value: float
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if not callable(hypothesis_func):
        raise TypeError("hypothesis_func must be a callable.")
    if np.isnan(null_hypothesis_value) or not np.isfinite(null_hypothesis_value):
        raise ValueError("null_hypothesis_value must be finite.")

def _normalize_data(
    data: np.ndarray,
    normalization: Optional[str]
) -> np.ndarray:
    """Normalize data based on specified method."""
    if normalization is None:
        return data
    elif normalization == 'standard':
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / (std + 1e-8)
    elif normalization == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val + 1e-8)
    else:
        raise ValueError("Unsupported normalization method.")

def _compute_z_test(
    statistic: float,
    null_hypothesis_value: float,
    alternative: str
) -> Dict[str, Any]:
    """Compute z-test results."""
    z_score = (statistic - null_hypothesis_value) / np.sqrt(np.var(statistic))
    if alternative == 'two_sided':
        p_value = 2 * (1 - _standard_normal_cdf(np.abs(z_score)))
    elif alternative == 'less':
        p_value = _standard_normal_cdf(z_score)
    elif alternative == 'greater':
        p_value = 1 - _standard_normal_cdf(z_score)
    else:
        raise ValueError("Unsupported alternative hypothesis.")
    return {
        'statistic': z_score,
        'p_value': p_value
    }

def _compute_t_test(
    statistic: float,
    null_hypothesis_value: float,
    alternative: str
) -> Dict[str, Any]:
    """Compute t-test results."""
    t_score = (statistic - null_hypothesis_value) / np.sqrt(np.var(statistic) / len(statistic))
    df = len(statistic) - 1
    if alternative == 'two_sided':
        p_value = 2 * (1 - _t_distribution_cdf(np.abs(t_score), df))
    elif alternative == 'less':
        p_value = _t_distribution_cdf(t_score, df)
    elif alternative == 'greater':
        p_value = 1 - _t_distribution_cdf(t_score, df)
    else:
        raise ValueError("Unsupported alternative hypothesis.")
    return {
        'statistic': t_score,
        'p_value': p_value
    }

def _compute_metrics(
    result: Dict[str, Any],
    statistic: float,
    null_hypothesis_value: float,
    custom_metric: Optional[Callable[[np.ndarray, float], float]]
) -> Dict[str, Any]:
    """Compute additional metrics."""
    metrics = {
        'statistic_value': statistic,
        'null_hypothesis_value': null_hypothesis_value
    }
    if custom_metric is not None:
        metrics['custom_metric'] = custom_metric(statistic, null_hypothesis_value)
    return metrics

def _check_warnings(
    data: np.ndarray,
    normalized_data: np.ndarray,
    statistic: float
) -> List[str]:
    """Check for potential warnings."""
    warnings = []
    if np.any(np.isnan(data)):
        warnings.append("Input data contains NaN values.")
    if np.any(np.isinf(normalized_data)):
        warnings.append("Normalized data contains infinite values.")
    if np.isnan(statistic):
        warnings.append("Computed statistic is NaN.")
    return warnings

def _standard_normal_cdf(x: float) -> float:
    """Compute the CDF of the standard normal distribution."""
    return 0.5 * (1 + np.sign(x) * np.sqrt(2 / np.pi) *
                  (np.abs(x) * np.exp(-x**2 / 2) +
                   (1/3) * x * (1 - x**2 / 5) +
                   (7/945) * x**3 * (1 - 3*x**2 / 7) +
                   (19/6531) * x**5))

def _t_distribution_cdf(x: float, df: int) -> float:
    """Compute the CDF of the t-distribution."""
    if df <= 0:
        raise ValueError("Degrees of freedom must be positive.")
    return _incomplete_beta((df + 1) / 2, df / 2,
                           (df + x**2) / (x**2 + df))

def _incomplete_beta(a: float, b: float, x: float) -> float:
    """Compute the incomplete beta function."""
    if x <= 0 or x >= 1:
        return np.where(x < 0, 0.0, 1.0)
    return _regularized_beta(a, b, x)

def _regularized_beta(a: float, b: float, x: float) -> float:
    """Compute the regularized incomplete beta function."""
    if x == 0 or x == 1:
        return np.where(x < 0.5, 0.0, 1.0)
    return _beta_inc(a, b, x)

def _beta_inc(a: float, b: float, x: float) -> float:
    """Compute the beta function."""
    if a <= 0 or b <= 0:
        raise ValueError("Parameters must be positive.")
    return np.exp(_lgamma(a + b) - _lgamma(a) - _lgamma(b) +
                  np.log(x)**a + np.log(1 - x)**b)

def _lgamma(x: float) -> float:
    """Compute the log gamma function."""
    return np.log(np.abs(_gamma(x)))

def _gamma(x: float) -> float:
    """Compute the gamma function."""
    if x < 0.5:
        return np.pi / (np.sin(np.pi * x) * _gamma(1 - x))
    else:
        return np.sqrt(2 * np.pi / x) * (x / np.e)**x * _gamma_approx(x)

def _gamma_approx(x: float) -> float:
    """Approximate the gamma function."""
    coeffs = [
        76.18009172947146, -86.50532032941677,
        24.01409824083091, -1.231739572450155,
        0.1208650973866179e-2, -0.5395239384953e-5
    ]
    y = x - 1
    result = coeffs[0]
    for c in coeffs[1:]:
        result = result * y + c
    return 1 / (result * x)

################################################################################
# tests_statistiques
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Any

def tests_statistiques_fit(
    X: np.ndarray,
    y: np.ndarray,
    test_type: str = 't_test',
    normalization: Optional[str] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'closed_form',
    alpha: float = 0.05,
    custom_test_func: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform statistical tests with configurable parameters.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    test_type : str, optional
        Type of statistical test ('t_test', 'anova', etc.).
    normalization : str or None, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to evaluate ('mse', 'mae', 'r2', etc.) or custom callable.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', etc.).
    alpha : float, optional
        Significance level for the test.
    custom_test_func : callable or None, optional
        Custom test function to override default behavior.
    **kwargs :
        Additional keyword arguments for specific tests.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_norm = _apply_normalization(X, normalization) if normalization else X

    # Prepare test parameters
    params = {
        'test_type': test_type,
        'normalization': normalization,
        'metric': metric,
        'solver': solver,
        'alpha': alpha
    }

    # Perform the test
    if custom_test_func is not None:
        result = custom_test_func(X_norm, y, **kwargs)
    else:
        if test_type == 't_test':
            result = _perform_t_test(X_norm, y, solver=solver, **kwargs)
        elif test_type == 'anova':
            result = _perform_anova(X_norm, y, solver=solver, **kwargs)
        else:
            raise ValueError(f"Unsupported test type: {test_type}")

    # Calculate metrics
    metrics = _calculate_metrics(y, result['predictions'], metric)

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

    return output

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to input data."""
    if method == 'standard':
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        X_norm = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    elif method == 'none':
        X_norm = X
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    return X_norm

def _perform_t_test(X: np.ndarray, y: np.ndarray, solver: str = 'closed_form', **kwargs) -> Dict[str, Any]:
    """Perform t-test."""
    # Example implementation - replace with actual test logic
    if solver == 'closed_form':
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
    else:
        raise ValueError(f"Unsupported solver: {solver}")

    predictions = X @ beta
    residuals = y - predictions

    # Calculate t-statistic (simplified example)
    t_stat = np.mean(residuals) / (np.std(residuals, ddof=1) / np.sqrt(len(y)))

    return {
        'statistic': t_stat,
        'p_value': _calculate_p_value(t_stat, df=len(y)-1),
        'predictions': predictions,
        'coefficients': beta
    }

def _perform_anova(X: np.ndarray, y: np.ndarray, solver: str = 'closed_form', **kwargs) -> Dict[str, Any]:
    """Perform ANOVA."""
    # Example implementation - replace with actual test logic
    if solver == 'closed_form':
        groups = np.unique(X, axis=0)
        group_means = [np.mean(y[X == group], axis=0) for group in groups]
        overall_mean = np.mean(y)
    else:
        raise ValueError(f"Unsupported solver: {solver}")

    # Calculate F-statistic (simplified example)
    between_group_variance = np.sum([len(y[X == group]) * (group_mean - overall_mean)**2 for group, group_mean in zip(groups, group_means)])
    within_group_variance = np.sum((y - overall_mean)**2)

    f_stat = (between_group_variance / (len(groups) - 1)) / (within_group_variance / (len(y) - len(groups)))

    return {
        'statistic': f_stat,
        'p_value': _calculate_p_value(f_stat, dfn=len(groups)-1, dfd=len(y)-len(groups)),
        'group_means': group_means,
        'overall_mean': overall_mean
    }

def _calculate_p_value(statistic: float, df: int = 1, dfn: Optional[int] = None, dfd: Optional[int] = None) -> float:
    """Calculate p-value based on test statistic."""
    if dfn is not None and dfd is not None:
        # F-distribution
        from scipy.stats import f
        p_value = 1 - f.cdf(statistic, dfn, dfd)
    else:
        # T-distribution
        from scipy.stats import t
        p_value = 2 * (1 - t.cdf(np.abs(statistic), df))
    return p_value

def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, metric: Union[str, Callable]) -> Dict[str, float]:
    """Calculate specified metrics."""
    metrics = {}

    if isinstance(metric, str):
        if metric == 'mse':
            metrics['mse'] = np.mean((y_true - y_pred)**2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred)**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            metrics['r2'] = 1 - (ss_res / ss_tot)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    elif callable(metric):
        metrics['custom'] = metric(y_true, y_pred)
    else:
        raise ValueError("Metric must be a string or callable")

    return metrics

################################################################################
# modèles_statistiques
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def modèles_statistiques_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalisation: str = 'standard',
    métrique: Union[str, Callable] = 'mse',
    distance: Optional[Union[str, Callable]] = None,
    solveur: str = 'closed_form',
    régularisation: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fonction principale pour l'estimation des modèles statistiques.

    Parameters
    ----------
    X : np.ndarray
        Matrice de caractéristiques (n_samples, n_features).
    y : np.ndarray
        Vecteur cible (n_samples,).
    normalisation : str, optional
        Type de normalisation ('none', 'standard', 'minmax', 'robust').
    métrique : str or Callable, optional
        Métrique d'évaluation ('mse', 'mae', 'r2', 'logloss') ou callable.
    distance : str or Callable, optional
        Distance pour les modèles ('euclidean', 'manhattan', 'cosine', 'minkowski') ou callable.
    solveur : str, optional
        Solveur utilisé ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    régularisation : str, optional
        Type de régularisation ('none', 'l1', 'l2', 'elasticnet').
    tol : float, optional
        Tolérance pour la convergence.
    max_iter : int, optional
        Nombre maximal d'itérations.
    custom_metric : Callable, optional
        Fonction de métrique personnalisée.
    custom_distance : Callable, optional
        Fonction de distance personnalisée.

    Returns
    -------
    Dict
        Dictionnaire contenant les résultats, métriques, paramètres utilisés et avertissements.
    """
    # Validation des entrées
    _validate_inputs(X, y)

    # Normalisation des données
    X_norm = _apply_normalisation(X, normalisation)

    # Choix de la métrique
    metric_func = _get_metric_function(métrique, custom_metric)

    # Choix de la distance si nécessaire
    if distance is not None:
        distance_func = _get_distance_function(distance, custom_distance)
    else:
        distance_func = None

    # Estimation des paramètres
    params, warnings = _estimate_parameters(
        X_norm, y,
        solveur=solveur,
        régularisation=réregularisation,
        tol=tol,
        max_iter=max_iter,
        distance_func=distance_func
    )

    # Calcul des métriques
    metrics = _compute_metrics(X_norm, y, params, metric_func)

    # Retour des résultats
    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalisation': normalisation,
            'métrique': métrique if isinstance(métrique, str) else 'custom',
            'distance': distance if isinstance(distance, str) else 'custom',
            'solveur': solveur,
            'régularisation': régularisation
        },
        'warnings': warnings
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validation des entrées."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X doit être une matrice (n_samples, n_features) et y un vecteur (n_samples,).")
    if X.shape[0] != y.shape[0]:
        raise ValueError("Le nombre d'échantillons dans X et y doit être identique.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Les données contiennent des valeurs NaN.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Les données contiennent des valeurs infinies.")

def _apply_normalisation(X: np.ndarray, normalisation: str) -> np.ndarray:
    """Application de la normalisation."""
    if normalisation == 'none':
        return X
    elif normalisation == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalisation == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalisation == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError(f"Normalisation inconnue: {normalisation}")

def _get_metric_function(metric: Union[str, Callable], custom_metric: Optional[Callable] = None) -> Callable:
    """Récupération de la fonction de métrique."""
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
            raise ValueError(f"Métrique inconnue: {metric}")
    elif callable(metric):
        return metric
    elif custom_metric is not None:
        return custom_metric
    else:
        raise ValueError("Aucune métrique valide fournie.")

def _get_distance_function(distance: Union[str, Callable], custom_distance: Optional[Callable] = None) -> Callable:
    """Récupération de la fonction de distance."""
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
            raise ValueError(f"Distance inconnue: {distance}")
    elif callable(distance):
        return distance
    elif custom_distance is not None:
        return custom_distance
    else:
        raise ValueError("Aucune distance valide fournie.")

def _estimate_parameters(
    X: np.ndarray,
    y: np.ndarray,
    solveur: str,
    régularisation: Optional[str],
    tol: float,
    max_iter: int,
    distance_func: Optional[Callable] = None
) -> tuple:
    """Estimation des paramètres du modèle."""
    warnings = []

    if solveur == 'closed_form':
        params = _closed_form_solution(X, y, régularisation)
    elif solveur == 'gradient_descent':
        params = _gradient_descent(X, y, tol, max_iter, régularisation)
    elif solveur == 'newton':
        params = _newton_method(X, y, tol, max_iter)
    elif solveur == 'coordinate_descent':
        params = _coordinate_descent(X, y, tol, max_iter)
    else:
        raise ValueError(f"Solveur inconnu: {solveur}")

    return params, warnings

def _compute_metrics(X: np.ndarray, y: np.ndarray, params: np.ndarray, metric_func: Callable) -> Dict:
    """Calcul des métriques."""
    y_pred = X @ params
    return {metric_func.__name__: metric_func(y, y_pred)}

def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcul de l'erreur quadratique moyenne."""
    return np.mean((y_true - y_pred) ** 2)

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcul de l'erreur absolue moyenne."""
    return np.mean(np.abs(y_true - y_pred))

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcul du coefficient de détermination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcul de la logloss."""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calcul de la distance euclidienne."""
    return np.linalg.norm(x - y)

def _manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calcul de la distance de Manhattan."""
    return np.sum(np.abs(x - y))

def _cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calcul de la distance cosinus."""
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def _closed_form_solution(X: np.ndarray, y: np.ndarray, régularisation: Optional[str]) -> np.ndarray:
    """Résolution en forme fermée."""
    if régularisation == 'l1':
        raise NotImplementedError("La résolution en forme fermée avec L1 n'est pas implémentée.")
    elif régularisation == 'l2':
        return np.linalg.inv(X.T @ X + 1e-6 * np.eye(X.shape[1])) @ X.T @ y
    else:
        return np.linalg.inv(X.T @ X) @ X.T @ y

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int,
    régularisation: Optional[str]
) -> np.ndarray:
    """Descente de gradient."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    learning_rate = 0.01
    for _ in range(max_iter):
        gradient = X.T @ (X @ params - y) / len(y)
        if régularisation == 'l2':
            gradient += 1e-4 * params
        elif régularisation == 'l1':
            gradient += 1e-4 * np.sign(params)
        params -= learning_rate * gradient
        if np.linalg.norm(gradient) < tol:
            break
    return params

def _newton_method(X: np.ndarray, y: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
    """Méthode de Newton."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = X.T @ (X @ params - y)
        hessian = X.T @ X
        params -= np.linalg.inv(hessian) @ gradient
        if np.linalg.norm(gradient) < tol:
            break
    return params

def _coordinate_descent(X: np.ndarray, y: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
    """Descente de coordonnées."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            params_j_old = params[j]
            params[j] = np.sum((y - np.dot(X, params) + params_j_old * X_j) * X_j) / np.sum(X_j ** 2)
        if np.linalg.norm(params - params_old) < tol:
            break
        params_old = params.copy()
    return params
