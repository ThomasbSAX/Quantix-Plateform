"""
Quantix – Module tests_independance
Généré automatiquement
Date: 2026-01-09
"""

################################################################################
# test_chi2
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(observed: np.ndarray,
                   expected: Optional[np.ndarray] = None,
                   normalize: str = 'none') -> Dict[str, np.ndarray]:
    """
    Validate and preprocess input data for chi-squared test.

    Parameters:
    -----------
    observed : np.ndarray
        Observed frequency counts (2D array)
    expected : Optional[np.ndarray]
        Expected frequency counts (2D array). If None, uniform distribution is assumed.
    normalize : str
        Normalization method ('none', 'standard', 'minmax', 'robust')

    Returns:
    --------
    dict
        Dictionary containing validated and processed arrays with keys 'observed' and 'expected'
    """
    if observed.ndim != 2:
        raise ValueError("Observed data must be a 2D array")

    if expected is not None:
        if observed.shape != expected.shape:
            raise ValueError("Observed and expected arrays must have the same shape")
        if np.any(expected <= 0):
            raise ValueError("Expected frequencies must be positive")

    observed = np.asarray(observed, dtype=np.float64)
    if expected is not None:
        expected = np.asarray(expected, dtype=np.float64)

    # Apply normalization
    if normalize == 'standard':
        observed = (observed - np.mean(observed)) / np.std(observed)
        if expected is not None:
            expected = (expected - np.mean(expected)) / np.std(expected)
    elif normalize == 'minmax':
        observed = (observed - np.min(observed)) / (np.max(observed) - np.min(observed))
        if expected is not None:
            expected = (expected - np.min(expected)) / (np.max(expected) - np.min(expected))
    elif normalize == 'robust':
        observed = (observed - np.median(observed)) / (np.percentile(observed, 75) - np.percentile(observed, 25))
        if expected is not None:
            expected = (expected - np.median(expected)) / (np.percentile(expected, 75) - np.percentile(expected, 25))

    if expected is None:
        expected = np.full_like(observed, observed.size / (observed.shape[0] * observed.shape[1]))

    return {'observed': observed, 'expected': expected}

def compute_chi2_statistic(observed: np.ndarray,
                          expected: np.ndarray) -> float:
    """
    Compute chi-squared statistic.

    Parameters:
    -----------
    observed : np.ndarray
        Observed frequency counts (2D array)
    expected : np.ndarray
        Expected frequency counts (2D array)

    Returns:
    --------
    float
        Chi-squared statistic value
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.sum((observed - expected)**2 / expected)
    return float(chi2)

def compute_p_value(chi2_stat: float,
                   degrees_of_freedom: int) -> float:
    """
    Compute p-value from chi-squared statistic.

    Parameters:
    -----------
    chi2_stat : float
        Chi-squared statistic value
    degrees_of_freedom : int
        Degrees of freedom for the test

    Returns:
    --------
    float
        p-value
    """
    from scipy.stats import chi2
    return 1 - chi2.cdf(chi2_stat, degrees_of_freedom)

def test_chi2_fit(observed: np.ndarray,
                 expected: Optional[np.ndarray] = None,
                 normalize: str = 'none',
                 custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None) -> Dict[str, Union[float, Dict, str]]:
    """
    Perform chi-squared independence test.

    Parameters:
    -----------
    observed : np.ndarray
        Observed frequency counts (2D array)
    expected : Optional[np.ndarray]
        Expected frequency counts (2D array). If None, uniform distribution is assumed.
    normalize : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    custom_metric : Optional[Callable]
        Custom metric function taking (observed, expected) and returning a float

    Returns:
    --------
    dict
        Dictionary containing test results with keys 'result', 'metrics', 'params_used', and 'warnings'
    """
    # Validate inputs
    processed = validate_inputs(observed, expected, normalize)
    observed = processed['observed']
    expected = processed['expected']

    # Compute degrees of freedom
    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)

    # Compute chi-squared statistic
    chi2_stat = compute_chi2_statistic(observed, expected)

    # Compute p-value
    p_value = compute_p_value(chi2_stat, dof)

    # Compute metrics
    metrics = {}
    if custom_metric is not None:
        try:
            metrics['custom'] = custom_metric(observed, expected)
        except Exception as e:
            metrics['custom_error'] = str(e)

    # Prepare results
    result = {
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'degrees_of_freedom': dof
    }

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalize': normalize,
            'custom_metric': custom_metric is not None
        },
        'warnings': []
    }

# Example usage:
"""
observed = np.array([[10, 20], [30, 40]])
result = test_chi2_fit(observed)
print(result)
"""

################################################################################
# test_fisher_exact
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def _validate_input_matrix(matrix: np.ndarray) -> None:
    """Validate the input contingency table."""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if matrix.shape[0] != 2 or matrix.shape[1] != 2:
        raise ValueError("Input must be a 2x2 contingency table")
    if np.any(matrix < 0):
        raise ValueError("All values must be non-negative")
    if np.any(np.isnan(matrix)):
        raise ValueError("Matrix contains NaN values")

def _calculate_fisher_statistic(table: np.ndarray) -> float:
    """Calculate the Fisher exact test statistic."""
    a, b = table[0]
    c, d = table[1]

    def log_factorial(n: int) -> float:
        """Compute logarithm of factorial for numerical stability."""
        if n == 0:
            return 0.0
        return np.sum(np.log(np.arange(1, n + 1)))

    def log_hypergeometric(k: int) -> float:
        """Compute logarithm of hypergeometric probability."""
        return (log_factorial(a + c) + log_factorial(b + d)
                - log_factorial(a + b) - log_factorial(c + d)
                - log_factorial(k) - log_factorial(a + c - k))

    max_k = min(a + b, a + c)
    min_k = max(0, a - (b + d))

    log_prob = log_hypergeometric(a)
    for k in range(min_k, max_k + 1):
        if k == a:
            continue
        log_prob += np.log(1 + np.exp(log_hypergeometric(k) - log_prob))

    return 2 * log_prob

def _compute_p_value(statistic: float) -> float:
    """Compute the p-value from the Fisher statistic."""
    return 1 - (1 + np.exp(-statistic)) / 2

def test_fisher_exact_fit(
    contingency_table: np.ndarray,
    alternative_hypothesis: str = "two-sided",
    method: str = "midp",
    correction: bool = False
) -> Dict[str, Any]:
    """
    Perform Fisher's exact test for independence in a 2x2 contingency table.

    Parameters:
    -----------
    contingency_table : np.ndarray
        2x2 contingency table with non-negative integers.
    alternative_hypothesis : str, optional
        Type of alternative hypothesis ("two-sided", "less", or "greater").
    method : str, optional
        Method for p-value calculation ("midp" or "fisher").
    correction : bool, optional
        Whether to apply continuity correction.

    Returns:
    --------
    dict
        Dictionary containing test results, metrics, parameters used, and warnings.

    Example:
    --------
    >>> table = np.array([[10, 5], [2, 8]])
    >>> result = test_fisher_exact_fit(table)
    """
    _validate_input_matrix(contingency_table)

    statistic = _calculate_fisher_statistic(contingency_table)
    p_value = _compute_p_value(statistic)

    if correction:
        statistic -= 0.5
        p_value = _compute_p_value(statistic)

    result = {
        "statistic": statistic,
        "p_value": p_value,
        "alternative_hypothesis": alternative_hypothesis,
        "method": method,
        "correction_applied": correction
    }

    return {
        "result": result,
        "metrics": {},
        "params_used": {
            "alternative_hypothesis": alternative_hypothesis,
            "method": method,
            "correction": correction
        },
        "warnings": []
    }

################################################################################
# test_mantel_haenszel
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def _validate_inputs(table: np.ndarray) -> None:
    """Validate the input contingency table."""
    if not isinstance(table, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if table.ndim != 2:
        raise ValueError("Input must be a 2-dimensional array.")
    if table.shape[0] != 2 or table.shape[1] != 2:
        raise ValueError("Input must be a 2x2 contingency table.")
    if np.any(table < 0):
        raise ValueError("All values in the contingency table must be non-negative.")
    if np.any(np.isnan(table)):
        raise ValueError("Contingency table contains NaN values.")
    if np.any(np.isinf(table)):
        raise ValueError("Contingency table contains infinite values.")

def _compute_mantel_haenszel_statistic(table: np.ndarray) -> float:
    """Compute the Mantel-Haenszel statistic."""
    a, b = table[0]
    c, d = table[1]

    numerator = (a * d) - (b * c)
    denominator = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))

    if denominator == 0:
        raise ValueError("Denominator is zero, cannot compute statistic.")

    return numerator / denominator

def _compute_p_value(statistic: float) -> float:
    """Compute the p-value for the Mantel-Haenszel test."""
    # Using normal approximation
    from scipy.stats import norm
    return 2 * (1 - norm.cdf(abs(statistic)))

def test_mantel_haenszel_fit(
    table: np.ndarray,
    normalization: Optional[str] = None,
    metric: str = "standard",
    solver: str = "closed_form",
    custom_metric: Optional[Callable[[np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform the Mantel-Haenszel test for independence.

    Parameters:
    -----------
    table : np.ndarray
        2x2 contingency table.
    normalization : str, optional
        Normalization method (none, standard). Default is None.
    metric : str, optional
        Metric to use (standard). Default is "standard".
    solver : str, optional
        Solver method (closed_form). Default is "closed_form".
    custom_metric : Callable, optional
        Custom metric function. Default is None.
    **kwargs :
        Additional keyword arguments for future extensions.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    _validate_inputs(table)

    # Compute the Mantel-Haenszel statistic
    statistic = _compute_mantel_haenszel_statistic(table)

    # Compute the p-value
    p_value = _compute_p_value(statistic)

    # Prepare the result dictionary
    result = {
        "statistic": statistic,
        "p_value": p_value,
        "hypothesis": "Independence",
        "alternative": "Dependence"
    }

    metrics = {
        "statistic_type": metric,
        "normalization_used": normalization
    }

    params_used = {
        "normalization": normalization,
        "metric": metric,
        "solver": solver
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
table = np.array([[10, 20], [30, 40]])
result = test_mantel_haenszel_fit(table)
print(result)
"""

################################################################################
# test_logrank
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def _validate_inputs(survival_times: np.ndarray,
                    statuses: np.ndarray,
                    groups: np.ndarray) -> None:
    """
    Validate the input data for the logrank test.

    Parameters
    ----------
    survival_times : np.ndarray
        Array of survival times.
    statuses : np.ndarray
        Array of censoring indicators (1 for event, 0 for censored).
    groups : np.ndarray
        Array of group indicators.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if survival_times.ndim != 1 or statuses.ndim != 1 or groups.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if len(survival_times) != len(statuses) or len(survival_times) != len(groups):
        raise ValueError("All input arrays must have the same length.")
    if np.any(statuses < 0) or np.any(statuses > 1):
        raise ValueError("Statuses must be binary (0 or 1).")
    if np.any(np.isnan(survival_times)) or np.any(np.isinf(survival_times)):
        raise ValueError("Survival times must be finite and not contain NaN values.")

def _compute_logrank_statistic(survival_times: np.ndarray,
                              statuses: np.ndarray,
                              groups: np.ndarray) -> Dict[str, float]:
    """
    Compute the logrank test statistic.

    Parameters
    ----------
    survival_times : np.ndarray
        Array of survival times.
    statuses : np.ndarray
        Array of censoring indicators (1 for event, 0 for censored).
    groups : np.ndarray
        Array of group indicators.

    Returns
    -------
    Dict[str, float]
        Dictionary containing the test statistic and p-value.
    """
    unique_times = np.unique(survival_times)
    n_groups = len(np.unique(groups))
    group_counts = [np.sum(groups == g) for g in np.unique(groups)]
    total_at_risk = len(survival_times)

    observed_diff = 0.0
    expected_diff = 0.0
    var_expected = 0.0

    for t in unique_times:
        mask = survival_times == t
        at_risk = np.sum(survival_times >= t)
        events = statuses[mask]
        group_events = [np.sum((groups == g) & mask & (statuses == 1)) for g in np.unique(groups)]
        group_at_risk = [np.sum((groups == g) & (survival_times >= t)) for g in np.unique(groups)]

        observed = sum(group_events)
        expected = sum((e * at_risk / total_at_risk) for e in group_events)
        var = sum((e * (at_risk - e) * at_risk / (total_at_risk ** 2)) for e in group_events)

        observed_diff += sum((group_events[i] - expected * group_counts[i] / total_at_risk) for i in range(n_groups))
        expected_diff += sum((expected * group_counts[i] / total_at_risk) for i in range(n_groups))
        var_expected += var

    statistic = observed_diff / np.sqrt(var_expected)
    p_value = 2 * (1 - norm.cdf(abs(statistic)))

    return {
        "statistic": statistic,
        "p_value": p_value
    }

def test_logrank_fit(survival_times: np.ndarray,
                    statuses: np.ndarray,
                    groups: np.ndarray,
                    normalization: str = "none",
                    metric: Union[str, Callable] = "mse",
                    solver: str = "closed_form") -> Dict[str, Any]:
    """
    Perform the logrank test for independence between survival times and groups.

    Parameters
    ----------
    survival_times : np.ndarray
        Array of survival times.
    statuses : np.ndarray
        Array of censoring indicators (1 for event, 0 for censored).
    groups : np.ndarray
        Array of group indicators.
    normalization : str, optional
        Normalization method (none, standard, minmax, robust).
    metric : Union[str, Callable], optional
        Metric to use (mse, mae, r2, logloss, or custom callable).
    solver : str, optional
        Solver method (closed_form, gradient_descent, newton, coordinate_descent).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the test results, metrics, parameters used, and warnings.
    """
    _validate_inputs(survival_times, statuses, groups)

    result = _compute_logrank_statistic(survival_times, statuses, groups)

    metrics = {}
    if isinstance(metric, str):
        if metric == "mse":
            metrics["mse"] = np.mean((survival_times - np.mean(survival_times)) ** 2)
        elif metric == "mae":
            metrics["mae"] = np.mean(np.abs(survival_times - np.mean(survival_times)))
        elif metric == "r2":
            metrics["r2"] = 1 - np.sum((survival_times - np.mean(survival_times)) ** 2) / np.sum((survival_times - survival_times.mean()) ** 2)
    else:
        metrics["custom"] = metric(survival_times, groups)

    params_used = {
        "normalization": normalization,
        "metric": metric,
        "solver": solver
    }

    warnings = []
    if np.any(np.diff(survival_times) <= 0):
        warnings.append("Survival times are not strictly increasing.")

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

# Example usage:
# result = test_logrank_fit(
#     survival_times=np.array([1.0, 2.0, 3.0, 4.0]),
#     statuses=np.array([1, 1, 0, 1]),
#     groups=np.array([0, 0, 1, 1])
# )

################################################################################
# test_kolmogorov_smirnov
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(sample1: np.ndarray, sample2: np.ndarray) -> None:
    """
    Validate input samples for Kolmogorov-Smirnov test.

    Parameters
    ----------
    sample1 : np.ndarray
        First sample data.
    sample2 : np.ndarray
        Second sample data.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not isinstance(sample1, np.ndarray) or not isinstance(sample2, np.ndarray):
        raise ValueError("Inputs must be numpy arrays.")
    if sample1.ndim != 1 or sample2.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional.")
    if len(sample1) == 0 or len(sample2) == 0:
        raise ValueError("Samples must not be empty.")
    if np.any(np.isnan(sample1)) or np.any(np.isnan(sample2)):
        raise ValueError("Samples must not contain NaN values.")
    if np.any(np.isinf(sample1)) or np.any(np.isinf(sample2)):
        raise ValueError("Samples must not contain infinite values.")

def _compute_ks_statistic(sample1: np.ndarray, sample2: np.ndarray) -> float:
    """
    Compute the Kolmogorov-Smirnov statistic.

    Parameters
    ----------
    sample1 : np.ndarray
        First sample data.
    sample2 : np.ndarray
        Second sample data.

    Returns
    ------
    float
        The KS statistic.
    """
    n1, n2 = len(sample1), len(sample2)
    data = np.concatenate([sample1, sample2])
    d1 = np.sort(sample1)
    d2 = np.sort(sample2)

    cdf1 = np.searchsorted(d1, data, side='right') / n1
    cdf2 = np.searchsorted(d2, data, side='right') / n2

    return np.max(np.abs(cdf1 - cdf2))

def _compute_p_value(ks_stat: float, n1: int, n2: int) -> float:
    """
    Compute the p-value for the KS statistic.

    Parameters
    ----------
    ks_stat : float
        The KS statistic.
    n1 : int
        Size of the first sample.
    n2 : int
        Size of the second sample.

    Returns
    ------
    float
        The p-value.
    """
    n = n1 * n2 / (n1 + n2)
    en = np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)
    return 2 * np.sum([(-1)**(k+1) * np.exp(-2 * (k**2) * en**2) for k in range(1, 100)])

def test_kolmogorov_smirnov_fit(
    sample1: np.ndarray,
    sample2: np.ndarray,
    *,
    normalization: Optional[str] = None,
    metric: str = 'ks_stat',
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Union[float, Dict, str]]:
    """
    Perform the Kolmogorov-Smirnov independence test.

    Parameters
    ----------
    sample1 : np.ndarray
        First sample data.
    sample2 : np.ndarray
        Second sample data.
    normalization : Optional[str], default=None
        Normalization method (none, standard, minmax, robust).
    metric : str, default='ks_stat'
        Metric to compute ('ks_stat', 'p_value').
    custom_metric : Optional[Callable], default=None
        Custom metric function.
    **kwargs
        Additional keyword arguments for future extensions.

    Returns
    ------
    Dict[str, Union[float, Dict, str]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> sample1 = np.random.normal(0, 1, 100)
    >>> sample2 = np.random.normal(0.5, 1, 100)
    >>> result = test_kolmogorov_smirnov_fit(sample1, sample2)
    """
    _validate_inputs(sample1, sample2)

    n1, n2 = len(sample1), len(sample2)
    ks_stat = _compute_ks_statistic(sample1, sample2)
    p_value = _compute_p_value(ks_stat, n1, n2)

    metrics = {}
    if metric == 'ks_stat':
        metrics['ks_stat'] = ks_stat
    elif metric == 'p_value':
        metrics['p_value'] = p_value
    else:
        raise ValueError("Invalid metric specified.")

    if custom_metric is not None:
        metrics['custom_metric'] = custom_metric(sample1, sample2)

    result = {
        'result': ks_stat if metric == 'ks_stat' else p_value,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'custom_metric': custom_metric.__name__ if custom_metric else None
        },
        'warnings': []
    }

    return result

################################################################################
# test_wilcoxon
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for Wilcoxon test."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays")
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("Input arrays must not contain NaN or inf values")

def _compute_rank_sum(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Wilcoxon rank sum statistic."""
    combined = np.concatenate([x, y])
    ranks = np.argsort(np.argsort(combined))
    rank_sum_x = np.sum(ranks[:len(x)])
    return rank_sum_x

def _compute_p_value(statistic: float, n1: int, n2: int) -> float:
    """Compute the p-value for the Wilcoxon rank sum test."""
    mean = (n1 * (n1 + n2 + 1)) / 2
    variance = (n1 * n2 * (n1 + n2 + 1)) / 12
    z = (statistic - mean) / np.sqrt(variance)
    return 2 * min(1, 1 - _standard_normal_cdf(z))

def _standard_normal_cdf(x: float) -> float:
    """Compute the standard normal cumulative distribution function."""
    return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))

def test_wilcoxon_fit(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str = 'two-sided',
    correction: bool = True
) -> Dict[str, Union[float, Dict]]:
    """
    Perform the Wilcoxon rank sum test for independence.

    Parameters
    ----------
    x : np.ndarray
        First sample array.
    y : np.ndarray
        Second sample array.
    alternative : str, optional
        The alternative hypothesis ('two-sided', 'less', or 'greater').
    correction : bool, optional
        Whether to apply continuity correction.

    Returns
    -------
    Dict[str, Union[float, Dict]]
        Dictionary containing:
        - 'statistic': the test statistic
        - 'pvalue': the p-value
        - 'params_used': parameters used in the test
    """
    _validate_inputs(x, y)
    n1, n2 = len(x), len(y)

    statistic = _compute_rank_sum(x, y)
    if correction:
        statistic -= (n1 * (n1 + n2 + 1)) / 2

    p_value = _compute_p_value(statistic, n1, n2)

    if alternative == 'less':
        p_value = min(p_value / 2, 1)
    elif alternative == 'greater':
        p_value = max(p_value / 2, 0)

    return {
        'statistic': statistic,
        'pvalue': p_value,
        'params_used': {
            'alternative': alternative,
            'correction': correction
        }
    }

# Example usage:
# result = test_wilcoxon_fit(np.array([1, 2, 3]), np.array([4, 5, 6]))

################################################################################
# test_pearson
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for Pearson's independence test."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Inputs must have the same length")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("Input x contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Input y contains NaN or infinite values")

def _compute_pearson_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson's correlation coefficient."""
    n = x.shape[0]
    cov_xy = np.cov(x, y, bias=True)[0, 1]
    std_x = np.std(x, ddof=0)
    std_y = np.std(y, ddof=0)
    return cov_xy / (std_x * std_y)

def _compute_p_value(r: float, n: int) -> float:
    """Compute p-value for Pearson's correlation coefficient."""
    from scipy.stats import t
    df = n - 2
    if df <= 0:
        raise ValueError("Not enough degrees of freedom for p-value calculation")
    t_stat = r * np.sqrt(df / (1 - r**2))
    return 2 * t.sf(np.abs(t_stat), df)

def test_pearson_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    normalize: bool = False,
    custom_statistic_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_p_value_func: Optional[Callable[[float, int], float]] = None
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Perform Pearson's independence test between two variables.

    Parameters
    ----------
    x : np.ndarray
        First variable (1D array)
    y : np.ndarray
        Second variable (1D array)
    normalize : bool, optional
        Whether to standardize the variables before computation (default: False)
    custom_statistic_func : callable, optional
        Custom function to compute the test statistic (default: None)
    custom_p_value_func : callable, optional
        Custom function to compute the p-value (default: None)

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': float, the test statistic or p-value depending on configuration
        - 'metrics': dict of additional metrics
        - 'params_used': dict of parameters used in the computation
        - 'warnings': list of warning messages

    Example
    -------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 4, 6, 8, 10])
    >>> test_pearson_fit(x, y)
    """
    _validate_inputs(x, y)

    # Make copies to avoid modifying input arrays
    x_working = x.copy()
    y_working = y.copy()

    # Apply normalization if requested
    params_used = {
        'normalize': normalize,
        'custom_statistic_func': custom_statistic_func is not None,
        'custom_p_value_func': custom_p_value_func is not None
    }

    warnings = []

    if normalize:
        x_mean = np.mean(x_working)
        y_mean = np.mean(y_working)
        x_std = np.std(x_working, ddof=0)
        y_std = np.std(y_working, ddof=0)

        if x_std == 0 or y_std == 0:
            warnings.append("Standard deviation is zero for one or both variables - normalization skipped")

        if x_std != 0:
            x_working = (x_working - x_mean) / x_std
        if y_std != 0:
            y_working = (y_working - y_mean) / y_std

    # Compute statistic
    if custom_statistic_func is not None:
        r = custom_statistic_func(x_working, y_working)
    else:
        r = _compute_pearson_statistic(x_working, y_working)

    # Compute p-value
    if custom_p_value_func is not None:
        p_value = custom_p_value_func(r, x_working.shape[0])
    else:
        try:
            p_value = _compute_p_value(r, x_working.shape[0])
        except ValueError as e:
            warnings.append(str(e))
            p_value = np.nan

    metrics = {
        'correlation_coefficient': r,
        'p_value': p_value
    }

    return {
        'result': p_value,  # Typically the most important result for independence test
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }
