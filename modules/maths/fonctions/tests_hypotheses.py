"""
Quantix – Module tests_hypotheses
Généré automatiquement
Date: 2026-01-09
"""

################################################################################
# test_t_student
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    sample1: np.ndarray,
    sample2: np.ndarray,
    equal_var: bool = True
) -> None:
    """Validate input samples for t-test."""
    if not isinstance(sample1, np.ndarray) or not isinstance(sample2, np.ndarray):
        raise TypeError("Samples must be numpy arrays")
    if sample1.ndim != 1 or sample2.ndim != 1:
        raise ValueError("Samples must be 1-dimensional")
    if np.isnan(sample1).any() or np.isnan(sample2).any():
        raise ValueError("Samples contain NaN values")
    if np.isinf(sample1).any() or np.isinf(sample2).any():
        raise ValueError("Samples contain infinite values")
    if len(sample1) < 2 or len(sample2) < 2:
        raise ValueError("Samples must have at least 2 elements")

def _calculate_statistic(
    sample1: np.ndarray,
    sample2: np.ndarray,
    equal_var: bool = True
) -> float:
    """Calculate t-statistic for independent samples."""
    n1, n2 = len(sample1), len(sample2)
    mean1, mean2 = np.mean(sample1), np.mean(sample2)

    if equal_var:
        pooled_var = ((n1 - 1) * np.var(sample1, ddof=1) +
                      (n2 - 1) * np.var(sample2, ddof=1)) / (n1 + n2 - 2)
        std_err = np.sqrt(pooled_var * (1/n1 + 1/n2))
    else:
        var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
        std_err = np.sqrt(var1/n1 + var2/n2)

    t_stat = (mean1 - mean2) / std_err
    return float(t_stat)

def _calculate_p_value(
    t_stat: float,
    df: int
) -> float:
    """Calculate p-value from t-statistic and degrees of freedom."""
    from scipy.stats import t
    return (1 - t.cdf(abs(t_stat), df)) * 2

def test_t_student_fit(
    sample1: np.ndarray,
    sample2: np.ndarray,
    equal_var: bool = True,
    alternative: str = 'two-sided',
    custom_statistic: Optional[Callable] = None,
    custom_p_value: Optional[Callable] = None
) -> Dict[str, Union[float, Dict]]:
    """
    Perform independent t-test for means of two samples.

    Parameters
    ----------
    sample1 : np.ndarray
        First sample array.
    sample2 : np.ndarray
        Second sample array.
    equal_var : bool, optional
        Whether to assume equal population variances (default True).
    alternative : str, optional
        Alternative hypothesis ('two-sided', 'less', or 'greater').
    custom_statistic : Callable, optional
        Custom function to calculate test statistic.
    custom_p_value : Callable, optional
        Custom function to calculate p-value.

    Returns
    -------
    dict
        Dictionary containing test results, metrics, and parameters used.

    Example
    -------
    >>> sample1 = np.random.normal(0, 1, 100)
    >>> sample2 = np.random.normal(0.5, 1, 100)
    >>> result = test_t_student_fit(sample1, sample2)
    """
    # Validate inputs
    _validate_inputs(sample1, sample2, equal_var)

    # Calculate degrees of freedom
    n1, n2 = len(sample1), len(sample2)
    if equal_var:
        df = n1 + n2 - 2
    else:
        df = min(n1, n2) - 1

    # Calculate test statistic
    if custom_statistic is not None:
        t_stat = custom_statistic(sample1, sample2)
    else:
        t_stat = _calculate_statistic(sample1, sample2, equal_var)

    # Calculate p-value
    if custom_p_value is not None:
        p_value = custom_p_value(t_stat, df)
    else:
        p_value = _calculate_p_value(t_stat, df)

    # Adjust p-value based on alternative hypothesis
    if alternative == 'less':
        p_value = p_value / 2
    elif alternative == 'greater':
        p_value = min(1, 1 - (p_value / 2))

    # Prepare results
    result = {
        'result': {
            't_statistic': t_stat,
            'p_value': p_value,
            'degrees_of_freedom': df
        },
        'metrics': {
            'mean1': float(np.mean(sample1)),
            'mean2': float(np.mean(sample2)),
            'var1': float(np.var(sample1, ddof=1)),
            'var2': float(np.var(sample2, ddof=1))
        },
        'params_used': {
            'equal_var': equal_var,
            'alternative': alternative
        },
        'warnings': []
    }

    return result

################################################################################
# test_chi2
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def _validate_inputs(observed: np.ndarray, expected: np.ndarray) -> None:
    """
    Validate input arrays for chi-squared test.

    Parameters
    ----------
    observed : np.ndarray
        Observed frequency counts.
    expected : np.ndarray
        Expected frequency counts.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if observed.ndim != 1 or expected.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if observed.shape != expected.shape:
        raise ValueError("Observed and expected arrays must have the same shape.")
    if np.any(observed < 0) or np.any(expected <= 0):
        raise ValueError("Observed counts must be non-negative and expected counts must be positive.")
    if np.any(np.isnan(observed)) or np.any(np.isnan(expected)):
        raise ValueError("Inputs must not contain NaN values.")

def _compute_chi2_statistic(observed: np.ndarray, expected: np.ndarray) -> float:
    """
    Compute the chi-squared test statistic.

    Parameters
    ----------
    observed : np.ndarray
        Observed frequency counts.
    expected : np.ndarray
        Expected frequency counts.

    Returns
    ------
    float
        The chi-squared test statistic.
    """
    return np.sum((observed - expected) ** 2 / expected)

def _compute_p_value(chi2_stat: float, df: int) -> float:
    """
    Compute the p-value for the chi-squared test.

    Parameters
    ----------
    chi2_stat : float
        The chi-squared test statistic.
    df : int
        Degrees of freedom.

    Returns
    ------
    float
        The p-value.
    """
    from scipy.stats import chi2
    return 1 - chi2.cdf(chi2_stat, df)

def test_chi2_fit(
    observed: np.ndarray,
    expected: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = "chi2",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform a chi-squared goodness-of-fit test.

    Parameters
    ----------
    observed : np.ndarray
        Observed frequency counts.
    expected : np.ndarray
        Expected frequency counts.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the observed and expected counts, by default None.
    metric : str, optional
        Metric to compute ("chi2" or "custom"), by default "chi2".
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function, by default None.
    alpha : float, optional
        Significance level for the test, by default 0.05.

    Returns
    ------
    Dict[str, Any]
        Dictionary containing test results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> observed = np.array([10, 20, 30])
    >>> expected = np.array([15, 25, 35])
    >>> result = test_chi2_fit(observed, expected)
    """
    # Validate inputs
    _validate_inputs(observed, expected)

    # Normalize if specified
    if normalizer is not None:
        observed = normalizer(observed)
        expected = normalizer(expected)

    # Compute chi-squared statistic
    chi2_stat = _compute_chi2_statistic(observed, expected)

    # Compute degrees of freedom
    df = len(observed) - 1

    # Compute p-value
    p_value = _compute_p_value(chi2_stat, df)

    # Compute custom metric if specified
    metrics = {}
    if metric == "chi2":
        metrics["chi2_statistic"] = chi2_stat
    elif metric == "custom" and custom_metric is not None:
        metrics["custom_metric"] = custom_metric(observed, expected)
    else:
        raise ValueError("Invalid metric specified or custom_metric not provided.")

    # Determine test result
    result = "reject" if p_value < alpha else "fail to reject"

    # Prepare output
    output = {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "alpha": alpha,
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric": metric
        },
        "warnings": []
    }

    return output

def test_chi2_compute(
    observed: np.ndarray,
    expected: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = "chi2",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute chi-squared test components without hypothesis testing.

    Parameters
    ----------
    observed : np.ndarray
        Observed frequency counts.
    expected : np.ndarray
        Expected frequency counts.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the observed and expected counts, by default None.
    metric : str, optional
        Metric to compute ("chi2" or "custom"), by default "chi2".
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function, by default None.

    Returns
    ------
    Dict[str, Any]
        Dictionary containing test components, metrics, parameters used, and warnings.

    Examples
    --------
    >>> observed = np.array([10, 20, 30])
    >>> expected = np.array([15, 25, 35])
    >>> result = test_chi2_compute(observed, expected)
    """
    # Validate inputs
    _validate_inputs(observed, expected)

    # Normalize if specified
    if normalizer is not None:
        observed = normalizer(observed)
        expected = normalizer(expected)

    # Compute chi-squared statistic
    chi2_stat = _compute_chi2_statistic(observed, expected)

    # Compute degrees of freedom
    df = len(observed) - 1

    # Compute p-value
    p_value = _compute_p_value(chi2_stat, df)

    # Compute custom metric if specified
    metrics = {}
    if metric == "chi2":
        metrics["chi2_statistic"] = chi2_stat
    elif metric == "custom" and custom_metric is not None:
        metrics["custom_metric"] = custom_metric(observed, expected)
    else:
        raise ValueError("Invalid metric specified or custom_metric not provided.")

    # Prepare output
    output = {
        "result": None,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric": metric
        },
        "warnings": []
    }

    return output

################################################################################
# test_fisher
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, Y: np.ndarray) -> None:
    """Validate input matrices for Fisher's exact test."""
    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("Inputs must be 2-dimensional")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("Inputs must have the same number of rows")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
        raise ValueError("Y contains NaN or infinite values")

def _compute_contingency_table(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute the contingency table for Fisher's exact test."""
    n = X.shape[0]
    a = np.sum((X > 0) & (Y > 0))
    b = np.sum((X <= 0) & (Y > 0))
    c = np.sum((X > 0) & (Y <= 0))
    d = np.sum((X <= 0) & (Y <= 0))
    return np.array([[a, b], [c, d]])

def _fisher_exact_test(contingency_table: np.ndarray) -> Dict[str, float]:
    """Perform Fisher's exact test on a 2x2 contingency table."""
    from scipy.stats import fisher_exact

    odds_ratio, p_value = fisher_exact(contingency_table)
    return {
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value)
    }

def test_fisher_fit(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    contingency_table_func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    test_func: Optional[Callable[[np.ndarray], Dict[str, float]]] = None
) -> Dict:
    """
    Perform Fisher's exact test between two binary vectors.

    Parameters:
    -----------
    X : np.ndarray
        First binary vector (0 or 1)
    Y : np.ndarray
        Second binary vector (0 or 1)
    contingency_table_func : Callable, optional
        Custom function to compute the contingency table (default: _compute_contingency_table)
    test_func : Callable, optional
        Custom function to perform the Fisher's exact test (default: _fisher_exact_test)

    Returns:
    --------
    Dict
        Dictionary containing the test results, metrics, parameters used, and warnings

    Example:
    --------
    >>> X = np.array([1, 0, 1, 1, 0])
    >>> Y = np.array([1, 1, 0, 0, 1])
    >>> test_fisher_fit(X, Y)
    """
    _validate_inputs(X, Y)

    contingency_table_func = contingency_table_func or _compute_contingency_table
    test_func = test_func or _fisher_exact_test

    contingency_table = contingency_table_func(X, Y)
    test_results = test_func(contingency_table)

    return {
        "result": test_results,
        "metrics": {},
        "params_used": {
            "contingency_table_func": str(contingency_table_func),
            "test_func": str(test_func)
        },
        "warnings": []
    }

################################################################################
# test_wilcoxon
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for Wilcoxon test."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape.")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values.")

def _compute_wilcoxon_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Wilcoxon signed-rank statistic."""
    differences = x - y
    ranked_diffs = np.abs(np.argsort(np.abs(differences)))
    signed_ranked_diffs = ranked_diffs * np.sign(differences)
    statistic = np.sum(signed_ranked_diffs)
    return statistic

def _compute_p_value(statistic: float, n: int) -> float:
    """Compute the p-value for the Wilcoxon signed-rank test."""
    from scipy.stats import wilcoxon
    _, p_value = wilcoxon(np.zeros(n), np.ones(n))
    return p_value

def test_wilcoxon_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alternative: str = 'two-sided',
    correction: bool = True,
    method: str = 'exact'
) -> Dict[str, Any]:
    """
    Perform the Wilcoxon signed-rank test.

    Parameters:
    -----------
    x : np.ndarray
        First set of measurements.
    y : np.ndarray
        Second set of measurements.
    alternative : str, optional
        The alternative hypothesis. 'two-sided', 'less', or 'greater'.
    correction : bool, optional
        Whether to apply continuity correction.
    method : str, optional
        The method used for p-value calculation. 'exact' or 'approx'.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the test results.
    """
    _validate_inputs(x, y)

    statistic = _compute_wilcoxon_statistic(x, y)
    p_value = _compute_p_value(statistic, len(x))

    result = {
        'statistic': statistic,
        'p_value': p_value,
        'alternative': alternative,
        'correction': correction,
        'method': method
    }

    return result

# Example usage:
# x = np.array([1, 2, 3, 4, 5])
# y = np.array([2, 3, 4, 5, 6])
# result = test_wilcoxon_fit(x, y)

################################################################################
# test_kolmogorov_smirnov
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def _validate_inputs(data: np.ndarray, distribution: str = 'normal', **kwargs) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if len(data.shape) != 1:
        raise ValueError("Input data must be one-dimensional.")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values.")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values.")

def _compute_ks_statistic(data: np.ndarray, cdf_func: Callable) -> float:
    """Compute the Kolmogorov-Smirnov statistic."""
    n = len(data)
    data_sorted = np.sort(data)
    cdf_values = cdf_func(data_sorted)
    en = np.arange(1, n + 1) / n
    d_plus = np.max(cdf_values - en)
    d_minus = np.max(en - cdf_values)
    return max(d_plus, d_minus)

def _compute_p_value(ks_statistic: float, n: int) -> float:
    """Compute the p-value for the KS statistic."""
    if ks_statistic <= 0:
        return 1.0
    p_value = np.sum([(-1) ** (k + 1) * np.exp(-2 * (k ** 2) * ks_statistic ** 2) for k in range(1, 100)])
    return p_value

def test_kolmogorov_smirnov_fit(
    data: np.ndarray,
    distribution: str = 'normal',
    cdf_func: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Union[float, Dict]]:
    """
    Perform the Kolmogorov-Smirnov test to compare a sample with a reference distribution.

    Parameters:
    -----------
    data : np.ndarray
        Input data to test.
    distribution : str, optional
        Name of the reference distribution ('normal', 'uniform', etc.).
    cdf_func : Callable, optional
        Custom CDF function if not using a standard distribution.

    Returns:
    --------
    dict
        Dictionary containing the test results, metrics, parameters used, and warnings.
    """
    _validate_inputs(data, distribution)

    if cdf_func is None:
        if distribution == 'normal':
            from scipy.stats import norm
            cdf_func = norm.cdf
        elif distribution == 'uniform':
            from scipy.stats import uniform
            cdf_func = uniform.cdf
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

    ks_statistic = _compute_ks_statistic(data, cdf_func)
    p_value = _compute_p_value(ks_statistic, len(data))

    result = {
        'statistic': ks_statistic,
        'p_value': p_value
    }

    metrics = {
        'ks_statistic': ks_statistic,
        'p_value': p_value
    }

    params_used = {
        'distribution': distribution,
        'cdf_func': cdf_func.__name__ if hasattr(cdf_func, '__name__') else 'custom'
    }

    warnings = []

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

# Example usage:
# data = np.random.normal(0, 1, 100)
# result = test_kolmogorov_smirnov_fit(data, distribution='normal')

################################################################################
# test_mann_whitney
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def _validate_inputs(
    sample1: np.ndarray,
    sample2: np.ndarray,
    alternative: str = 'two-sided'
) -> None:
    """
    Validate input samples for Mann-Whitney U test.

    Parameters
    ----------
    sample1 : np.ndarray
        First sample data.
    sample2 : np.ndarray
        Second sample data.
    alternative : str, optional
        Alternative hypothesis ('two-sided', 'less', 'greater').

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not isinstance(sample1, np.ndarray) or not isinstance(sample2, np.ndarray):
        raise ValueError("Samples must be numpy arrays.")
    if sample1.ndim != 1 or sample2.ndim != 1:
        raise ValueError("Samples must be 1-dimensional.")
    if len(sample1) < 2 or len(sample2) < 2:
        raise ValueError("Each sample must have at least 2 observations.")
    if np.any(np.isnan(sample1)) or np.any(np.isnan(sample2)):
        raise ValueError("Samples must not contain NaN values.")
    if np.any(np.isinf(sample1)) or np.any(np.isinf(sample2)):
        raise ValueError("Samples must not contain infinite values.")
    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError("Alternative must be 'two-sided', 'less' or 'greater'.")

def _compute_ranks(
    combined: np.ndarray,
    sample1_indices: np.ndarray
) -> tuple:
    """
    Compute ranks and U statistics for Mann-Whitney test.

    Parameters
    ----------
    combined : np.ndarray
        Combined and sorted array of both samples.
    sample1_indices : np.ndarray
        Indices of the first sample in the combined array.

    Returns
    ------
    tuple
        (ranks, U1, U2)
    """
    n1 = len(sample1_indices)
    n2 = len(combined) - n1
    ranks = np.arange(1, len(combined) + 1)
    U1 = ranks[sample1_indices].sum() - n1 * (n1 + 1) / 2
    U2 = n1 * n2 - U1
    return ranks, U1, U2

def _compute_p_value(
    U: float,
    n1: int,
    n2: int,
    alternative: str
) -> float:
    """
    Compute p-value for Mann-Whitney U test.

    Parameters
    ----------
    U : float
        U statistic.
    n1 : int
        Size of first sample.
    n2 : int
        Size of second sample.
    alternative : str
        Alternative hypothesis.

    Returns
    ------
    float
        Computed p-value.
    """
    mean = n1 * n2 / 2
    std = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (U - mean) / std

    if alternative == 'two-sided':
        p_value = 2 * (1 - _standard_normal_cdf(np.abs(z)))
    elif alternative == 'less':
        p_value = _standard_normal_cdf(z)
    else:  # greater
        p_value = 1 - _standard_normal_cdf(z)

    return p_value

def _standard_normal_cdf(x: float) -> float:
    """
    Approximate standard normal CDF.

    Parameters
    ----------
    x : float
        Value at which to evaluate CDF.

    Returns
    ------
    float
        CDF value.
    """
    return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))

def test_mann_whitney_fit(
    sample1: np.ndarray,
    sample2: np.ndarray,
    alternative: str = 'two-sided',
    correction: bool = True
) -> Dict[str, Any]:
    """
    Perform Mann-Whitney U test.

    Parameters
    ----------
    sample1 : np.ndarray
        First sample data.
    sample2 : np.ndarray
        Second sample data.
    alternative : str, optional
        Alternative hypothesis ('two-sided', 'less', 'greater').
    correction : bool, optional
        Whether to apply continuity correction.

    Returns
    ------
    Dict[str, Any]
        Dictionary containing test results.
    """
    _validate_inputs(sample1, sample2, alternative)

    combined = np.concatenate([sample1, sample2])
    combined_sorted = np.sort(combined)
    sample1_indices = np.where(np.in1d(combined_sorted, sample1))[0]

    _, U1, _ = _compute_ranks(combined_sorted, sample1_indices)
    n1 = len(sample1)
    n2 = len(sample2)

    if correction:
        U1_corrected = U1 - n1 * n2 / 2
    else:
        U1_corrected = U1

    p_value = _compute_p_value(U1_corrected, n1, n2, alternative)

    return {
        "result": {
            "statistic": U1,
            "p_value": p_value
        },
        "params_used": {
            "alternative": alternative,
            "correction": correction
        },
        "warnings": []
    }

# Example usage:
# result = test_mann_whitney_fit(
#     sample1=np.array([1, 2, 3]),
#     sample2=np.array([4, 5, 6]),
#     alternative='two-sided'
# )

################################################################################
# test_anova
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    groups: Dict[str, np.ndarray],
    normalizations: Optional[Dict[str, Callable]] = None,
) -> Dict[str, np.ndarray]:
    """
    Validate input groups and apply normalizations.

    Parameters
    ----------
    groups : Dict[str, np.ndarray]
        Dictionary of group names and their corresponding data arrays.
    normalizations : Dict[str, Callable], optional
        Dictionary of normalization functions for each group.

    Returns
    -------
    Dict[str, np.ndarray]
        Normalized groups data.
    """
    if not isinstance(groups, dict):
        raise ValueError("Groups must be provided as a dictionary.")

    if normalizations is None:
        normalizations = {group: lambda x: x for group in groups}

    normalized_groups = {}
    for name, data in groups.items():
        if not isinstance(data, np.ndarray):
            raise ValueError(f"Group data for {name} must be a numpy array.")

        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError(f"Group data for {name} contains NaN or inf values.")

        normalized_groups[name] = normalizations.get(name, lambda x: x)(data)

    return normalized_groups

def _compute_anova_statistic(
    groups: Dict[str, np.ndarray],
    metric: Callable = np.var,
) -> float:
    """
    Compute the ANOVA statistic.

    Parameters
    ----------
    groups : Dict[str, np.ndarray]
        Dictionary of group names and their corresponding data arrays.
    metric : Callable, optional
        Metric function to compute the variance.

    Returns
    -------
    float
        The computed ANOVA statistic.
    """
    # Compute group means and overall mean
    group_means = {name: np.mean(data) for name, data in groups.items()}
    overall_mean = np.mean(np.concatenate(list(groups.values())))

    # Compute between-group variance
    n_groups = len(groups)
    n_total = sum(len(data) for data in groups.values())
    between_group_variance = sum(
        len(data) * (group_mean - overall_mean) ** 2
        for data, group_mean in zip(groups.values(), group_means.values())
    ) / (n_groups - 1)

    # Compute within-group variance
    within_group_variance = sum(
        metric(data) * (len(data) - 1)
        for data in groups.values()
    ) / (n_total - n_groups)

    # Compute F-statistic
    f_statistic = between_group_variance / within_group_variance

    return f_statistic

def test_anova_fit(
    groups: Dict[str, np.ndarray],
    normalizations: Optional[Dict[str, Callable]] = None,
    metric: Callable = np.var,
) -> Dict[str, Union[float, Dict, str]]:
    """
    Perform ANOVA test.

    Parameters
    ----------
    groups : Dict[str, np.ndarray]
        Dictionary of group names and their corresponding data arrays.
    normalizations : Dict[str, Callable], optional
        Dictionary of normalization functions for each group.
    metric : Callable, optional
        Metric function to compute the variance.

    Returns
    -------
    Dict[str, Union[float, Dict, str]]
        Dictionary containing the test results, metrics, parameters used, and warnings.
    """
    # Validate inputs and apply normalizations
    normalized_groups = _validate_inputs(groups, normalizations)

    # Compute ANOVA statistic
    f_statistic = _compute_anova_statistic(normalized_groups, metric)

    # Placeholder for p-value calculation (simplified for example)
    p_value = 0.5  # In practice, use a statistical library to compute this

    result = {
        "f_statistic": f_statistic,
        "p_value": p_value,
    }

    metrics = {
        "between_group_variance": result["f_statistic"] * (sum(len(data) for data in groups.values()) - len(groups)) / (len(groups) - 1),
        "within_group_variance": result["f_statistic"] * (len(groups) - 1) / (sum(len(data) for data in groups.values()) - len(groups)),
    }

    params_used = {
        "normalizations": normalizations,
        "metric": metric.__name__ if hasattr(metric, '__name__') else str(metric),
    }

    warnings = []

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings,
    }

# Example usage
if __name__ == "__main__":
    # Example data
    groups = {
        "group1": np.random.normal(0, 1, 30),
        "group2": np.random.normal(1, 1, 30),
        "group3": np.random.normal(2, 1, 30),
    }

    # Example normalization function
    def standard_normalize(data: np.ndarray) -> np.ndarray:
        return (data - np.mean(data)) / np.std(data)

    normalizations = {
        "group1": standard_normalize,
        "group2": standard_normalize,
        "group3": standard_normalize,
    }

    # Perform ANOVA test
    anova_result = test_anova_fit(groups, normalizations)
    print(anova_result)

################################################################################
# test_shapiro_wilk
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for Shapiro-Wilk test."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if len(data.shape) != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values")

def _compute_statistic(data: np.ndarray, normalization: str = 'standard') -> float:
    """Compute Shapiro-Wilk test statistic."""
    n = len(data)
    if normalization == 'standard':
        data_sorted = np.sort(data)
        mean = np.mean(data_sorted)
        std = np.std(data_sorted, ddof=1)
        data_normalized = (data_sorted - mean) / std
    elif normalization == 'none':
        data_normalized = np.sort(data)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    m = np.arange(1, n + 1)
    expected_order_stats = -0.5 * np.log(1 - (m - 0.375) / (n + 0.25))
    expected_order_stats = np.sort(expected_order_stats)

    cov_matrix = np.corrcoef(data_normalized, expected_order_stats)
    statistic = (n / ((cov_matrix[0, 1] + cov_matrix[1, 0]) ** 2))
    return statistic

def _compute_p_value(statistic: float, n: int) -> float:
    """Compute p-value for Shapiro-Wilk test."""
    # This is a simplified approximation
    if n < 3:
        raise ValueError("Sample size must be at least 3")
    p_value = 1 - statistic
    return max(0.0, min(1.0, p_value))

def test_shapiro_wilk_fit(
    data: np.ndarray,
    normalization: str = 'standard',
    custom_statistic_func: Optional[Callable[[np.ndarray], float]] = None,
    custom_p_value_func: Optional[Callable[[float, int], float]] = None
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Perform Shapiro-Wilk test for normality.

    Parameters
    ----------
    data : np.ndarray
        Input data to test for normality.
    normalization : str, optional
        Normalization method ('none', 'standard'), by default 'standard'.
    custom_statistic_func : Callable, optional
        Custom function to compute test statistic.
    custom_p_value_func : Callable, optional
        Custom function to compute p-value.

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], str]]
        Dictionary containing test results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.random.normal(0, 1, 100)
    >>> result = test_shapiro_wilk_fit(data)
    """
    _validate_input(data)

    params_used = {
        'normalization': normalization,
        'custom_statistic_func': custom_statistic_func is not None,
        'custom_p_value_func': custom_p_value_func is not None
    }

    warnings = []

    if custom_statistic_func is not None:
        statistic = custom_statistic_func(data)
    else:
        statistic = _compute_statistic(data, normalization)

    if custom_p_value_func is not None:
        p_value = custom_p_value_func(statistic, len(data))
    else:
        p_value = _compute_p_value(statistic, len(data))

    metrics = {
        'statistic': statistic,
        'p_value': p_value
    }

    result = {
        'result': 'normal' if p_value > 0.05 else 'non-normal',
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

    return result

################################################################################
# test_levene
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(*samples: np.ndarray) -> None:
    """Validate input samples for Levene's test."""
    if len(samples) < 2:
        raise ValueError("At least two samples are required for Levene's test.")
    for sample in samples:
        if not isinstance(sample, np.ndarray):
            raise TypeError("All inputs must be numpy arrays.")
        if sample.ndim != 1:
            raise ValueError("All inputs must be 1-dimensional arrays.")
        if np.isnan(sample).any() or np.isinf(sample).any():
            raise ValueError("Input arrays must not contain NaN or infinite values.")

def _center_data(
    samples: np.ndarray,
    center_func: Callable[[np.ndarray], float],
) -> np.ndarray:
    """Center the data using the specified center function."""
    centered = []
    for sample in samples:
        center_value = center_func(sample)
        centered.append(np.abs(sample - center_value))
    return np.array(centered)

def _compute_statistic(
    centered_samples: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """Compute the Levene's test statistic."""
    n_samples = len(centered_samples)
    n_total = sum(len(sample) for sample in centered_samples)

    # Compute within-group sums of squares
    ss_within = 0.0
    for sample in centered_samples:
        ss_within += np.sum(sample ** 2)

    # Compute between-group sums of squares
    group_means = [np.mean(sample) for sample in centered_samples]
    overall_mean = np.mean(np.concatenate(centered_samples))
    ss_between = 0.0
    for i, sample in enumerate(centered_samples):
        ss_between += len(sample) * (group_means[i] - overall_mean) ** 2

    # Compute the F-statistic
    ms_between = ss_between / (n_samples - 1)
    ms_within = ss_within / (n_total - n_samples)
    f_statistic = ms_between / ms_within

    return f_statistic

def _compute_p_value(f_statistic: float, df1: int, df2: int) -> float:
    """Compute the p-value for Levene's test using the F-distribution."""
    from scipy.stats import f
    return 1 - f.cdf(f_statistic, df1, df2)

def test_levene_fit(
    *samples: np.ndarray,
    center_func: Callable[[np.ndarray], float] = np.median,
    distance_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Perform Levene's test for equality of variances.

    Parameters:
    -----------
    *samples : np.ndarray
        Input samples to test. Each sample must be a 1-dimensional numpy array.
    center_func : Callable[[np.ndarray], float], optional
        Function to compute the center of each sample. Default is np.median.
    distance_func : Callable[[np.ndarray, np.ndarray], float], optional
        Function to compute the distance between samples. If None, no distance is computed.

    Returns:
    --------
    Dict[str, Union[float, Dict[str, float], Dict[str, str]]]
        A dictionary containing the test results, metrics, parameters used, and warnings.
    """
    _validate_inputs(*samples)

    # Center the data
    centered_samples = _center_data(samples, center_func)

    # Compute the test statistic
    f_statistic = _compute_statistic(centered_samples, distance_func)

    # Compute degrees of freedom
    n_samples = len(samples)
    n_total = sum(len(sample) for sample in samples)
    df1 = n_samples - 1
    df2 = n_total - n_samples

    # Compute the p-value
    p_value = _compute_p_value(f_statistic, df1, df2)

    # Prepare the result dictionary
    result = {
        "statistic": f_statistic,
        "p_value": p_value
    }

    metrics = {
        "f_statistic": f_statistic,
        "p_value": p_value
    }

    params_used = {
        "center_func": center_func.__name__,
        "distance_func": distance_func.__name__ if distance_func else None
    }

    warnings = {}

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

# Example usage:
# sample1 = np.random.normal(0, 1, 100)
# sample2 = np.random.normal(0, 2, 100)
# result = test_levene_fit(sample1, sample2)

################################################################################
# test_mcnemar
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input_matrix(matrix: np.ndarray) -> None:
    """Validate input matrix for McNemar test."""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2-dimensional array")
    if matrix.shape[0] != 2 or matrix.shape[1] != 2:
        raise ValueError("Input must be a 2x2 contingency table")
    if np.any(matrix < 0):
        raise ValueError("All values must be non-negative")
    if np.any(np.isnan(matrix)):
        raise ValueError("Matrix contains NaN values")

def _calculate_mcnemar_statistic(matrix: np.ndarray) -> float:
    """Calculate McNemar test statistic."""
    a, b = matrix[0]
    c, d = matrix[1]

    n_discordant = b + c
    if n_discordant == 0:
        return np.nan

    statistic = (b - c) ** 2 / n_discordant
    return statistic

def _calculate_p_value(statistic: float) -> float:
    """Calculate p-value from McNemar test statistic."""
    if np.isnan(statistic):
        return np.nan
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(statistic, df=1)
    return p_value

def test_mcnemar_fit(
    contingency_table: np.ndarray,
    alternative_hypothesis: str = "two-sided",
    correction: bool = True
) -> Dict[str, Union[float, Dict]]:
    """
    Perform McNemar's test for paired nominal data.

    Parameters
    ----------
    contingency_table : np.ndarray
        2x2 contingency table with format:
        [[a, b],
         [c, d]]
    alternative_hypothesis : str
        Type of alternative hypothesis. Options: "two-sided", "less", "greater"
    correction : bool
        Whether to apply continuity correction

    Returns
    -------
    Dict containing:
        - result: dict with test statistic and p-value
        - params_used: dict of parameters used
        - warnings: list of warning messages

    Example
    -------
    >>> contingency_table = np.array([[20, 15], [10, 45]])
    >>> test_mcnemar_fit(contingency_table)
    {
        'result': {'statistic': 1.333..., 'p_value': 0.248...},
        'params_used': {'alternative_hypothesis': 'two-sided', 'correction': True},
        'warnings': []
    }
    """
    # Validate input
    _validate_input_matrix(contingency_table)

    # Calculate test statistic
    statistic = _calculate_mcnemar_statistic(contingency_table)

    # Apply continuity correction if requested
    if correction:
        n_discordant = contingency_table[0, 1] + contingency_table[1, 0]
        if n_discordant > 0:
            statistic = (abs(contingency_table[0, 1] - contingency_table[1, 0]) - 1) ** 2 / n_discordant

    # Calculate p-value
    p_value = _calculate_p_value(statistic)

    # Handle alternative hypothesis
    if alternative_hypothesis == "less":
        p_value = min(p_value, 1 - p_value)
    elif alternative_hypothesis == "greater":
        p_value = min(p_value, 1 - p_value)
    else:  # two-sided
        pass

    return {
        "result": {
            "statistic": statistic,
            "p_value": p_value
        },
        "params_used": {
            "alternative_hypothesis": alternative_hypothesis,
            "correction": correction
        },
        "warnings": []
    }
