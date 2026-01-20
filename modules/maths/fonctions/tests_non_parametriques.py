"""
Quantix – Module tests_non_parametriques
Généré automatiquement
Date: 2026-01-09
"""

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
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def _calculate_rank_sum(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate the Wilcoxon rank sum statistic."""
    combined = np.concatenate([x, y])
    ranks = np.argsort(np.argsort(combined))
    rank_sum_x = np.sum(ranks[:len(x)])
    return rank_sum_x

def _calculate_p_value(statistic: float, n1: int, n2: int) -> float:
    """Calculate the p-value for Wilcoxon test using normal approximation."""
    mean = (n1 * (n1 + n2 + 1)) / 2
    variance = (n1 * n2 * (n1 + n2 + 1)) / 12
    z = (statistic - mean) / np.sqrt(variance)
    p_value = 2 * min(1, 1 - _standard_normal_cdf(z))
    return p_value

def _standard_normal_cdf(x: float) -> float:
    """Approximate standard normal cumulative distribution function."""
    return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))

def test_wilcoxon_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    correction: str = "none",
    alternative: str = "two-sided"
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Perform Wilcoxon rank sum test (Mann-Whitney U test).

    Parameters
    ----------
    x : np.ndarray
        First sample array.
    y : np.ndarray
        Second sample array.
    correction : str, optional
        Type of p-value correction ("none", "holm", "bonferroni").
    alternative : str, optional
        Alternative hypothesis ("two-sided", "less", "greater").

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], str]]
        Dictionary containing test results, metrics, parameters used and warnings.

    Example
    -------
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([4, 5, 6])
    >>> result = test_wilcoxon_fit(x, y)
    """
    _validate_inputs(x, y)

    n1 = len(x)
    n2 = len(y)

    statistic = _calculate_rank_sum(x, y)
    p_value = _calculate_p_value(statistic, n1, n2)

    if correction == "holm":
        p_value = _holm_correction(p_value)
    elif correction == "bonferroni":
        p_value *= 2

    if alternative in ["less", "greater"]:
        p_value = min(p_value, 1 - p_value)

    return {
        "result": {
            "statistic": statistic,
            "p_value": p_value
        },
        "metrics": {},
        "params_used": {
            "correction": correction,
            "alternative": alternative
        },
        "warnings": []
    }

def _holm_correction(p_values: float) -> float:
    """Apply Holm correction for multiple comparisons."""
    return p_values * 2

################################################################################
# test_kruskal_wallis
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(*groups: np.ndarray) -> None:
    """Validate input groups for Kruskal-Wallis test."""
    if len(groups) < 2:
        raise ValueError("At least two groups are required for Kruskal-Wallis test.")
    for group in groups:
        if not isinstance(group, np.ndarray):
            raise TypeError("All inputs must be numpy arrays.")
        if group.ndim != 1:
            raise ValueError("Each input must be a 1-dimensional array.")
        if np.any(np.isnan(group)) or np.any(np.isinf(group)):
            raise ValueError("Input arrays must not contain NaN or infinite values.")

def _rank_data(*groups: np.ndarray) -> tuple:
    """Compute ranks for all data points across groups."""
    combined = np.concatenate(groups)
    ranks = np.argsort(np.argsort(combined))
    return tuple(ranks[i:j] for i, j in zip(np.cumsum([0] + [len(g) for g in groups])[:-1],
                                           np.cumsum([len(g) for g in groups])))

def _compute_statistic(*groups: np.ndarray, rank_method: str = 'average') -> float:
    """Compute the Kruskal-Wallis H statistic."""
    ranks = _rank_data(*groups)
    n_groups = len(groups)
    N = sum(len(group) for group in groups)

    if rank_method == 'average':
        # Average ranks
        avg_ranks = [np.mean(rank) for rank in ranks]
    elif rank_method == 'median':
        # Median ranks
        avg_ranks = [np.median(rank) for rank in ranks]
    else:
        raise ValueError("rank_method must be 'average' or 'median'.")

    # Compute the H statistic
    sum_ranks_squared = sum(n * (avg_rank - (N + 1) / 2)**2 for n, avg_rank in zip([len(g) for g in groups], avg_ranks))
    H = (12 / (N * (N + 1))) * sum_ranks_squared - 3 * (N + 1)

    return H

def _compute_p_value(H: float, n_groups: int) -> float:
    """Compute the p-value for the Kruskal-Wallis test using chi-square approximation."""
    if n_groups < 3:
        raise ValueError("Chi-square approximation requires at least 3 groups.")
    p_value = 1 - _chi_square_cdf(H, df=n_groups - 1)
    return p_value

def _chi_square_cdf(x: float, df: int) -> float:
    """Compute the CDF of the chi-square distribution."""
    from scipy.stats import chi2
    return chi2.cdf(x, df)

def test_kruskal_wallis_fit(
    *groups: np.ndarray,
    rank_method: str = 'average',
    correction: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform the Kruskal-Wallis H test for independent samples.

    Parameters:
    -----------
    *groups : np.ndarray
        Input groups to compare. Each group must be a 1-dimensional numpy array.
    rank_method : str, optional
        Method for computing ranks ('average' or 'median'), default is 'average'.
    correction : str, optional
        Correction method for p-value ('bonferroni' or None), default is None.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the test results, metrics, parameters used, and warnings.
    """
    _validate_inputs(*groups)

    H = _compute_statistic(*groups, rank_method=rank_method)
    n_groups = len(groups)

    p_value = _compute_p_value(H, n_groups)
    if correction == 'bonferroni':
        p_value = p_value * n_groups

    result = {
        "statistic": H,
        "p_value": p_value,
        "n_groups": n_groups
    }

    metrics = {
        "rank_method": rank_method,
        "correction": correction
    }

    params_used = {
        "rank_method": rank_method,
        "correction": correction
    }

    warnings = []

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

################################################################################
# test_mann_whitney
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for Mann-Whitney U test."""
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional")
    if len(x) != len(y):
        raise ValueError("Inputs must have the same length")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("Inputs must contain only finite values")

def _compute_rank_sum(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the rank sum statistic for Mann-Whitney U test."""
    combined = np.concatenate([x, y])
    ranks = np.argsort(np.argsort(combined))
    rank_sum_x = np.sum(ranks[:len(x)])
    return rank_sum_x

def _compute_u_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Mann-Whitney U statistic."""
    n1, n2 = len(x), len(y)
    rank_sum_x = _compute_rank_sum(x, y)
    u1 = rank_sum_x - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    return min(u1, u2)

def _compute_p_value(u: float, n1: int, n2: int) -> float:
    """Compute the p-value for Mann-Whitney U test."""
    from scipy.stats import mannwhitneyu
    return mannwhitneyu(x, y, alternative='two-sided').pvalue

def test_mann_whitney_fit(
    x: np.ndarray,
    y: np.ndarray,
    normalization: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform Mann-Whitney U test with configurable options.

    Parameters:
    - x: First sample array
    - y: Second sample array
    - normalization: Normalization method ('none', 'standard', 'minmax', 'robust')
    - custom_metric: Optional custom metric function
    - **kwargs: Additional parameters for the test

    Returns:
    Dictionary containing results, metrics, parameters used and warnings
    """
    _validate_inputs(x, y)

    # Apply normalization if specified
    if normalization == 'standard':
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
    elif normalization == 'minmax':
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif normalization == 'robust':
        x = (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25))
        y = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))

    # Compute U statistic
    u_statistic = _compute_u_statistic(x, y)
    p_value = _compute_p_value(u_statistic, len(x), len(y))

    # Compute custom metric if provided
    metrics = {}
    if custom_metric is not None:
        metrics['custom_metric'] = custom_metric(x, y)

    return {
        'result': {
            'u_statistic': u_statistic,
            'p_value': p_value
        },
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'custom_metric': custom_metric is not None
        },
        'warnings': []
    }

# Example usage:
# x = np.random.normal(0, 1, 100)
# y = np.random.normal(1, 1, 100)
# result = test_mann_whitney_fit(x, y, normalization='standard')

################################################################################
# test_sign
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for the sign test."""
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape.")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values.")

def _compute_sign_statistic(x: np.ndarray, y: np.ndarray) -> int:
    """Compute the sign statistic for the sign test."""
    differences = x - y
    signs = np.sign(differences)
    valid_signs = signs[~np.isnan(signs)]  # Exclude zero differences
    return np.sum(valid_signs > 0)

def _compute_p_value(statistic: int, n: int) -> float:
    """Compute the p-value for the sign test."""
    if statistic > n / 2:
        return 1 - _binom_cdf(n, statistic, 0.5)
    else:
        return _binom_cdf(n - 1, statistic, 0.5)

def _binom_cdf(n: int, k: int, p: float) -> float:
    """Compute the binomial cumulative distribution function."""
    from scipy.stats import binom
    return binom.cdf(k, n, p)

def test_sign_fit(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str = 'two-sided',
    method: str = 'exact'
) -> Dict[str, Any]:
    """
    Perform the sign test for paired samples.

    Parameters:
    -----------
    x : np.ndarray
        First set of sample measurements.
    y : np.ndarray
        Second set of sample measurements.
    alternative : str, optional
        The alternative hypothesis to test. Must be 'two-sided', 'less', or 'greater'.
    method : str, optional
        The method to use for p-value computation. Must be 'exact' or 'normal'.

    Returns:
    --------
    result : dict
        A dictionary containing the test results, including the statistic,
        p-value, and other relevant information.
    """
    _validate_inputs(x, y)
    n = x.size

    statistic = _compute_sign_statistic(x, y)

    if method == 'exact':
        p_value = _compute_p_value(statistic, n)
    else:
        raise ValueError("Only 'exact' method is currently supported.")

    if alternative == 'two-sided':
        p_value = min(p_value, 1 - p_value)
    elif alternative == 'less':
        if statistic > n / 2:
            p_value = 1
    elif alternative == 'greater':
        if statistic < n / 2:
            p_value = 1
    else:
        raise ValueError("Alternative must be 'two-sided', 'less', or 'greater'.")

    result = {
        "statistic": statistic,
        "p_value": p_value,
        "n_samples": n,
        "alternative": alternative,
        "method": method
    }

    return result

# Example usage:
# x = np.array([1.2, 2.3, 3.4, 4.5])
# y = np.array([1.0, 2.0, 3.0, 4.0])
# result = test_sign_fit(x, y)

################################################################################
# test_friedman
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data for Friedman test."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 3:
        raise ValueError("Input data must be a 3D array (k blocks, n subjects, p conditions)")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or infinite values")

def _compute_rank_averages(data: np.ndarray) -> tuple:
    """Compute rank averages for each condition across blocks."""
    k, n, p = data.shape
    ranks = np.zeros_like(data)

    for i in range(k):
        # Rank data within each block
        ranked = np.argsort(np.argsort(data[i, :, :], axis=1), axis=1) + 1
        ranks[i, :, :] = ranked

    rank_averages = np.mean(ranks, axis=1)
    return ranks, rank_averages

def _compute_friedman_statistic(rank_averages: np.ndarray) -> float:
    """Compute the Friedman test statistic."""
    k, p = rank_averages.shape
    n = rank_averages.shape[1]

    # Calculate the Friedman statistic
    chi_square = (12 * n) / (k * (p + 1)) * np.sum((np.mean(rank_averages, axis=0) - (p + 1)/2)**2)
    return chi_square

def _compute_p_value(statistic: float, k: int) -> float:
    """Compute the p-value for the Friedman test statistic."""
    # Approximate using chi-square distribution with (k-1) degrees of freedom
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(statistic, df=k-1)
    return p_value

def test_friedman_fit(
    data: np.ndarray,
    normalization: Optional[str] = None,
    metric: Union[str, Callable] = "mean",
    p_value_method: str = "chi2"
) -> Dict[str, Union[float, Dict]]:
    """
    Perform the Friedman test for non-parametric repeated measures analysis.

    Parameters
    ----------
    data : np.ndarray
        3D array of shape (k blocks, n subjects, p conditions)
    normalization : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'
    metric : str or callable, optional
        Metric to compute: 'mean', 'median', custom callable
    p_value_method : str, optional
        Method for p-value computation: 'chi2' (default)

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': test statistic and p-value
        - 'metrics': computed metrics
        - 'params_used': parameters used
        - 'warnings': any warnings generated

    Example
    -------
    >>> data = np.random.rand(3, 5, 4)  # 3 blocks, 5 subjects, 4 conditions
    >>> result = test_friedman_fit(data)
    """
    # Validate inputs
    _validate_inputs(data)

    # Apply normalization if specified
    if normalization == 'standard':
        data = (data - np.mean(data, axis=(1, 2), keepdims=True)) / np.std(data, axis=(1, 2), keepdims=True)
    elif normalization == 'minmax':
        data = (data - np.min(data, axis=(1, 2), keepdims=True)) / (np.max(data, axis=(1, 2), keepdims=True) - np.min(data, axis=(1, 2), keepdims=True))
    elif normalization == 'robust':
        data = (data - np.median(data, axis=(1, 2), keepdims=True)) / (np.percentile(data, 75, axis=(1, 2), keepdims=True) - np.percentile(data, 25, axis=(1, 2), keepdims=True))

    # Compute rank averages
    ranks, rank_averages = _compute_rank_averages(data)

    # Compute test statistic
    statistic = _compute_friedman_statistic(rank_averages)

    # Compute p-value
    if p_value_method == 'chi2':
        p_value = _compute_p_value(statistic, data.shape[0])
    else:
        raise ValueError(f"Unknown p-value method: {p_value_method}")

    # Compute metrics
    if isinstance(metric, str):
        if metric == 'mean':
            metrics = {'mean_rank': np.mean(rank_averages, axis=1)}
        elif metric == 'median':
            metrics = {'median_rank': np.median(rank_averages, axis=1)}
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        metrics = {'custom_metric': metric(rank_averages)}

    # Prepare output
    result = {
        'statistic': statistic,
        'p_value': p_value
    }

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'p_value_method': p_value_method
        },
        'warnings': []
    }

################################################################################
# test_kolmogorov_smirnov
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def _validate_inputs(sample1: np.ndarray, sample2: Optional[np.ndarray] = None) -> None:
    """
    Validate input samples for Kolmogorov-Smirnov test.

    Parameters
    ----------
    sample1 : np.ndarray
        First sample data.
    sample2 : Optional[np.ndarray]
        Second sample data. If None, test against uniform distribution.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not isinstance(sample1, np.ndarray) or (sample2 is not None and not isinstance(sample2, np.ndarray)):
        raise ValueError("Inputs must be numpy arrays.")
    if sample1.ndim != 1 or (sample2 is not None and sample2.ndim != 1):
        raise ValueError("Inputs must be 1-dimensional.")
    if np.any(np.isnan(sample1)) or (sample2 is not None and np.any(np.isnan(sample2))):
        raise ValueError("Inputs must not contain NaN values.")
    if len(sample1) < 2 or (sample2 is not None and len(sample2) < 2):
        raise ValueError("Each sample must have at least 2 elements.")

def _compute_ks_statistic(sample1: np.ndarray, sample2: Optional[np.ndarray] = None) -> float:
    """
    Compute Kolmogorov-Smirnov statistic.

    Parameters
    ----------
    sample1 : np.ndarray
        First sample data.
    sample2 : Optional[np.ndarray]
        Second sample data. If None, test against uniform distribution.

    Returns
    ------
    float
        Kolmogorov-Smirnov statistic.
    """
    n1 = len(sample1)
    if sample2 is None:
        # Test against uniform distribution
        sample2 = np.random.uniform(size=n1)
    else:
        n2 = len(sample2)

    # Combine and sort the samples
    combined = np.concatenate([sample1, sample2])
    combined.sort()

    # Compute ECDFs
    n_combined = len(combined)
    ecdf1 = np.array([np.sum(sample1 <= x) / n1 for x in combined])
    ecdf2 = np.array([np.sum(sample2 <= x) / len(sample2) for x in combined])

    # Compute KS statistic
    ks_statistic = np.max(np.abs(ecdf1 - ecdf2))
    return ks_statistic

def _compute_p_value(ks_statistic: float, n1: int, n2: Optional[int] = None) -> float:
    """
    Compute p-value for Kolmogorov-Smirnov test.

    Parameters
    ----------
    n1 : int
        Size of first sample.
    n2 : Optional[int]
        Size of second sample. If None, test against uniform distribution.
    ks_statistic : float
        Kolmogorov-Smirnov statistic.

    Returns
    ------
    float
        p-value.
    """
    if n2 is None:
        # Approximation for uniform distribution
        p_value = 2 * np.sum([(-1)**(k+1) * np.exp(-2 * (k**2) * ks_statistic**2) for k in range(1, 100)])
    else:
        # Two-sample KS test p-value approximation
        en = np.sqrt(n1 * n2 / (n1 + n2))
        p_value = 2 * np.sum([(-1)**(k+1) * np.exp(-2 * (k**2) * en**2 * ks_statistic**2) for k in range(1, 100)])
    return p_value

def test_kolmogorov_smirnov_fit(
    sample1: np.ndarray,
    sample2: Optional[np.ndarray] = None,
    normalization: str = "none",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform Kolmogorov-Smirnov test with configurable options.

    Parameters
    ----------
    sample1 : np.ndarray
        First sample data.
    sample2 : Optional[np.ndarray]
        Second sample data. If None, test against uniform distribution.
    normalization : str
        Normalization method ("none", "standard", "minmax", "robust").
    custom_metric : Optional[Callable]
        Custom metric function.
    **kwargs
        Additional keyword arguments for future extensions.

    Returns
    ------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> sample1 = np.random.normal(size=100)
    >>> sample2 = np.random.normal(loc=1, size=100)
    >>> result = test_kolmogorov_smirnov_fit(sample1, sample2)
    """
    # Validate inputs
    _validate_inputs(sample1, sample2)

    # Apply normalization if specified
    if normalization != "none":
        if normalization == "standard":
            sample1 = (sample1 - np.mean(sample1)) / np.std(sample1)
            if sample2 is not None:
                sample2 = (sample2 - np.mean(sample2)) / np.std(sample2)
        elif normalization == "minmax":
            sample1 = (sample1 - np.min(sample1)) / (np.max(sample1) - np.min(sample1))
            if sample2 is not None:
                sample2 = (sample2 - np.min(sample2)) / (np.max(sample2) - np.min(sample2))
        elif normalization == "robust":
            sample1 = (sample1 - np.median(sample1)) / (np.percentile(sample1, 75) - np.percentile(sample1, 25))
            if sample2 is not None:
                sample2 = (sample2 - np.median(sample2)) / (np.percentile(sample2, 75) - np.percentile(sample2, 25))

    # Compute KS statistic
    ks_statistic = _compute_ks_statistic(sample1, sample2)

    # Compute p-value
    n1 = len(sample1)
    if sample2 is None:
        p_value = _compute_p_value(n1, None, ks_statistic)
    else:
        n2 = len(sample2)
        p_value = _compute_p_value(n1, n2, ks_statistic)

    # Compute metrics
    metrics = {}
    if custom_metric is not None:
        metrics["custom_metric"] = custom_metric(sample1, sample2)

    # Prepare results
    result = {
        "result": {
            "ks_statistic": ks_statistic,
            "p_value": p_value
        },
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "custom_metric": custom_metric is not None
        },
        "warnings": []
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
        raise ValueError("All values must be non-negative and expected counts must be positive.")
    if np.isnan(observed).any() or np.isnan(expected).any():
        raise ValueError("Inputs must not contain NaN values.")

def _compute_chi2_statistic(observed: np.ndarray, expected: np.ndarray) -> float:
    """
    Compute the chi-squared statistic.

    Parameters
    ----------
    observed : np.ndarray
        Observed frequency counts.
    expected : np.ndarray
        Expected frequency counts.

    Returns
    ------
    float
        The chi-squared statistic.
    """
    return np.sum((observed - expected) ** 2 / expected)

def _compute_p_value(chi2_stat: float, df: int) -> float:
    """
    Compute the p-value for the chi-squared statistic.

    Parameters
    ----------
    chi2_stat : float
        The chi-squared statistic.
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
    correction: bool = False,
    metric_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Perform a chi-squared goodness-of-fit test.

    Parameters
    ----------
    observed : np.ndarray
        Observed frequency counts.
    expected : np.ndarray
        Expected frequency counts.
    correction : bool, optional
        Whether to apply Yates' continuity correction (default: False).
    metric_func : Callable, optional
        Custom metric function to compute additional metrics.

    Returns
    ------
    Dict[str, Any]
        Dictionary containing test results, metrics, parameters used, and warnings.
    """
    _validate_inputs(observed, expected)

    # Compute chi-squared statistic
    if correction:
        observed_corrected = np.abs(observed - expected) - 0.5
        chi2_stat = _compute_chi2_statistic(observed_corrected, expected)
    else:
        chi2_stat = _compute_chi2_statistic(observed, expected)

    # Degrees of freedom
    df = len(observed) - 1

    # Compute p-value
    p_value = _compute_p_value(chi2_stat, df)

    # Compute additional metrics if custom function is provided
    metrics = {}
    if metric_func is not None:
        try:
            metrics["custom_metric"] = metric_func(observed, expected)
        except Exception as e:
            metrics["custom_metric_error"] = str(e)

    # Prepare output
    result = {
        "chi2_statistic": chi2_stat,
        "p_value": p_value,
        "degrees_of_freedom": df
    }

    return {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "correction": correction,
            "metric_func": metric_func.__name__ if metric_func else None
        },
        "warnings": []
    }

# Example usage:
"""
observed = np.array([10, 20, 30])
expected = np.array([15, 25, 25])
result = test_chi2_fit(observed, expected)
print(result)
"""

################################################################################
# test_fisher_exact
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(table: np.ndarray) -> None:
    """Validate the input contingency table."""
    if not isinstance(table, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if table.ndim != 2 or table.shape[0] != 2 or table.shape[1] != 2:
        raise ValueError("Input must be a 2x2 contingency table.")
    if np.any(table < 0):
        raise ValueError("All values in the contingency table must be non-negative.")
    if np.any(np.isnan(table)):
        raise ValueError("Contingency table contains NaN values.")
    if np.any(np.isinf(table)):
        raise ValueError("Contingency table contains infinite values.")

def _calculate_statistic(table: np.ndarray) -> float:
    """Calculate the Fisher exact test statistic."""
    a, b, c, d = table.ravel()
    odds_ratio = (a * d) / (b * c)
    return np.log(odds_ratio)

def _calculate_p_value(table: np.ndarray, alternative: str = 'two-sided') -> float:
    """Calculate the p-value for the Fisher exact test."""
    a, b, c, d = table.ravel()
    n = np.sum(table)
    p_value = 0.0

    if alternative == 'two-sided':
        max_prob = min(a + b, c + d, a + c)
        for i in range(max_prob + 1):
            k = min(i, a + b)
            l = i - k
            m = min(b, c) if (a + b < a + c) else 0
            n = min(b, c) if (a + b >= a + c) else 0
            p = np.math.factorial(a + b) * np.math.factorial(c + d) * np.math.factorial(a + c) * np.math.factorial(b + d)
            q = np.math.factorial(n) * np.math.factorial(a + b - k) * np.math.factorial(c + d - l) * np.math.factorial(k + l)
            p_value += p / q
        p_value = 1 - p_value
    elif alternative == 'less':
        max_prob = min(a + b, c + d)
        for i in range(max_prob + 1):
            k = min(i, a + b)
            l = i - k
            p = np.math.factorial(a + b) * np.math.factorial(c + d) * np.math.factorial(a + c) * np.math.factorial(b + d)
            q = np.math.factorial(n) * np.math.factorial(a + b - k) * np.math.factorial(c + d - l) * np.math.factorial(k + l)
            p_value += p / q
    elif alternative == 'greater':
        max_prob = min(a + b, c + d)
        for i in range(max_prob + 1):
            k = min(i, a + b)
            l = i - k
            p = np.math.factorial(a + b) * np.math.factorial(c + d) * np.math.factorial(a + c) * np.math.factorial(b + d)
            q = np.math.factorial(n) * np.math.factorial(a + b - k) * np.math.factorial(c + d - l) * np.math.factorial(k + l)
            p_value += p / q
        p_value = 1 - p_value

    return p_value

def test_fisher_exact_fit(
    table: np.ndarray,
    alternative: str = 'two-sided',
    correction: bool = False
) -> Dict[str, Union[float, Dict]]:
    """
    Perform Fisher's exact test for a 2x2 contingency table.

    Parameters
    ----------
    table : np.ndarray
        A 2x2 contingency table.
    alternative : str, optional
        The alternative hypothesis. Must be 'two-sided', 'less', or 'greater'.
    correction : bool, optional
        Whether to apply a continuity correction.

    Returns
    -------
    result : dict
        A dictionary containing the test results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> table = np.array([[10, 5], [2, 8]])
    >>> result = test_fisher_exact_fit(table)
    """
    _validate_inputs(table)

    statistic = _calculate_statistic(table)
    p_value = _calculate_p_value(table, alternative)

    if correction:
        a, b, c, d = table.ravel()
        n = np.sum(table)
        continuity_correction = 0.5 * (np.abs(a * d - b * c) / np.sqrt((a + b) * (c + d) * a * c * b * d / n))
        p_value = 2 * (1 - np.normal.cdf(continuity_correction))

    result = {
        "result": {
            "statistic": statistic,
            "p_value": p_value
        },
        "metrics": {},
        "params_used": {
            "alternative": alternative,
            "correction": correction
        },
        "warnings": []
    }

    return result
