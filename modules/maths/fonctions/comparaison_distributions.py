"""
Quantix – Module comparaison_distributions
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# test_chi2
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def _validate_inputs(observed: np.ndarray, expected: np.ndarray) -> None:
    """Validate input arrays for chi-squared test."""
    if observed.ndim != 1 or expected.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if observed.shape != expected.shape:
        raise ValueError("Observed and expected arrays must have the same shape.")
    if np.any(observed < 0) or np.any(expected <= 0):
        raise ValueError("All values must be non-negative, and expected values must be positive.")
    if np.isnan(observed).any() or np.isnan(expected).any():
        raise ValueError("Input arrays must not contain NaN values.")
    if np.isinf(observed).any() or np.isinf(expected).any():
        raise ValueError("Input arrays must not contain infinite values.")

def _compute_chi2_statistic(observed: np.ndarray, expected: np.ndarray) -> float:
    """Compute the chi-squared statistic."""
    return np.sum((observed - expected) ** 2 / expected)

def _compute_p_value(chi2_stat: float, df: int) -> float:
    """Compute the p-value for the chi-squared statistic."""
    from scipy.stats import chi2
    return 1 - chi2.cdf(chi2_stat, df)

def _compute_metrics(
    observed: np.ndarray,
    expected: np.ndarray,
    metric_funcs: Dict[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute additional metrics using provided callables."""
    return {name: func(observed, expected) for name, func in metric_funcs.items()}

def test_chi2_fit(
    observed: np.ndarray,
    expected: np.ndarray,
    metric_funcs: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
    normalize: bool = False
) -> Dict[str, Any]:
    """
    Perform a chi-squared goodness-of-fit test.

    Parameters:
    -----------
    observed : np.ndarray
        Observed frequency counts.
    expected : np.ndarray
        Expected frequency counts under the null hypothesis.
    metric_funcs : Optional[Dict[str, Callable]]
        Dictionary of additional metrics to compute.
    normalize : bool
        Whether to normalize the expected frequencies.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    _validate_inputs(observed, expected)

    if normalize:
        expected = expected / np.sum(expected) * np.sum(observed)

    chi2_stat = _compute_chi2_statistic(observed, expected)
    df = len(observed) - 1
    p_value = _compute_p_value(chi2_stat, df)

    metrics = {}
    if metric_funcs is not None:
        metrics.update(_compute_metrics(observed, expected, metric_funcs))

    result = {
        "statistic": chi2_stat,
        "p_value": p_value,
        "degrees_of_freedom": df
    }

    warnings = []
    if p_value < 0.05:
        warnings.append("The null hypothesis may be rejected at the 5% significance level.")

    return {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "normalize": normalize
        },
        "warnings": warnings
    }

# Example usage:
"""
observed = np.array([10, 20, 30])
expected = np.array([15, 25, 25])

def mse(obs: np.ndarray, exp: np.ndarray) -> float:
    return np.mean((obs - exp) ** 2)

metrics = {"mse": mse}
result = test_chi2_fit(observed, expected, metric_funcs=metrics)
print(result)
"""

################################################################################
# test_kolmogorov_smirnov
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def _validate_inputs(sample1: np.ndarray, sample2: np.ndarray) -> None:
    """Validate input samples for Kolmogorov-Smirnov test."""
    if not isinstance(sample1, np.ndarray) or not isinstance(sample2, np.ndarray):
        raise TypeError("Samples must be numpy arrays")
    if sample1.ndim != 1 or sample2.ndim != 1:
        raise ValueError("Samples must be 1-dimensional")
    if len(sample1) == 0 or len(sample2) == 0:
        raise ValueError("Samples cannot be empty")
    if np.any(np.isnan(sample1)) or np.any(np.isnan(sample2)):
        raise ValueError("Samples cannot contain NaN values")
    if np.any(np.isinf(sample1)) or np.any(np.isinf(sample2)):
        raise ValueError("Samples cannot contain infinite values")

def _compute_ks_statistic(sample1: np.ndarray, sample2: np.ndarray) -> float:
    """Compute the Kolmogorov-Smirnov statistic between two samples."""
    n1, n2 = len(sample1), len(sample2)
    data = np.concatenate([sample1, sample2])
    sorted_data = np.sort(data)
    cdf1 = np.searchsorted(sample1, sorted_data, side='right') / n1
    cdf2 = np.searchsorted(sample2, sorted_data, side='right') / n2
    return np.max(np.abs(cdf1 - cdf2))

def _compute_p_value(ks_stat: float, n1: int, n2: int) -> float:
    """Compute the p-value for the Kolmogorov-Smirnov test."""
    en = np.sqrt((n1 + n2) / (n1 * n2))
    return 2 * np.sum([(-1)**(k+1) * np.exp(-2 * (k**2) * ks_stat**2 * en**2)
                       for k in range(1, 100)])

def test_kolmogorov_smirnov_fit(
    sample1: np.ndarray,
    sample2: np.ndarray,
    *,
    normalization: Optional[str] = None,
    metric: str = 'ks_stat',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform the Kolmogorov-Smirnov test between two samples.

    Parameters:
    -----------
    sample1 : np.ndarray
        First sample of data points.
    sample2 : np.ndarray
        Second sample of data points.
    normalization : str, optional
        Normalization method to apply (none, standard, minmax, robust).
    metric : str
        Metric to compute ('ks_stat', 'p_value').
    custom_metric : Callable, optional
        Custom metric function.
    **kwargs :
        Additional keyword arguments for future extensions.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.

    Example:
    --------
    >>> sample1 = np.random.normal(0, 1, 100)
    >>> sample2 = np.random.normal(0.5, 1, 100)
    >>> result = test_kolmogorov_smirnov_fit(sample1, sample2)
    """
    _validate_inputs(sample1, sample2)

    # Apply normalization if specified
    if normalization == 'standard':
        mean1, std1 = np.mean(sample1), np.std(sample1)
        mean2, std2 = np.mean(sample2), np.std(sample2)
        sample1 = (sample1 - mean1) / std1
        sample2 = (sample2 - mean2) / std2
    elif normalization == 'minmax':
        min1, max1 = np.min(sample1), np.max(sample1)
        min2, max2 = np.min(sample2), np.max(sample2)
        sample1 = (sample1 - min1) / (max1 - min1)
        sample2 = (sample2 - min2) / (max2 - min2)

    # Compute KS statistic
    ks_stat = _compute_ks_statistic(sample1, sample2)

    # Compute p-value if requested
    p_value = None
    if metric == 'p_value':
        p_value = _compute_p_value(ks_stat, len(sample1), len(sample2))

    # Compute custom metric if provided
    custom_metric_value = None
    if custom_metric is not None:
        custom_metric_value = custom_metric(sample1, sample2)

    return {
        'result': ks_stat if metric == 'ks_stat' else p_value,
        'metrics': {
            'ks_statistic': ks_stat,
            'p_value': p_value if metric == 'p_value' else None,
            'custom_metric': custom_metric_value
        },
        'params_used': {
            'normalization': normalization,
            'metric': metric
        },
        'warnings': []
    }

################################################################################
# test_anderson_darling
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(data: np.ndarray, distribution: str = 'normal') -> None:
    """Validate input data and distribution type."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if len(data.shape) != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or infinite values")
    if distribution not in ['normal', 'exponential', 'weibull']:
        raise ValueError(f"Unsupported distribution: {distribution}")

def _compute_statistic(data: np.ndarray, distribution: str = 'normal') -> float:
    """Compute the Anderson-Darling statistic for the given data and distribution."""
    n = len(data)
    sorted_data = np.sort(data)

    if distribution == 'normal':
        mean, std = np.mean(sorted_data), np.std(sorted_data)
        z = (sorted_data - mean) / std
    elif distribution == 'exponential':
        scale = np.mean(sorted_data)
        z = -np.log(1 - (sorted_data / scale))
    elif distribution == 'weibull':
        shape = _estimate_weibull_shape(sorted_data)
        scale = np.mean(sorted_data) / (1 + 1/shape)
        z = np.log(sorted_data) - (-np.log(-np.log(1 - (sorted_data / scale))))

    i = np.arange(1, n + 1)
    statistic = -n - (2 / n) * np.sum((2 * i - 1) * (np.log(z[i-1]) + np.log(1 - z[n-i])))
    return statistic

def _estimate_weibull_shape(data: np.ndarray) -> float:
    """Estimate the shape parameter for Weibull distribution using maximum likelihood."""
    n = len(data)
    log_data = np.log(data)
    sum_log = np.sum(log_data)

    def objective(shape: float) -> float:
        term1 = -n * np.log(shape)
        term2 = (shape - 1) / shape * sum_log
        return -(term1 + term2)

    # Simple optimization (in practice, use scipy.optimize)
    shapes = np.linspace(0.1, 5, 100)
    objectives = [objective(shape) for shape in shapes]
    best_shape = shapes[np.argmin(objectives)]
    return best_shape

def _compute_critical_values(distribution: str = 'normal') -> Dict[str, float]:
    """Compute critical values for the Anderson-Darling test."""
    if distribution == 'normal':
        return {
            '1%': 0.787,
            '2.5%': 0.656,
            '5%': 0.543,
            '10%': 0.461
        }
    elif distribution == 'exponential':
        return {
            '1%': 0.245,
            '2.5%': 0.319,
            '5%': 0.412,
            '10%': 0.539
        }
    elif distribution == 'weibull':
        return {
            '1%': 0.342,
            '2.5%': 0.416,
            '5%': 0.502,
            '10%': 0.607
        }

def test_anderson_darling_fit(
    data: np.ndarray,
    distribution: str = 'normal',
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Perform the Anderson-Darling test to compare sample distribution with a reference distribution.

    Parameters:
    -----------
    data : np.ndarray
        Input data to test.
    distribution : str, optional
        Reference distribution ('normal', 'exponential', 'weibull'), default='normal'.
    significance_level : float, optional
        Significance level for the test (0 < level < 1), default=0.05.

    Returns:
    --------
    dict
        Dictionary containing test results, metrics, parameters used, and warnings.
    """
    _validate_inputs(data, distribution)

    statistic = _compute_statistic(data, distribution)
    critical_values = _compute_critical_values(distribution)

    # Determine if we reject the null hypothesis
    critical_value = next((v for k, v in sorted(critical_values.items(), reverse=True)
                          if float(k.replace('%', '')) / 100 <= significance_level), None)

    result = {
        'statistic': statistic,
        'critical_value': critical_value,
        'significance_level': significance_level,
        'reject_null': statistic > critical_value
    }

    return {
        'result': result,
        'metrics': {},
        'params_used': {'distribution': distribution, 'significance_level': significance_level},
        'warnings': []
    }

# Example usage:
# result = test_anderson_darling_fit(np.random.normal(0, 1, 100), distribution='normal')

################################################################################
# test_shapiro_wilk
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for Shapiro-Wilk test."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if len(data.shape) != 1:
        raise ValueError("Input data must be one-dimensional.")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data must not contain NaN or infinite values.")
    if len(data) < 3:
        raise ValueError("Input data must have at least 3 samples.")

def _shapiro_wilk_statistic(data: np.ndarray) -> float:
    """Calculate the Shapiro-Wilk test statistic."""
    n = len(data)
    data_sorted = np.sort(data)

    # Calculate the expected values under normality
    expected_values = np.array([np.mean(data) + (i - 0.5) * np.std(data) / (n / 2)
                               for i in range(1, n + 1)])

    # Calculate the covariance matrix
    cov_matrix = np.cov(data_sorted, expected_values)

    # Calculate the test statistic
    numerator = np.sum((data_sorted - np.mean(data)) ** 2)
    denominator = cov_matrix[0, 1] * n
    statistic = numerator / denominator

    return statistic

def _shapiro_wilk_p_value(statistic: float, n: int) -> float:
    """Calculate the p-value for the Shapiro-Wilk test."""
    # This is a simplified approximation; in practice, you would use precomputed tables or interpolation
    if n < 3:
        raise ValueError("Sample size must be at least 3.")
    if n > 5000:
        raise ValueError("Sample size must be less than or equal to 5000.")

    # Approximation for p-value (simplified)
    p_value = 1 - np.exp(-3.67 * statistic + 0.48 * n ** 0.5 - 1.3)
    return max(0, min(1, p_value))

def test_shapiro_wilk_fit(
    data: np.ndarray,
    normalization: Optional[str] = None,
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    **kwargs
) -> Dict[str, Union[float, Dict, str]]:
    """
    Perform the Shapiro-Wilk test for normality.

    Parameters
    ----------
    data : np.ndarray
        Input data to test for normality.
    normalization : str, optional
        Type of normalization to apply. Options: 'none', 'standard', 'minmax', 'robust'.
    custom_normalization : callable, optional
        Custom normalization function.
    **kwargs :
        Additional keyword arguments for future extensions.

    Returns
    -------
    dict
        Dictionary containing the test results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.random.normal(0, 1, 100)
    >>> result = test_shapiro_wilk_fit(data, normalization='standard')
    """
    # Validate input data
    _validate_input(data)

    # Apply normalization if specified
    if normalization == 'standard':
        data = (data - np.mean(data)) / np.std(data)
    elif normalization == 'minmax':
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
    elif normalization == 'robust':
        data = (data - np.median(data)) / (np.percentile(data, 75) - np.percentile(data, 25))
    elif custom_normalization is not None:
        data = custom_normalization(data)

    # Calculate Shapiro-Wilk statistic
    statistic = _shapiro_wilk_statistic(data)
    n = len(data)

    # Calculate p-value
    p_value = _shapiro_wilk_p_value(statistic, n)

    # Prepare the result dictionary
    result = {
        "result": {
            "statistic": statistic,
            "p_value": p_value
        },
        "metrics": {},
        "params_used": {
            "normalization": normalization,
            "custom_normalization": custom_normalization is not None
        },
        "warnings": []
    }

    return result

################################################################################
# test_jarque_bera
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for Jarque-Bera test."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if len(data.shape) != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values")
    if len(data) < 3:
        raise ValueError("Input data must have at least 3 samples")

def _compute_skewness(data: np.ndarray) -> float:
    """Compute sample skewness."""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    if std == 0:
        return 0.0
    skewness = (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)
    return skewness

def _compute_kurtosis(data: np.ndarray) -> float:
    """Compute sample kurtosis."""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    if std == 0:
        return 0.0
    kurtosis = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((data - mean) / std) ** 4) - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return kurtosis

def _jarque_bera_statistic(skewness: float, kurtosis: float) -> float:
    """Compute Jarque-Bera test statistic."""
    n = len(skewness)
    jb_statistic = (n / 6) * (skewness ** 2 + ((kurtosis - 3) ** 2) / 4)
    return jb_statistic

def _jarque_bera_p_value(jb_statistic: float, n: int) -> float:
    """Compute p-value for Jarque-Bera test statistic."""
    from scipy.stats import chi2
    df = 2
    p_value = 1 - chi2.cdf(jb_statistic, df)
    return p_value

def test_jarque_bera_fit(
    data: np.ndarray,
    normalize: bool = False,
    normalization_method: str = "standard",
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    **kwargs
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Perform Jarque-Bera test for normality.

    Parameters
    ----------
    data : np.ndarray
        Input data to test for normality.
    normalize : bool, optional
        Whether to normalize the data before testing. Default is False.
    normalization_method : str, optional
        Method for normalization ('standard', 'minmax', 'robust'). Default is 'standard'.
    custom_normalization : callable, optional
        Custom normalization function. If provided, overrides normalization_method.

    Returns
    -------
    dict
        Dictionary containing test results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.random.normal(0, 1, 100)
    >>> result = test_jarque_bera_fit(data)
    """
    _validate_input(data)

    # Normalization
    if normalize:
        if custom_normalization is not None:
            data = custom_normalization(data)
        else:
            if normalization_method == "standard":
                data = (data - np.mean(data)) / np.std(data)
            elif normalization_method == "minmax":
                data = (data - np.min(data)) / (np.max(data) - np.min(data))
            elif normalization_method == "robust":
                median = np.median(data)
                iqr = np.percentile(data, 75) - np.percentile(data, 25)
                data = (data - median) / iqr
            else:
                raise ValueError("Invalid normalization method")

    # Compute skewness and kurtosis
    skewness = _compute_skewness(data)
    kurtosis = _compute_kurtosis(data)

    # Compute test statistic and p-value
    jb_statistic = _jarque_bera_statistic(skewness, kurtosis)
    p_value = _jarque_bera_p_value(jb_statistic, len(data))

    # Prepare results
    result = {
        "result": {
            "statistic": jb_statistic,
            "p_value": p_value
        },
        "metrics": {
            "skewness": skewness,
            "kurtosis": kurtosis
        },
        "params_used": {
            "normalize": normalize,
            "normalization_method": normalization_method if not custom_normalization else "custom",
        },
        "warnings": []
    }

    return result

################################################################################
# test_lilliefors
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data for Lilliefors test."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data cannot contain NaN or infinite values")

def _compute_empirical_cdf(data: np.ndarray) -> tuple:
    """Compute empirical CDF and sorted data."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    cdf_values = np.arange(1, n + 1) / n
    return sorted_data, cdf_values

def _compute_theoretical_cdf(data: np.ndarray,
                            loc: float = 0.0,
                            scale: float = 1.0) -> np.ndarray:
    """Compute theoretical CDF for normal distribution."""
    return (data - loc) / scale

def _compute_lilliefors_statistic(sorted_data: np.ndarray,
                                 empirical_cdf: np.ndarray) -> float:
    """Compute Lilliefors test statistic."""
    n = len(sorted_data)
    max_diff = 0.0
    for i in range(n):
        diff = abs(empirical_cdf[i] - _compute_theoretical_cdf(sorted_data[i]))
        if diff > max_diff:
            max_diff = diff
    return max_diff

def _compute_p_value(statistic: float) -> float:
    """Compute p-value for Lilliefors test statistic."""
    # This is a placeholder - in practice you would use a lookup table or approximation
    return 1.0 - statistic

def test_lilliefors_fit(data: np.ndarray,
                        loc: float = 0.0,
                        scale: float = 1.0,
                        custom_cdf: Optional[Callable] = None) -> Dict[str, Union[float, Dict]]:
    """
    Perform Lilliefors test for normality.

    Parameters:
    -----------
    data : np.ndarray
        Input data to test for normality
    loc : float, optional
        Location parameter for normal distribution (default 0.0)
    scale : float, optional
        Scale parameter for normal distribution (default 1.0)
    custom_cdf : Callable, optional
        Custom CDF function to use instead of normal distribution

    Returns:
    --------
    dict
        Dictionary containing test results, metrics, and parameters used

    Example:
    --------
    >>> data = np.random.normal(0, 1, 100)
    >>> result = test_lilliefors_fit(data)
    """
    _validate_inputs(data)

    sorted_data, empirical_cdf = _compute_empirical_cdf(data)

    if custom_cdf is not None:
        theoretical_cdf = custom_cdf(sorted_data)
    else:
        theoretical_cdf = _compute_theoretical_cdf(sorted_data, loc, scale)

    statistic = _compute_lilliefors_statistic(sorted_data, empirical_cdf)
    p_value = _compute_p_value(statistic)

    result = {
        "result": {
            "statistic": statistic,
            "p_value": p_value
        },
        "metrics": {
            "max_difference": statistic
        },
        "params_used": {
            "loc": loc,
            "scale": scale,
            "custom_cdf": custom_cdf is not None
        },
        "warnings": []
    }

    return result

################################################################################
# test_watson_u2
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for Watson U2 test."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Input arrays must be 1-dimensional")
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("Input array x contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Input array y contains NaN or infinite values")

def _compute_watson_u2_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Watson U2 test statistic."""
    n = len(x)
    combined = np.concatenate([x, y])
    sorted_combined = np.sort(combined)
    ranks_x = np.array([np.sum(sorted_combined < xi) for xi in x])
    ranks_y = np.array([np.sum(sorted_combined < yi) for yi in y])
    u2 = (np.sum(ranks_x**3) + np.sum((n + ranks_y)**3)) / (n**3)
    return u2

def _compute_p_value(u2: float, n: int) -> float:
    """Compute the p-value for the Watson U2 test statistic."""
    # Approximation using normal distribution
    mean = 0.5 * (n + 1) / 3
    variance = (n + 1) * (5*n**2 - 6*n + 3) / 90
    z = (u2 - mean) / np.sqrt(variance)
    p_value = 1 - norm.cdf(z)
    return p_value

def test_watson_u2_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    normalization: str = "none",
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform the Watson U2 test for comparing two distributions.

    Parameters:
    -----------
    x : np.ndarray
        First sample of data points.
    y : np.ndarray
        Second sample of data points.
    normalization : str, optional
        Type of normalization to apply ("none", "standard", "minmax", "robust").
    custom_normalization : callable, optional
        Custom normalization function.

    Returns:
    --------
    dict
        Dictionary containing test results, metrics, parameters used, and warnings.
    """
    _validate_inputs(x, y)

    # Apply normalization
    if normalization == "standard":
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
    elif normalization == "minmax":
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif normalization == "robust":
        x = (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25))
        y = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    elif custom_normalization is not None:
        x = custom_normalization(x)
        y = custom_normalization(y)

    # Compute test statistic
    u2 = _compute_watson_u2_statistic(x, y)
    n = len(x)
    p_value = _compute_p_value(u2, n)

    # Prepare results
    result = {
        "statistic": u2,
        "p_value": p_value,
        "null_hypothesis": "The two distributions are the same",
        "alternative_hypothesis": "The two distributions are different"
    }

    metrics = {
        "test_statistic": u2,
        "p_value": p_value
    }

    params_used = {
        "normalization": normalization,
        "custom_normalization": custom_normalization is not None
    }

    warnings = []

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

################################################################################
# test_cramer_von_mises
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    sample: np.ndarray,
    distribution: Callable,
    params: Optional[np.ndarray] = None
) -> None:
    """Validate inputs for Cramer-von Mises test."""
    if not isinstance(sample, np.ndarray):
        raise TypeError("Sample must be a numpy array")
    if not callable(distribution):
        raise TypeError("Distribution must be a callable function")
    if params is not None and not isinstance(params, np.ndarray):
        raise TypeError("Parameters must be a numpy array or None")
    if len(sample.shape) != 1:
        raise ValueError("Sample must be a 1-dimensional array")
    if np.any(np.isnan(sample)):
        raise ValueError("Sample contains NaN values")
    if params is not None and np.any(np.isnan(params)):
        raise ValueError("Parameters contain NaN values")

def _compute_cv_statistic(
    sample: np.ndarray,
    distribution: Callable,
    params: Optional[np.ndarray] = None
) -> float:
    """Compute the Cramer-von Mises statistic."""
    n = len(sample)
    sample_sorted = np.sort(sample)

    if params is None:
        # Estimate parameters using MLE (simplified for demonstration)
        params = _estimate_parameters(sample, distribution)

    cdf_values = np.array([distribution(x, *params) for x in sample_sorted])
    empirical_cdf = np.arange(1, n + 1) / n
    differences = cdf_values - empirical_cdf

    cv_statistic = (1 / (12 * n)) + np.sum(differences**2) - (n / (4 * (n - 1)))
    return cv_statistic

def _estimate_parameters(
    sample: np.ndarray,
    distribution: Callable
) -> np.ndarray:
    """Estimate parameters for the given distribution."""
    # Simplified MLE estimation (to be replaced with actual implementation)
    if distribution.__name__ == 'norm':
        return np.array([np.mean(sample), np.std(sample)])
    elif distribution.__name__ == 'uniform':
        return np.array([np.min(sample), np.max(sample)])
    else:
        raise NotImplementedError("Parameter estimation not implemented for this distribution")

def test_cramer_von_mises_fit(
    sample: np.ndarray,
    distribution: Callable,
    params: Optional[np.ndarray] = None,
    normalize: str = 'none',
    metric: str = 'mse',
    custom_metric: Optional[Callable] = None,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, Union[float, Dict[str, float], np.ndarray, str]]:
    """
    Perform the Cramer-von Mises test for goodness of fit.

    Parameters:
    - sample: Observed data
    - distribution: Probability distribution function (callable)
    - params: Parameters for the distribution
    - normalize: Normalization method ('none', 'standard', 'minmax')
    - metric: Metric for optimization ('mse', 'mae', 'custom')
    - custom_metric: Custom metric function if needed
    - tol: Tolerance for convergence
    - max_iter: Maximum number of iterations

    Returns:
    Dictionary containing results, metrics, parameters used, and warnings
    """
    _validate_inputs(sample, distribution, params)

    # Normalize sample if needed
    if normalize == 'standard':
        sample = (sample - np.mean(sample)) / np.std(sample)
    elif normalize == 'minmax':
        sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))

    # Compute CV statistic
    cv_statistic = _compute_cv_statistic(sample, distribution, params)

    # Calculate p-value (simplified for demonstration)
    p_value = _compute_p_value(cv_statistic, len(sample))

    # Prepare metrics
    metrics = {
        'cv_statistic': cv_statistic,
        'p_value': p_value
    }

    # Prepare result dictionary
    result = {
        'result': cv_statistic,
        'metrics': metrics,
        'params_used': params if params is not None else _estimate_parameters(sample, distribution),
        'warnings': []
    }

    return result

def _compute_p_value(
    cv_statistic: float,
    n_samples: int
) -> float:
    """Compute p-value for the Cramer-von Mises statistic."""
    # Simplified approximation (to be replaced with actual implementation)
    return 1 - np.exp(-6 * cv_statistic * n_samples)

# Example usage:
"""
from scipy.stats import norm

sample = np.random.normal(0, 1, 100)
result = test_cramer_von_mises_fit(sample, norm)
print(result)
"""

################################################################################
# test_energy_distance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def test_energy_distance_fit(
    sample1: np.ndarray,
    sample2: np.ndarray,
    distance_metric: Union[str, Callable] = 'euclidean',
    normalization: str = 'none',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Perform the energy distance test between two samples.

    Parameters:
    -----------
    sample1 : np.ndarray
        First sample of data points.
    sample2 : np.ndarray
        Second sample of data points.
    distance_metric : str or callable, optional
        Distance metric to use. Can be 'euclidean', 'manhattan', 'cosine',
        'minkowski', or a custom callable.
    normalization : str, optional
        Normalization method. Can be 'none', 'standard', 'minmax', or 'robust'.
    solver : str, optional
        Solver to use. Can be 'closed_form', 'gradient_descent', 'newton',
        or 'coordinate_descent'.
    regularization : str, optional
        Regularization method. Can be 'none', 'l1', 'l2', or 'elasticnet'.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(sample1, sample2)

    # Normalize data if required
    normalized_sample1, normalized_sample2 = _apply_normalization(
        sample1, sample2, normalization
    )

    # Compute energy distance statistic
    statistic = _compute_energy_distance(
        normalized_sample1, normalized_sample2,
        distance_metric, custom_metric
    )

    # Solve for parameters if needed
    params = _solve_parameters(
        statistic, solver, regularization,
        tol, max_iter, **kwargs
    )

    # Compute metrics
    metrics = _compute_metrics(statistic, params)

    return {
        'result': statistic,
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

def _validate_inputs(sample1: np.ndarray, sample2: np.ndarray) -> None:
    """Validate input samples."""
    if not isinstance(sample1, np.ndarray) or not isinstance(sample2, np.ndarray):
        raise TypeError("Samples must be numpy arrays.")
    if sample1.ndim != 2 or sample2.ndim != 2:
        raise ValueError("Samples must be 2-dimensional.")
    if sample1.shape[1] != sample2.shape[1]:
        raise ValueError("Samples must have the same number of features.")
    if np.any(np.isnan(sample1)) or np.any(np.isnan(sample2)):
        raise ValueError("Samples must not contain NaN values.")
    if np.any(np.isinf(sample1)) or np.any(np.isinf(sample2)):
        raise ValueError("Samples must not contain infinite values.")

def _apply_normalization(
    sample1: np.ndarray,
    sample2: np.ndarray,
    normalization: str
) -> tuple:
    """Apply normalization to samples."""
    if normalization == 'standard':
        mean1, std1 = np.mean(sample1, axis=0), np.std(sample1, axis=0)
        mean2, std2 = np.mean(sample2, axis=0), np.std(sample2, axis=0)
        normalized_sample1 = (sample1 - mean1) / std1
        normalized_sample2 = (sample2 - mean2) / std2
    elif normalization == 'minmax':
        min1, max1 = np.min(sample1, axis=0), np.max(sample1, axis=0)
        min2, max2 = np.min(sample2, axis=0), np.max(sample2, axis=0)
        normalized_sample1 = (sample1 - min1) / (max1 - min1 + 1e-8)
        normalized_sample2 = (sample2 - min2) / (max2 - min2 + 1e-8)
    elif normalization == 'robust':
        med1, iqr1 = np.median(sample1, axis=0), np.percentile(sample1, 75, axis=0) - np.percentile(sample1, 25, axis=0)
        med2, iqr2 = np.median(sample2, axis=0), np.percentile(sample2, 75, axis=0) - np.percentile(sample2, 25, axis=0)
        normalized_sample1 = (sample1 - med1) / (iqr1 + 1e-8)
        normalized_sample2 = (sample2 - med2) / (iqr2 + 1e-8)
    else:
        normalized_sample1, normalized_sample2 = sample1, sample2
    return normalized_sample1, normalized_sample2

def _compute_energy_distance(
    sample1: np.ndarray,
    sample2: np.ndarray,
    distance_metric: Union[str, Callable],
    custom_metric: Optional[Callable] = None
) -> float:
    """Compute the energy distance statistic."""
    n1, n2 = sample1.shape[0], sample2.shape[0]
    combined = np.vstack((sample1, sample2))
    n = combined.shape[0]

    if callable(distance_metric):
        distance_func = distance_metric
    elif distance_metric == 'euclidean':
        distance_func = lambda x, y: np.linalg.norm(x - y)
    elif distance_metric == 'manhattan':
        distance_func = lambda x, y: np.sum(np.abs(x - y))
    elif distance_metric == 'cosine':
        distance_func = lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    elif distance_metric == 'minkowski':
        def distance_func(x, y):
            return np.sum(np.abs(x - y) ** 3) ** (1/3)
    else:
        raise ValueError("Unsupported distance metric.")

    if custom_metric is not None:
        distance_func = custom_metric

    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = distance_func(combined[i], combined[j])

    W1 = np.ones(n1) / n1
    W2 = np.ones(n2) / n2

    energy_distance = 2 * (np.sum(W1[:, np.newaxis] * distances[:n1, :n1]) +
                          np.sum(W2[:, np.newaxis] * distances[n1:, n1:])) - np.sum((W1[:, np.newaxis] + W2[:, np.newaxis]) * distances[:n1, n1:])

    return energy_distance

def _solve_parameters(
    statistic: float,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    **kwargs
) -> Dict:
    """Solve for parameters using the specified solver."""
    params = {
        'statistic': statistic,
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'max_iter': max_iter
    }
    return params

def _compute_metrics(
    statistic: float,
    params: Dict
) -> Dict:
    """Compute metrics based on the energy distance statistic."""
    return {
        'energy_distance': statistic,
        'solver_used': params['solver'],
        'regularization_used': params['regularization']
    }

# Example usage:
# result = test_energy_distance_fit(
#     sample1=np.random.randn(100, 5),
#     sample2=np.random.randn(100, 5),
#     distance_metric='euclidean',
#     normalization='standard'
# )

################################################################################
# test_wasserstein_distance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    sample1: np.ndarray,
    sample2: np.ndarray,
    distance_metric: Union[str, Callable],
    normalize: str = "none",
) -> None:
    """Validate input samples and parameters."""
    if sample1.ndim != 2 or sample2.ndim != 2:
        raise ValueError("Samples must be 2-dimensional arrays.")
    if sample1.shape[0] != sample2.shape[0]:
        raise ValueError("Samples must have the same number of observations.")
    if normalize not in ["none", "standard", "minmax", "robust"]:
        raise ValueError("Invalid normalization method.")
    if isinstance(distance_metric, str) and distance_metric not in ["euclidean", "manhattan", "cosine"]:
        raise ValueError("Invalid distance metric.")

def _normalize_samples(
    sample1: np.ndarray,
    sample2: np.ndarray,
    method: str = "none",
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize samples based on the specified method."""
    if method == "standard":
        mean1, std1 = np.mean(sample1, axis=0), np.std(sample1, axis=0)
        mean2, std2 = np.mean(sample2, axis=0), np.std(sample2, axis=0)
        sample1 = (sample1 - mean1) / std1
        sample2 = (sample2 - mean2) / std2
    elif method == "minmax":
        min1, max1 = np.min(sample1, axis=0), np.max(sample1, axis=0)
        min2, max2 = np.min(sample2, axis=0), np.max(sample2, axis=0)
        sample1 = (sample1 - min1) / (max1 - min1 + 1e-8)
        sample2 = (sample2 - min2) / (max2 - min2 + 1e-8)
    elif method == "robust":
        med1, iqr1 = np.median(sample1, axis=0), np.percentile(sample1, 75, axis=0) - np.percentile(sample1, 25, axis=0)
        med2, iqr2 = np.median(sample2, axis=0), np.percentile(sample2, 75, axis=0) - np.percentile(sample2, 25, axis=0)
        sample1 = (sample1 - med1) / (iqr1 + 1e-8)
        sample2 = (sample2 - med2) / (iqr2 + 1e-8)
    return sample1, sample2

def _compute_distance_matrix(
    sample1: np.ndarray,
    sample2: np.ndarray,
    distance_metric: Union[str, Callable],
) -> np.ndarray:
    """Compute the distance matrix between two samples."""
    if isinstance(distance_metric, str):
        if distance_metric == "euclidean":
            return np.sqrt(np.sum((sample1[:, np.newaxis] - sample2) ** 2, axis=2))
        elif distance_metric == "manhattan":
            return np.sum(np.abs(sample1[:, np.newaxis] - sample2), axis=2)
        elif distance_metric == "cosine":
            return 1 - np.sum(sample1[:, np.newaxis] * sample2, axis=2) / (
                np.linalg.norm(sample1[:, np.newaxis], axis=2) * np.linalg.norm(sample2, axis=2)
            )
    else:
        return distance_metric(sample1[:, np.newaxis], sample2)

def _solve_wasserstein(
    distance_matrix: np.ndarray,
    solver: str = "closed_form",
) -> float:
    """Solve the Wasserstein distance using the specified solver."""
    if solver == "closed_form":
        return np.mean(np.min(distance_matrix, axis=1))
    else:
        raise NotImplementedError("Other solvers are not implemented yet.")

def test_wasserstein_distance_fit(
    sample1: np.ndarray,
    sample2: np.ndarray,
    distance_metric: Union[str, Callable] = "euclidean",
    normalize: str = "none",
    solver: str = "closed_form",
) -> Dict[str, Union[float, Dict, Dict]]:
    """
    Compute the Wasserstein distance between two samples.

    Parameters
    ----------
    sample1 : np.ndarray
        First sample of shape (n_samples, n_features).
    sample2 : np.ndarray
        Second sample of shape (n_samples, n_features).
    distance_metric : Union[str, Callable], optional
        Distance metric to use. Can be "euclidean", "manhattan", "cosine", or a custom callable.
    normalize : str, optional
        Normalization method. Can be "none", "standard", "minmax", or "robust".
    solver : str, optional
        Solver to use. Currently only "closed_form" is supported.

    Returns
    -------
    Dict[str, Union[float, Dict, Dict]]
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_inputs(sample1, sample2, distance_metric, normalize)
    sample1_norm, sample2_norm = _normalize_samples(sample1, sample2, normalize)
    distance_matrix = _compute_distance_matrix(sample1_norm, sample2_norm, distance_metric)
    wasserstein_distance = _solve_wasserstein(distance_matrix, solver)

    return {
        "result": wasserstein_distance,
        "metrics": {},
        "params_used": {
            "distance_metric": distance_metric,
            "normalize": normalize,
            "solver": solver,
        },
        "warnings": [],
    }

################################################################################
# test_hellinger_distance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    dist1: np.ndarray,
    dist2: np.ndarray,
    normalize: str = "none",
) -> None:
    """Validate input distributions and parameters."""
    if dist1.ndim != 1 or dist2.ndim != 1:
        raise ValueError("Distributions must be 1-dimensional.")
    if dist1.shape != dist2.shape:
        raise ValueError("Distributions must have the same shape.")
    if np.any(np.isnan(dist1)) or np.any(np.isinf(dist1)):
        raise ValueError("Distribution 1 contains NaN or Inf values.")
    if np.any(np.isnan(dist2)) or np.any(np.isinf(dist2)):
        raise ValueError("Distribution 2 contains NaN or Inf values.")
    if normalize not in ["none", "standard", "minmax", "robust"]:
        raise ValueError("Invalid normalization method.")

def _normalize_distributions(
    dist1: np.ndarray,
    dist2: np.ndarray,
    normalize: str = "none",
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize distributions based on the specified method."""
    if normalize == "none":
        return dist1, dist2
    elif normalize == "standard":
        mean1, std1 = np.mean(dist1), np.std(dist1)
        mean2, std2 = np.mean(dist2), np.std(dist2)
        dist1 = (dist1 - mean1) / std1
        dist2 = (dist2 - mean2) / std2
    elif normalize == "minmax":
        min1, max1 = np.min(dist1), np.max(dist1)
        min2, max2 = np.min(dist2), np.max(dist2)
        dist1 = (dist1 - min1) / (max1 - min1 + 1e-8)
        dist2 = (dist2 - min2) / (max2 - min2 + 1e-8)
    elif normalize == "robust":
        med1, iqr1 = np.median(dist1), np.percentile(dist1, 75) - np.percentile(dist1, 25)
        med2, iqr2 = np.median(dist2), np.percentile(dist2, 75) - np.percentile(dist2, 25)
        dist1 = (dist1 - med1) / (iqr1 + 1e-8)
        dist2 = (dist2 - med2) / (iqr2 + 1e-8)
    return dist1, dist2

def _compute_hellinger_distance(
    dist1: np.ndarray,
    dist2: np.ndarray,
) -> float:
    """Compute the Hellinger distance between two distributions."""
    sqrt_dist1 = np.sqrt(dist1)
    sqrt_dist2 = np.sqrt(dist2)
    return 0.5 * np.linalg.norm(sqrt_dist1 - sqrt_dist2)

def test_hellinger_distance_fit(
    dist1: np.ndarray,
    dist2: np.ndarray,
    normalize: str = "none",
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Compute the Hellinger distance between two distributions with optional normalization and metric.

    Parameters:
    -----------
    dist1 : np.ndarray
        First distribution.
    dist2 : np.ndarray
        Second distribution.
    normalize : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    metric : Callable, optional
        Custom metric function to compute additional metrics.

    Returns:
    --------
    Dict[str, Union[float, Dict[str, float], str]]
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_inputs(dist1, dist2, normalize)
    dist1_norm, dist2_norm = _normalize_distributions(dist1, dist2, normalize)
    hellinger_distance = _compute_hellinger_distance(dist1_norm, dist2_norm)

    metrics = {"hellinger_distance": hellinger_distance}
    if metric is not None:
        try:
            custom_metric = metric(dist1_norm, dist2_norm)
            metrics["custom_metric"] = custom_metric
        except Exception as e:
            warnings.append(f"Custom metric computation failed: {str(e)}")

    return {
        "result": hellinger_distance,
        "metrics": metrics,
        "params_used": {"normalize": normalize},
        "warnings": [],
    }

# Example usage:
# dist1 = np.array([0.1, 0.2, 0.3, 0.4])
# dist2 = np.array([0.2, 0.3, 0.4, 0.1])
# result = test_hellinger_distance_fit(dist1, dist2, normalize="standard")

################################################################################
# test_jensen_shannon_divergence
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    p: np.ndarray,
    q: np.ndarray,
    base: float = 2.0
) -> None:
    """Validate inputs for Jensen-Shannon Divergence test."""
    if p.ndim != 1 or q.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if p.shape != q.shape:
        raise ValueError("Inputs must have the same shape.")
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Inputs must contain non-negative values.")
    if not np.isclose(np.sum(p), 1.0) or not np.isclose(np.sum(q), 1.0):
        raise ValueError("Inputs must sum to 1 (probability distributions).")
    if base <= 0 or base == 1:
        raise ValueError("Base must be greater than 0 and not equal to 1.")

def _kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    base: float = 2.0
) -> float:
    """Compute Kullback-Leibler divergence between two distributions."""
    epsilon = 1e-10
    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)
    return np.sum(p * np.log2(p / q)) / np.log2(base)

def _jensen_shannon_divergence(
    p: np.ndarray,
    q: np.ndarray,
    base: float = 2.0
) -> float:
    """Compute Jensen-Shannon Divergence between two distributions."""
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m, base) + 0.5 * _kl_divergence(q, m, base)

def test_jensen_shannon_divergence_fit(
    p: np.ndarray,
    q: np.ndarray,
    base: float = 2.0,
    normalization: Optional[str] = None,
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]:
    """
    Compute Jensen-Shannon Divergence between two probability distributions.

    Parameters
    ----------
    p : np.ndarray
        First probability distribution.
    q : np.ndarray
        Second probability distribution.
    base : float, optional
        Logarithm base for divergence calculation (default is 2).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : Callable, optional
        Custom metric function to compute additional metrics.

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]
        Dictionary containing:
        - "result": Jensen-Shannon Divergence value
        - "metrics": Additional metrics if provided
        - "params_used": Parameters used in the computation
        - "warnings": Any warnings generated during computation

    Example
    -------
    >>> p = np.array([0.1, 0.2, 0.7])
    >>> q = np.array([0.3, 0.4, 0.3])
    >>> result = test_jensen_shannon_divergence_fit(p, q)
    """
    _validate_inputs(p, q, base)

    # Normalization (if required)
    if normalization == 'standard':
        p = (p - np.mean(p)) / np.std(p)
        q = (q - np.mean(q)) / np.std(q)
    elif normalization == 'minmax':
        p = (p - np.min(p)) / (np.max(p) - np.min(p))
        q = (q - np.min(q)) / (np.max(q) - np.min(q))
    elif normalization == 'robust':
        p = (p - np.median(p)) / (np.percentile(p, 75) - np.percentile(p, 25))
        q = (q - np.median(q)) / (np.percentile(q, 75) - np.percentile(q, 25))

    # Compute Jensen-Shannon Divergence
    jsd = _jensen_shannon_divergence(p, q, base)

    # Compute additional metrics if provided
    metrics = {}
    if metric is not None:
        try:
            metrics['custom_metric'] = metric(p, q)
        except Exception as e:
            warnings.append(f"Custom metric computation failed: {str(e)}")

    # Prepare output
    result = {
        "result": jsd,
        "metrics": metrics,
        "params_used": {
            "base": base,
            "normalization": normalization
        },
        "warnings": []
    }

    return result
