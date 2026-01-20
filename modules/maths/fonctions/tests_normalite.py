"""
Quantix – Module tests_normalite
Généré automatiquement
Date: 2026-01-09
"""

################################################################################
# shapiro_wilk
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(data: np.ndarray) -> None:
    """Validate input data for Shapiro-Wilk test."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if len(data.shape) != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data cannot contain NaN or infinite values")
    if len(data) < 3:
        raise ValueError("Sample size must be at least 3")

def calculate_order_statistics(data: np.ndarray) -> np.ndarray:
    """Calculate order statistics for Shapiro-Wilk test."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    return (sorted_data - sorted_data.mean()) / sorted_data.std(ddof=1)

def calculate_expected_order_statistics(n: int) -> np.ndarray:
    """Calculate expected order statistics for Shapiro-Wilk test."""
    if n < 3 or n > 5000:
        raise ValueError("Sample size must be between 3 and 5000")
    # Using precomputed values for expected order statistics
    if n <= 50:
        return np.array([-0.782, -0.463, -0.195, 0.195, 0.463, 0.782])
    else:
        # For larger samples, use asymptotic approximation
        return np.sqrt(2) * scipy.stats.norm.ppf(np.linspace(0.5/n, 1-0.5/n, n))

def calculate_shapiro_wilk_statistic(data: np.ndarray) -> float:
    """Calculate Shapiro-Wilk test statistic."""
    validate_input(data)
    n = len(data)
    order_stats = calculate_order_statistics(data)
    expected_order_stats = calculate_expected_order_statistics(n)

    # Calculate correlation coefficient
    numerator = np.sum(order_stats * expected_order_stats)
    denominator = np.sqrt(np.sum(order_stats**2) * np.sum(expected_order_stats**2))
    return (numerator / denominator)**2

def shapiro_wilk_fit(
    data: np.ndarray,
    normalization: str = "standard",
    p_value_method: str = "asymptotic",
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Union[float, Dict]]:
    """
    Perform Shapiro-Wilk normality test with configurable options.

    Parameters:
    - data: Input data array
    - normalization: Normalization method ('none', 'standard', 'robust')
    - p_value_method: Method for p-value calculation ('asymptotic', 'monte_carlo')
    - custom_metric: Optional custom metric function
    - **kwargs: Additional parameters for specific methods

    Returns:
    Dictionary containing test results, metrics, and parameters used
    """
    # Validate input
    validate_input(data)

    # Apply normalization if specified
    if normalization == "standard":
        data = (data - np.mean(data)) / np.std(data, ddof=1)
    elif normalization == "robust":
        data = (data - np.median(data)) / (2 * np.median(np.abs(data - np.median(data))))

    # Calculate test statistic
    W = calculate_shapiro_wilk_statistic(data)

    # Calculate p-value based on selected method
    if p_value_method == "asymptotic":
        p_value = _calculate_asymptotic_p_value(W, len(data))
    elif p_value_method == "monte_carlo":
        p_value = _calculate_monte_carlo_p_value(data, W, **kwargs)
    else:
        raise ValueError("Invalid p_value_method specified")

    # Calculate additional metrics if custom metric is provided
    metrics = {}
    if custom_metric is not None:
        metrics["custom"] = custom_metric(data)

    return {
        "result": {
            "statistic": W,
            "p_value": p_value
        },
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "p_value_method": p_value_method
        },
        "warnings": []
    }

def _calculate_asymptotic_p_value(W: float, n: int) -> float:
    """Calculate asymptotic p-value for Shapiro-Wilk test."""
    # Using approximation from Shapiro and Wilk (1965)
    z = np.sqrt(-np.log(W))
    return 1 - scipy.stats.norm.cdf(z)

def _calculate_monte_carlo_p_value(data: np.ndarray, W_obs: float, n_simulations: int = 1000) -> float:
    """Calculate p-value using Monte Carlo simulation."""
    n = len(data)
    simulations = np.random.normal(loc=np.mean(data), scale=np.std(data, ddof=1), size=(n_simulations, n))
    W_sim = np.array([calculate_shapiro_wilk_statistic(sim) for sim in simulations])
    return (np.sum(W_sim <= W_obs) + 1) / (n_simulations + 1)

# Example usage:
"""
data = np.random.normal(size=50)
results = shapiro_wilk_fit(data, normalization="standard", p_value_method="asymptotic")
print(results)
"""

################################################################################
# kolmogorov_smirnov
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data for Kolmogorov-Smirnov test."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values")

def _compute_ks_statistic(data: np.ndarray, cdf: Callable[[np.ndarray], float]) -> float:
    """Compute the Kolmogorov-Smirnov statistic."""
    n = len(data)
    sorted_data = np.sort(data)
    cdf_values = cdf(sorted_data)

    en = (np.arange(1, n + 1) - 0.5) / n
    d_plus = np.max(cdf_values - en)
    d_minus = np.max(en - cdf_values)
    return max(d_plus, d_minus)

def _compute_p_value(ks_statistic: float, n: int) -> float:
    """Compute the p-value for the Kolmogorov-Smirnov statistic."""
    # Approximation of the p-value using the asymptotic formula
    en = np.sqrt(n) * ks_statistic + 0.12 + 0.11 / np.sqrt(n)
    return 2 * np.sum(np.exp(-2 * en**2 - (1 / en) ** 3))

def kolmogorov_smirnov_fit(
    data: np.ndarray,
    cdf: Callable[[np.ndarray], float],
    normalization: str = "none",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform the Kolmogorov-Smirnov test for normality.

    Parameters:
    -----------
    data : np.ndarray
        Input data to test.
    cdf : Callable[[np.ndarray], float]
        Cumulative distribution function to compare against.
    normalization : str, optional
        Type of normalization to apply (default: "none").
    custom_metric : Callable, optional
        Custom metric function to use (default: None).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    _validate_inputs(data)

    # Apply normalization if specified
    if normalization == "standard":
        data = (data - np.mean(data)) / np.std(data)
    elif normalization == "minmax":
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
    elif normalization == "robust":
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        data = (data - median) / iqr

    # Compute KS statistic
    ks_statistic = _compute_ks_statistic(data, cdf)
    p_value = _compute_p_value(ks_statistic, len(data))

    # Compute custom metric if provided
    metrics = {}
    if custom_metric is not None:
        metrics["custom_metric"] = custom_metric(data, cdf(data))

    result = {
        "ks_statistic": ks_statistic,
        "p_value": p_value
    }

    return {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "custom_metric": custom_metric is not None
        },
        "warnings": []
    }

# Example usage:
# data = np.random.normal(0, 1, 100)
# cdf = lambda x: (1 + np.math.erf(x / np.sqrt(2))) / 2
# result = kolmogorov_smirnov_fit(data, cdf)

################################################################################
# anderson_darling
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for Anderson-Darling test."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values")

def _compute_statistic(data: np.ndarray, normalization: str = 'standard') -> float:
    """Compute Anderson-Darling statistic."""
    n = len(data)
    data_sorted = np.sort(data)

    if normalization == 'standard':
        mean = np.mean(data)
        std = np.std(data)
        normalized_data = (data_sorted - mean) / std
    elif normalization == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        normalized_data = (data_sorted - median) / iqr
    else:
        normalized_data = data_sorted

    # Calculate empirical CDF
    i = np.arange(1, n + 1)
    emp_cdf = i / n

    # Calculate theoretical CDF for normal distribution
    from scipy.stats import norm
    theo_cdf = norm.cdf(normalized_data)

    # Compute Anderson-Darling statistic
    statistic = -n - (1 / n) * np.sum((2 * i - 1) / n * (np.log(theo_cdf) + np.log(1 - theo_cdf)))

    return statistic

def _compute_p_value(statistic: float) -> float:
    """Compute p-value for Anderson-Darling statistic."""
    # Critical values and slopes for normal distribution
    critical_values = {
        0.15: (0.532, 0.748),
        0.10: (0.632, 0.948),
        0.05: (0.732, 1.248),
        0.025: (0.832, 1.448),
        0.01: (0.932, 1.748)
    }

    p_value = 1.0
    for alpha, (a, b) in critical_values.items():
        if statistic > a:
            p_value = alpha
        else:
            break

    return p_value

def anderson_darling_fit(
    data: np.ndarray,
    normalization: str = 'standard',
    custom_metric: Optional[Callable] = None
) -> Dict[str, Union[float, Dict]]:
    """
    Perform Anderson-Darling test for normality.

    Parameters
    ----------
    data : np.ndarray
        Input data to test for normality.
    normalization : str, optional
        Normalization method ('standard' or 'robust'), by default 'standard'.
    custom_metric : Callable, optional
        Custom metric function, by default None.

    Returns
    -------
    Dict[str, Union[float, Dict]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.random.normal(0, 1, 100)
    >>> result = anderson_darling_fit(data)
    """
    _validate_input(data)

    statistic = _compute_statistic(data, normalization)
    p_value = _compute_p_value(statistic)

    metrics = {}
    if custom_metric is not None:
        metrics['custom'] = custom_metric(data)

    result = {
        'statistic': statistic,
        'p_value': p_value
    }

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization
        },
        'warnings': []
    }

################################################################################
# jarque_bera
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
    n = len(skewness)  # Assuming skewness and kurtosis are computed from same data
    jb_stat = (n / 6) * (skewness ** 2 + ((kurtosis - 3) ** 2) / 4)
    return jb_stat

def _jarque_bera_pvalue(jb_stat: float, n: int) -> float:
    """Compute p-value for Jarque-Bera test statistic."""
    from scipy.stats import chi2
    return 1 - chi2.cdf(jb_stat, df=2)

def jarque_bera_fit(
    data: np.ndarray,
    normalize: bool = True,
    custom_skewness: Optional[Callable[[np.ndarray], float]] = None,
    custom_kurtosis: Optional[Callable[[np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Perform Jarque-Bera test for normality.

    Parameters
    ----------
    data : np.ndarray
        Input data to test for normality.
    normalize : bool, optional
        Whether to standardize the data before computation (default True).
    custom_skewness : Callable, optional
        Custom skewness function. Must take a numpy array and return a float.
    custom_kurtosis : Callable, optional
        Custom kurtosis function. Must take a numpy array and return a float.

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], str]]
        Dictionary containing:
        - "result": dict with test statistic and p-value
        - "metrics": dict with skewness and kurtosis values
        - "params_used": dict of parameters used in computation
        - "warnings": str with any warnings

    Examples
    --------
    >>> data = np.random.normal(0, 1, 100)
    >>> result = jarque_bera_fit(data)
    """
    _validate_input(data)

    warnings = []
    params_used = {
        "normalize": normalize,
        "custom_skewness": custom_skewness is not None,
        "custom_kurtosis": custom_kurtosis is not None
    }

    if normalize:
        data = (data - np.mean(data)) / np.std(data)

    # Compute skewness and kurtosis
    if custom_skewness is not None:
        skewness = custom_skewness(data)
    else:
        skewness = _compute_skewness(data)

    if custom_kurtosis is not None:
        kurtosis = custom_kurtosis(data)
    else:
        kurtosis = _compute_kurtosis(data)

    # Compute test statistic and p-value
    n = len(data)
    jb_stat = _jarque_bera_statistic(skewness, kurtosis)
    p_value = _jarque_bera_pvalue(jb_stat, n)

    metrics = {
        "skewness": skewness,
        "kurtosis": kurtosis
    }

    result = {
        "statistic": jb_stat,
        "p_value": p_value
    }

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": "\n".join(warnings) if warnings else None
    }

################################################################################
# lilliefors
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for Lilliefors test."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if len(data.shape) != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data must not contain NaN or infinite values")
    if len(data) < 3:
        raise ValueError("Input data must contain at least 3 samples")

def _compute_empirical_cdf(data: np.ndarray) -> tuple:
    """Compute empirical CDF and sorted data."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    y = np.arange(1, n + 1) / n
    return sorted_data, y

def _compute_lilliefors_statistic(sorted_data: np.ndarray,
                                 empirical_cdf: np.ndarray,
                                 mean: float = 0.0,
                                 std: float = 1.0) -> float:
    """Compute Lilliefors test statistic."""
    n = len(sorted_data)
    theoretical_cdf = 0.5 * (1 + np.erf((sorted_data - mean) / (std * np.sqrt(2))))
    differences = empirical_cdf - theoretical_cdf
    return np.max(np.abs(differences))

def _estimate_parameters(data: np.ndarray,
                        normalization: str = 'standard') -> Dict[str, float]:
    """Estimate parameters for normal distribution."""
    if normalization == 'standard':
        mean = np.mean(data)
        std = np.std(data, ddof=1)
    elif normalization == 'robust':
        mean = np.median(data)
        std = np.median(np.abs(data - mean)) * 1.4826
    else:
        mean = 0.0
        std = 1.0
    return {'mean': mean, 'std': std}

def lilliefors_fit(data: np.ndarray,
                   normalization: str = 'standard',
                   custom_normalization: Optional[Callable] = None) -> Dict:
    """
    Perform Lilliefors test for normality.

    Parameters
    ----------
    data : np.ndarray
        Input data to test for normality.
    normalization : str, optional
        Normalization method ('none', 'standard', 'robust').
    custom_normalization : callable, optional
        Custom normalization function.

    Returns
    -------
    dict
        Dictionary containing test results, metrics, parameters used and warnings.
    """
    # Validate input data
    _validate_input(data)

    # Estimate parameters based on normalization choice
    if custom_normalization is not None:
        params = custom_normalization(data)
    else:
        params = _estimate_parameters(data, normalization)

    # Compute empirical CDF
    sorted_data, empirical_cdf = _compute_empirical_cdf(data)

    # Compute Lilliefors statistic
    statistic = _compute_lilliefors_statistic(sorted_data, empirical_cdf,
                                             params['mean'], params['std'])

    # Get p-value (approximate)
    n = len(data)
    if n <= 20:
        p_value = np.nan
    else:
        # Approximate p-value using formula from Lilliefors (1967)
        z = statistic * np.sqrt(n) + 0.155 + 0.24 / np.sqrt(n)
        p_value = 1 - np.exp(-(z ** 2) / 2)

    # Prepare results
    result = {
        'statistic': statistic,
        'p_value': p_value,
        'normalization_used': normalization if custom_normalization is None else 'custom'
    }

    metrics = {
        'statistic': statistic,
        'p_value': p_value
    }

    params_used = {
        'mean': params['mean'],
        'std': params['std']
    }

    warnings = []
    if n < 20:
        warnings.append("Sample size is less than 20, p-value approximation may be unreliable")

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

# Example usage:
# data = np.random.normal(0, 1, 100)
# result = lilliefors_fit(data, normalization='standard')

################################################################################
# chi2
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data for chi2 test."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def _calculate_chi2_statistic(data: np.ndarray, bins: int = 10) -> float:
    """Calculate chi2 statistic for normality test."""
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    expected_prob = 1 / len(bin_edges)
    chi2_stat = np.sum((hist - expected_prob) ** 2 / expected_prob)
    return chi2_stat

def _estimate_parameters(data: np.ndarray) -> Dict[str, float]:
    """Estimate parameters for chi2 test."""
    return {
        'mean': np.mean(data),
        'std': np.std(data, ddof=1)
    }

def _calculate_metrics(data: np.ndarray,
                      metrics: Union[str, Callable],
                      params: Dict[str, float]) -> Dict[str, float]:
    """Calculate metrics for chi2 test."""
    if isinstance(metrics, str):
        if metrics == 'mse':
            return {'mse': np.mean((data - params['mean']) ** 2)}
        elif metrics == 'mae':
            return {'mae': np.mean(np.abs(data - params['mean']))}
        elif metrics == 'r2':
            ss_total = np.sum((data - np.mean(data)) ** 2)
            ss_res = np.sum((data - params['mean']) ** 2)
            return {'r2': 1 - (ss_res / ss_total)}
        else:
            raise ValueError(f"Unknown metric: {metrics}")
    elif callable(metrics):
        return {'custom_metric': metrics(data, params)}
    else:
        raise TypeError("Metrics must be a string or callable")

def chi2_fit(data: np.ndarray,
             bins: int = 10,
             metrics: Union[str, Callable] = 'mse',
             custom_metric: Optional[Callable] = None) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Perform chi2 goodness-of-fit test for normality.

    Parameters:
    -----------
    data : np.ndarray
        Input data to test for normality.
    bins : int, optional
        Number of bins for histogram (default: 10).
    metrics : str or callable, optional
        Metric to calculate ('mse', 'mae', 'r2') or custom callable (default: 'mse').
    custom_metric : callable, optional
        Custom metric function if metrics is 'custom'.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.

    Example:
    --------
    >>> data = np.random.normal(0, 1, 100)
    >>> result = chi2_fit(data)
    """
    _validate_inputs(data)

    params = _estimate_parameters(data)
    chi2_stat = _calculate_chi2_statistic(data, bins)

    if custom_metric is not None:
        metrics = custom_metric

    metrics_result = _calculate_metrics(data, metrics, params)

    return {
        'result': {
            'chi2_statistic': chi2_stat,
            'p_value': 0.5,  # Placeholder for actual p-value calculation
        },
        'metrics': metrics_result,
        'params_used': params,
        'warnings': []
    }

################################################################################
# q_q_plot
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data for Q-Q plot."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values")

def _compute_quantiles(data: np.ndarray) -> tuple:
    """Compute theoretical and sample quantiles for Q-Q plot."""
    n = len(data)
    sample_quantiles = np.sort(data)
    theoretical_quantiles = np.linspace(0.5/n, 1-0.5/n, n)
    return theoretical_quantiles, sample_quantiles

def _normalize_data(data: np.ndarray,
                   normalization: str = 'standard',
                   custom_normalization: Optional[Callable] = None) -> np.ndarray:
    """Normalize data according to specified method."""
    if custom_normalization is not None:
        return custom_normalization(data)

    if normalization == 'none':
        return data
    elif normalization == 'standard':
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std
    elif normalization == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)
    elif normalization == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _compute_metrics(theoretical: np.ndarray,
                    sample: np.ndarray,
                    metric: str = 'mse',
                    custom_metric: Optional[Callable] = None) -> Dict[str, float]:
    """Compute metrics between theoretical and sample quantiles."""
    if custom_metric is not None:
        return {'custom': custom_metric(theoretical, sample)}

    metrics = {}
    if metric == 'mse' or 'all' in metric:
        mse = np.mean((theoretical - sample) ** 2)
        metrics['mse'] = mse
    if metric == 'mae' or 'all' in metric:
        mae = np.mean(np.abs(theoretical - sample))
        metrics['mae'] = mae
    if metric == 'r2' or 'all' in metric:
        ss_res = np.sum((theoretical - sample) ** 2)
        ss_tot = np.sum((sample - np.mean(sample)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        metrics['r2'] = r2

    return metrics

def q_q_plot_fit(data: np.ndarray,
                normalization: str = 'standard',
                metric: Union[str, Callable] = 'mse',
                custom_normalization: Optional[Callable] = None,
                custom_metric: Optional[Callable] = None) -> Dict:
    """
    Compute Q-Q plot statistics and metrics.

    Parameters
    ----------
    data : np.ndarray
        Input data to test for normality.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to compute ('mse', 'mae', 'r2', 'all' or custom callable).
    custom_normalization : callable, optional
        Custom normalization function.
    custom_metric : callable, optional
        Custom metric function.

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': tuple of (theoretical_quantiles, sample_quantiles)
        - 'metrics': dictionary of computed metrics
        - 'params_used': dictionary of parameters used
        - 'warnings': list of warnings

    Example
    -------
    >>> data = np.random.normal(0, 1, 100)
    >>> result = q_q_plot_fit(data)
    """
    # Validate inputs
    _validate_inputs(data)

    # Normalize data
    normalized_data = _normalize_data(
        data,
        normalization=normalization,
        custom_normalization=custom_normalization
    )

    # Compute quantiles
    theoretical, sample = _compute_quantiles(normalized_data)

    # Compute metrics
    if isinstance(metric, str):
        computed_metrics = _compute_metrics(
            theoretical,
            sample,
            metric=metric
        )
    else:
        computed_metrics = _compute_metrics(
            theoretical,
            sample,
            custom_metric=metric
        )

    # Prepare output
    result = {
        'result': (theoretical, sample),
        'metrics': computed_metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric if isinstance(metric, str) else 'custom',
        },
        'warnings': []
    }

    return result
