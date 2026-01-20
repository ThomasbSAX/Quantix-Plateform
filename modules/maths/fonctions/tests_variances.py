"""
Quantix – Module tests_variances
Généré automatiquement
Date: 2026-01-09
"""

################################################################################
# test_bartlett
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(*samples: np.ndarray) -> None:
    """Validate input samples for Bartlett's test."""
    if len(samples) < 2:
        raise ValueError("At least two samples are required for Bartlett's test.")
    for sample in samples:
        if not isinstance(sample, np.ndarray):
            raise TypeError("All inputs must be numpy arrays.")
        if sample.ndim != 1:
            raise ValueError("All inputs must be 1-dimensional arrays.")
        if np.isnan(sample).any() or np.isinf(sample).any():
            raise ValueError("Input arrays must not contain NaN or infinite values.")

def _calculate_variances(*samples: np.ndarray) -> np.ndarray:
    """Calculate sample variances."""
    return np.array([np.var(sample, ddof=1) for sample in samples])

def _calculate_means(*samples: np.ndarray) -> np.ndarray:
    """Calculate sample means."""
    return np.array([np.mean(sample) for sample in samples])

def _calculate_statistic(variances: np.ndarray, means: np.ndarray,
                        n_samples: np.ndarray) -> float:
    """Calculate Bartlett's test statistic."""
    k = len(variances)
    c = np.log(np.sum(n_samples) - k)
    numerator = (k * np.log(variances.mean())) - np.sum(np.log(variances))
    denominator = 1 + (1 / (3 * (k - 1))) * (
        np.sum(1 / n_samples) - 1 / np.sum(n_samples)
    )
    return (n_samples.sum() - k) * numerator / denominator

def _calculate_p_value(statistic: float, df: int) -> float:
    """Calculate p-value for Bartlett's test using chi-square distribution."""
    from scipy.stats import chi2
    return 1 - chi2.cdf(statistic, df)

def test_bartlett_fit(
    *samples: np.ndarray,
    normalization: Optional[str] = None,
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform Bartlett's test for homogeneity of variances.

    Parameters:
    -----------
    *samples : np.ndarray
        Input samples to test.
    normalization : str, optional
        Type of normalization ('none', 'standard', 'minmax', 'robust').
    custom_normalization : callable, optional
        Custom normalization function.
    **kwargs :
        Additional keyword arguments for future extensions.

    Returns:
    --------
    dict
        Dictionary containing test results, metrics, parameters used, and warnings.
    """
    _validate_inputs(*samples)

    # Normalization
    normalized_samples = []
    for sample in samples:
        if normalization == 'standard':
            sample = (sample - np.mean(sample)) / np.std(sample)
        elif normalization == 'minmax':
            sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
        elif normalization == 'robust':
            sample = (sample - np.median(sample)) / (np.percentile(sample, 75) -
                                                    np.percentile(sample, 25))
        elif custom_normalization is not None:
            sample = custom_normalization(sample)
        normalized_samples.append(sample)

    n_samples = np.array([len(sample) for sample in normalized_samples])
    variances = _calculate_variances(*normalized_samples)
    means = _calculate_means(*normalized_samples)

    # Calculate test statistic
    df = len(samples) - 1
    statistic = _calculate_statistic(variances, means, n_samples)
    p_value = _calculate_p_value(statistic, df)

    # Prepare results
    result = {
        'statistic': statistic,
        'p_value': p_value,
        'variances': variances,
        'means': means
    }

    metrics = {
        'statistic_type': 'Bartlett',
        'degrees_of_freedom': df
    }

    params_used = {
        'normalization': normalization,
        'custom_normalization': custom_normalization is not None
    }

    warnings = []
    if any(n < 5 for n in n_samples):
        warnings.append("Some samples have fewer than 5 observations. Results may be unreliable.")

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

# Example usage:
# result = test_bartlett_fit(np.random.normal(0, 1, 100), np.random.normal(1, 2, 100))

################################################################################
# test_levene
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(*groups: np.ndarray) -> None:
    """Validate input groups for Levene's test."""
    if len(groups) < 2:
        raise ValueError("At least two groups are required for Levene's test.")
    for group in groups:
        if not isinstance(group, np.ndarray):
            raise TypeError("All inputs must be numpy arrays.")
        if group.ndim != 1:
            raise ValueError("Each input must be a 1-dimensional array.")
        if np.any(np.isnan(group)) or np.any(np.isinf(group)):
            raise ValueError("Input arrays must not contain NaN or infinite values.")

def _center_data(
    group: np.ndarray,
    center_func: Callable[[np.ndarray], float] = np.median
) -> np.ndarray:
    """Center the data using the specified center function."""
    return group - center_func(group)

def _compute_absolute_deviations(
    *groups: np.ndarray,
    center_func: Callable[[np.ndarray], float] = np.median
) -> np.ndarray:
    """Compute absolute deviations from the center for each group."""
    centered_groups = [_center_data(group, center_func) for group in groups]
    return np.concatenate([np.abs(centered_group) for centered_group in centered_groups])

def _compute_statistic(
    *groups: np.ndarray,
    center_func: Callable[[np.ndarray], float] = np.median
) -> float:
    """Compute the Levene's test statistic."""
    absolute_deviations = _compute_absolute_deviations(*groups, center_func=center_func)
    n_total = sum(len(group) for group in groups)
    grand_mean = np.mean(absolute_deviations)

    between_group_ss = 0.0
    for group in groups:
        n_i = len(group)
        mean_i = np.mean(_center_data(group, center_func))
        between_group_ss += n_i * (mean_i - grand_mean) ** 2

    within_group_ss = np.sum((absolute_deviations - grand_mean) ** 2)

    df_between = len(groups) - 1
    df_within = n_total - len(groups)

    statistic = (between_group_ss / df_between) / (within_group_ss / df_within)
    return statistic

def _compute_p_value(statistic: float, df_between: int, df_within: int) -> float:
    """Compute the p-value for Levene's test using an F-distribution approximation."""
    from scipy.stats import f
    return 1 - f.cdf(statistic, df_between, df_within)

def test_levene_fit(
    *groups: np.ndarray,
    center_func: Callable[[np.ndarray], float] = np.median,
    alpha: float = 0.05
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Perform Levene's test for equality of variances.

    Parameters
    ----------
    *groups : np.ndarray
        Input groups to test.
    center_func : Callable[[np.ndarray], float], optional
        Function used to compute the center of each group (default is np.median).
    alpha : float, optional
        Significance level for the test (default is 0.05).

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], str]]
        Dictionary containing the test results, metrics, parameters used, and warnings.

    Example
    -------
    >>> group1 = np.random.normal(0, 1, 100)
    >>> group2 = np.random.normal(0, 2, 100)
    >>> result = test_levene_fit(group1, group2)
    """
    _validate_inputs(*groups)

    statistic = _compute_statistic(*groups, center_func=center_func)
    n_total = sum(len(group) for group in groups)
    df_between = len(groups) - 1
    df_within = n_total - len(groups)

    p_value = _compute_p_value(statistic, df_between, df_within)
    reject_null = p_value < alpha

    result = {
        "statistic": statistic,
        "p_value": p_value,
        "reject_null": reject_null
    }

    metrics = {
        "df_between": df_between,
        "df_within": df_within
    }

    params_used = {
        "center_func": center_func.__name__,
        "alpha": alpha
    }

    warnings = []
    if reject_null:
        warnings.append("Null hypothesis of equal variances is rejected.")

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

################################################################################
# test_fligner_killeen
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(*samples: np.ndarray) -> None:
    """Validate input samples for Fligner-Killeen test."""
    if len(samples) < 2:
        raise ValueError("At least two samples are required.")
    for sample in samples:
        if not isinstance(sample, np.ndarray):
            raise TypeError("All inputs must be numpy arrays.")
        if sample.ndim != 1:
            raise ValueError("Each input must be a 1D array.")
        if np.any(np.isnan(sample)) or np.any(np.isinf(sample)):
            raise ValueError("Input arrays must not contain NaN or Inf values.")

def _compute_ranks(*samples: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute ranks for Fligner-Killeen test."""
    combined = np.concatenate(samples)
    ranks = np.argsort(np.argsort(combined))
    return {f'sample_{i}': ranks[:len(sample)] for i, sample in enumerate(samples)}

def _compute_statistic(ranks: Dict[str, np.ndarray], weights: Optional[np.ndarray] = None) -> float:
    """Compute Fligner-Killeen test statistic."""
    if weights is None:
        weights = np.ones(len(ranks))
    n_samples = len(ranks)
    statistic = 0.0
    for i, (_, rank) in enumerate(ranks.items()):
        statistic += weights[i] * np.sum((rank - 0.5) ** 2)
    statistic /= n_samples
    return statistic

def _compute_p_value(statistic: float, *samples: np.ndarray) -> float:
    """Compute p-value for Fligner-Killeen test using permutation approach."""
    combined = np.concatenate(samples)
    n_permutations = 1000
    perm_statistics = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_samples = np.array_split(combined, len(samples))
        perm_ranks = _compute_ranks(*perm_samples)
        perm_statistics.append(_compute_statistic(perm_ranks))
    p_value = np.mean(np.array(perm_statistics) >= statistic)
    return p_value

def test_fligner_killeen_fit(
    *samples: np.ndarray,
    normalization: str = 'none',
    weights: Optional[np.ndarray] = None,
    n_permutations: int = 1000
) -> Dict[str, Union[float, Dict[str, np.ndarray], str]]:
    """
    Perform Fligner-Killeen test for homogeneity of variances.

    Parameters
    ----------
    *samples : np.ndarray
        Input samples to test.
    normalization : str, optional
        Type of normalization ('none', 'standard', 'minmax', 'robust').
    weights : np.ndarray, optional
        Weights for each sample.
    n_permutations : int, optional
        Number of permutations for p-value computation.

    Returns
    -------
    Dict[str, Union[float, Dict[str, np.ndarray], str]]
        Dictionary containing test results, metrics, and parameters used.

    Example
    -------
    >>> sample1 = np.random.normal(0, 1, 100)
    >>> sample2 = np.random.normal(0, 2, 100)
    >>> result = test_fligner_killeen_fit(sample1, sample2)
    """
    _validate_inputs(*samples)

    if normalization != 'none':
        raise NotImplementedError("Normalization options other than 'none' are not implemented yet.")

    if weights is not None and len(weights) != len(samples):
        raise ValueError("Weights must have the same length as the number of samples.")

    ranks = _compute_ranks(*samples)
    statistic = _compute_statistic(ranks, weights)
    p_value = _compute_p_value(statistic, *samples)

    return {
        'result': {
            'statistic': statistic,
            'p_value': p_value
        },
        'metrics': {},
        'params_used': {
            'normalization': normalization,
            'weights': weights,
            'n_permutations': n_permutations
        },
        'warnings': []
    }

################################################################################
# test_brown_forsythe
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    groups: np.ndarray,
    values: np.ndarray,
    normalize: str = "standard",
    metric: Union[str, Callable] = "mse"
) -> None:
    """
    Validate input data and parameters for Brown-Forsythe test.

    Parameters
    ----------
    groups : np.ndarray
        Array of group labels.
    values : np.ndarray
        Array of values to test.
    normalize : str, optional
        Normalization method (none, standard, minmax, robust).
    metric : str or callable, optional
        Metric to use (mse, mae, custom callable).

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if groups.shape != values.shape:
        raise ValueError("groups and values must have the same shape")
    if np.any(np.isnan(groups)) or np.any(np.isinf(groups)):
        raise ValueError("groups contains NaN or inf values")
    if np.any(np.isnan(values)) or np.any(np.isinf(values)):
        raise ValueError("values contains NaN or inf values")
    if normalize not in ["none", "standard", "minmax", "robust"]:
        raise ValueError("normalize must be one of: none, standard, minmax, robust")
    if isinstance(metric, str) and metric not in ["mse", "mae"]:
        raise ValueError("metric must be one of: mse, mae or a callable")

def _normalize_data(
    values: np.ndarray,
    normalize: str = "standard"
) -> np.ndarray:
    """
    Normalize data according to specified method.

    Parameters
    ----------
    values : np.ndarray
        Array of values to normalize.
    normalize : str, optional
        Normalization method (none, standard, minmax, robust).

    Returns
    ------
    np.ndarray
        Normalized values.
    """
    if normalize == "none":
        return values
    elif normalize == "standard":
        mean = np.mean(values)
        std = np.std(values)
        return (values - mean) / std
    elif normalize == "minmax":
        min_val = np.min(values)
        max_val = np.max(values)
        return (values - min_val) / (max_val - min_val)
    elif normalize == "robust":
        median = np.median(values)
        iqr = np.percentile(values, 75) - np.percentile(values, 25)
        return (values - median) / iqr
    else:
        raise ValueError("Invalid normalization method")

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable] = "mse"
) -> float:
    """
    Compute specified metric between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    metric : str or callable, optional
        Metric to use (mse, mae, custom callable).

    Returns
    ------
    float
        Computed metric value.
    """
    if isinstance(metric, str):
        if metric == "mse":
            return np.mean((y_true - y_pred) ** 2)
        elif metric == "mae":
            return np.mean(np.abs(y_true - y_pred))
    else:
        return metric(y_true, y_pred)

def _brown_forsythe_statistic(
    groups: np.ndarray,
    values: np.ndarray,
    metric: Union[str, Callable] = "mse"
) -> float:
    """
    Compute Brown-Forsythe test statistic.

    Parameters
    ----------
    groups : np.ndarray
        Array of group labels.
    values : np.ndarray
        Array of values to test.
    metric : str or callable, optional
        Metric to use (mse, mae, custom callable).

    Returns
    ------
    float
        Brown-Forsythe test statistic.
    """
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    group_sizes = [np.sum(groups == g) for g in unique_groups]

    # Calculate group medians
    group_medians = np.array([np.median(values[groups == g]) for g in unique_groups])

    # Calculate absolute deviations from group medians
    abs_devs = np.abs(values - group_medians[groups])

    # Calculate metric for each group
    group_metrics = np.array([
        _compute_metric(abs_devs[groups == g], np.zeros_like(abs_devs[groups == g]), metric)
        for g in unique_groups
    ])

    # Calculate overall weighted mean of group metrics
    overall_mean = np.sum(group_metrics * group_sizes) / np.sum(group_sizes)

    # Calculate Brown-Forsythe statistic
    statistic = (np.sum((group_metrics - overall_mean) ** 2) /
                 (n_groups - 1)) / np.mean(group_sizes)

    return statistic

def test_brown_forsythe_fit(
    groups: np.ndarray,
    values: np.ndarray,
    normalize: str = "standard",
    metric: Union[str, Callable] = "mse"
) -> Dict:
    """
    Perform Brown-Forsythe test for equality of variances.

    Parameters
    ----------
    groups : np.ndarray
        Array of group labels.
    values : np.ndarray
        Array of values to test.
    normalize : str, optional
        Normalization method (none, standard, minmax, robust).
    metric : str or callable, optional
        Metric to use (mse, mae, custom callable).

    Returns
    ------
    dict
        Dictionary containing test results, metrics, parameters used and warnings.
    """
    # Validate inputs
    _validate_inputs(groups, values, normalize, metric)

    # Normalize data if specified
    normalized_values = _normalize_data(values, normalize)

    # Compute test statistic
    statistic = _brown_forsythe_statistic(groups, normalized_values, metric)

    # In a real implementation, you would compute p-value here
    # For this example, we'll just return the statistic
    p_value = np.nan  # Placeholder

    return {
        "result": {
            "statistic": statistic,
            "p_value": p_value
        },
        "metrics": {
            "normalization_used": normalize,
            "metric_used": metric
        },
        "params_used": {
            "groups_shape": groups.shape,
            "values_shape": values.shape
        },
        "warnings": []
    }

# Example usage:
"""
groups = np.array([1, 1, 2, 2, 3, 3])
values = np.array([5.1, 4.9, 6.2, 6.0, 7.3, 8.1])
result = test_brown_forsythe_fit(groups, values)
print(result)
"""

################################################################################
# test_ansari_bradley
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for Ansari-Bradley test."""
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional")
    if len(x) != len(y):
        raise ValueError("Inputs must have the same length")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("Inputs must not contain NaN values")
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        raise ValueError("Inputs must not contain infinite values")

def _compute_ranks(data: np.ndarray) -> tuple:
    """Compute ranks for the combined data."""
    combined = np.concatenate([data[0], data[1]])
    ranks = np.argsort(np.argsort(combined))
    return ranks[:len(data[0])], ranks[len(data[0]):]

def _ansari_bradley_statistic(ranks_x: np.ndarray, ranks_y: np.ndarray) -> float:
    """Compute the Ansari-Bradley test statistic."""
    n_x = len(ranks_x)
    n_y = len(ranks_y)
    total_n = n_x + n_y
    statistic = (n_x * n_y / total_n) * (
        np.sum((ranks_x - 0.5 * (n_x + 1))**2) / n_x +
        np.sum((ranks_y - 0.5 * (n_y + 1))**2) / n_y
    )
    return statistic

def _compute_p_value(statistic: float, n_x: int, n_y: int) -> float:
    """Compute the p-value for the Ansari-Bradley test."""
    # Approximation using normal distribution
    mean = (n_x + n_y + 1) / (2 * (n_x + n_y))
    variance = (n_x**3 - n_x) / (12 * (n_x + n_y)**2) + (
        n_y**3 - n_y
    ) / (12 * (n_x + n_y)**2)
    z_score = (statistic - mean) / np.sqrt(variance)
    p_value = 2 * min(1, 1 - norm.cdf(z_score))
    return p_value

def test_ansari_bradley_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    normalization: Optional[str] = None,
    metric: str = "mse",
    solver: str = "closed_form",
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform the Ansari-Bradley test for variance equality.

    Parameters:
    -----------
    x : np.ndarray
        First sample of data.
    y : np.ndarray
        Second sample of data.
    normalization : str, optional
        Normalization method (none, standard, minmax, robust).
    metric : str
        Metric to use for evaluation (mse, mae, r2).
    solver : str
        Solver method (closed_form).
    custom_metric : Callable, optional
        Custom metric function.
    **kwargs :
        Additional keyword arguments for the solver.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.

    Example:
    --------
    >>> x = np.random.normal(0, 1, 100)
    >>> y = np.random.normal(0, 2, 100)
    >>> result = test_ansari_bradley_fit(x, y)
    """
    _validate_inputs(x, y)

    # Normalization
    if normalization == "standard":
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
    elif normalization == "minmax":
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif normalization == "robust":
        x = (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25))
        y = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))

    # Prepare data
    data = (x, y)
    ranks_x, ranks_y = _compute_ranks(data)

    # Compute statistic
    statistic = _ansari_bradley_statistic(ranks_x, ranks_y)
    n_x = len(x)
    n_y = len(y)

    # Compute p-value
    from scipy.stats import norm
    p_value = _compute_p_value(statistic, n_x, n_y)

    # Metrics
    metrics = {}
    if metric == "mse":
        metrics["mse"] = np.mean((x - y)**2)
    elif metric == "mae":
        metrics["mae"] = np.mean(np.abs(x - y))
    elif metric == "r2":
        ss_total = np.sum((x - np.mean(x))**2)
        ss_res = np.sum((x - y)**2)
        metrics["r2"] = 1 - (ss_res / ss_total)

    if custom_metric is not None:
        metrics["custom"] = custom_metric(x, y)

    # Result
    result = {
        "statistic": statistic,
        "p_value": p_value,
        "hypothesis": "H0: variances are equal",
        "alternative": "H1: variances are not equal"
    }

    # Warnings
    warnings = []
    if p_value < 0.05:
        warnings.append("Significant difference in variances detected")

    return {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric,
            "solver": solver
        },
        "warnings": warnings
    }

################################################################################
# test_mood
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for Mood's median test."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Input arrays must be 1-dimensional")
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def _calculate_ranks(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate ranks for Mood's median test."""
    combined = np.column_stack((x, y))
    sorted_combined = np.sort(combined, axis=0)
    ranks = np.argsort(np.argsort(sorted_combined, axis=0), axis=0)
    x_ranks = ranks[:, 0]
    y_ranks = ranks[:, 1]
    return x_ranks, y_ranks

def _calculate_statistic(x_ranks: np.ndarray, y_ranks: np.ndarray) -> float:
    """Calculate the test statistic for Mood's median test."""
    n_x = len(x_ranks)
    n_y = len(y_ranks)
    total_n = n_x + n_y
    median_rank = (total_n + 1) / 2

    x_above_median = np.sum(x_ranks > median_rank)
    y_above_median = np.sum(y_ranks > median_rank)

    statistic = (x_above_median - n_x * 0.5) / np.sqrt(n_x * (total_n + 1) * total_n / 12)
    return statistic

def _calculate_p_value(statistic: float) -> float:
    """Calculate the p-value for Mood's median test."""
    # Approximation using normal distribution
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(statistic)))
    return p_value

def test_mood_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    rank_method: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] = _calculate_ranks,
    statistic_method: Callable[[np.ndarray, np.ndarray], float] = _calculate_statistic,
    p_value_method: Callable[[float], float] = _calculate_p_value,
) -> Dict[str, Any]:
    """
    Perform Mood's median test for comparing variances of two samples.

    Parameters:
    -----------
    x : np.ndarray
        First sample array.
    y : np.ndarray
        Second sample array.
    rank_method : Callable, optional
        Function to calculate ranks (default: _calculate_ranks).
    statistic_method : Callable, optional
        Function to calculate test statistic (default: _calculate_statistic).
    p_value_method : Callable, optional
        Function to calculate p-value (default: _calculate_p_value).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing test results, metrics, and parameters used.

    Example:
    --------
    >>> x = np.random.normal(0, 1, 50)
    >>> y = np.random.normal(0, 2, 50)
    >>> result = test_mood_fit(x, y)
    """
    _validate_inputs(x, y)

    x_ranks, y_ranks = rank_method(x, y)
    statistic = statistic_method(x_ranks, y_ranks)
    p_value = p_value_method(statistic)

    result = {
        "result": {
            "statistic": statistic,
            "p_value": p_value
        },
        "metrics": {},
        "params_used": {
            "rank_method": rank_method.__name__,
            "statistic_method": statistic_method.__name__,
            "p_value_method": p_value_method.__name__
        },
        "warnings": []
    }

    return result

################################################################################
# test_puri_sen
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if x.shape != y.shape:
        raise ValueError("Inputs must have the same shape.")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("Inputs must not contain NaN values.")
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        raise ValueError("Inputs must not contain infinite values.")

def _compute_rank_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the rank statistic for Puri-Sen test."""
    combined = np.concatenate([x, y])
    ranks = np.argsort(np.argsort(combined))
    rank_x = ranks[:len(x)]
    rank_y = ranks[len(x):]
    return np.sum(rank_x) - len(x) * (len(x) + 1) / 2

def _compute_p_value(statistic: float, n_x: int, n_y: int) -> float:
    """Compute the p-value for Puri-Sen test."""
    mean = 0
    variance = (n_x + n_y) * n_x * n_y / 12
    std_dev = np.sqrt(variance)
    z_score = (statistic - mean) / std_dev
    return 2 * (1 - norm.cdf(abs(z_score)))

def test_puri_sen_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    normalization: Optional[str] = None,
    metric: str = "rank",
    solver: str = "closed_form",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Perform the Puri-Sen test for comparing variances.

    Parameters:
    -----------
    x : np.ndarray
        First sample of data.
    y : np.ndarray
        Second sample of data.
    normalization : str, optional
        Normalization method (none, standard, minmax, robust).
    metric : str, optional
        Metric to use (rank).
    solver : str, optional
        Solver method (closed_form).
    custom_metric : Callable, optional
        Custom metric function.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    _validate_inputs(x, y)

    # Normalization
    if normalization == "standard":
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
    elif normalization == "minmax":
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif normalization == "robust":
        x = (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25))
        y = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))

    # Compute statistic
    if metric == "rank" or custom_metric is None:
        statistic = _compute_rank_statistic(x, y)
    else:
        statistic = custom_metric(x, y)

    # Compute p-value
    n_x = len(x)
    n_y = len(y)
    p_value = _compute_p_value(statistic, n_x, n_y)

    # Prepare output
    result = {
        "statistic": statistic,
        "p_value": p_value
    }

    metrics = {
        "metric_used": metric if custom_metric is None else "custom",
        "statistic_value": statistic,
        "p_value": p_value
    }

    params_used = {
        "normalization": normalization,
        "metric": metric,
        "solver": solver,
        "tol": tol,
        "max_iter": max_iter
    }

    warnings = []

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

################################################################################
# test_conover
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    samples: np.ndarray,
    alpha: float = 0.05
) -> None:
    """
    Validate input data for Conover test.

    Parameters
    ----------
    samples : np.ndarray
        Array of samples to test. Shape (n_samples, n_groups)
    alpha : float
        Significance level

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if not isinstance(samples, np.ndarray):
        raise ValueError("Samples must be a numpy array")
    if samples.ndim != 2:
        raise ValueError("Samples must be 2-dimensional")
    if np.any(np.isnan(samples)):
        raise ValueError("Samples contain NaN values")
    if np.any(np.isinf(samples)):
        raise ValueError("Samples contain infinite values")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha must be between 0 and 1")

def _compute_rank_averages(
    samples: np.ndarray
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Compute rank averages for each group.

    Parameters
    ----------
    samples : np.ndarray
        Array of samples to test. Shape (n_samples, n_groups)

    Returns
    -------
    dict
        Dictionary containing:
        - 'ranks': array of ranks for each sample
        - 'group_averages': average rank for each group
    """
    n_samples, n_groups = samples.shape

    # Flatten and rank all samples
    flattened = samples.flatten()
    ranks = np.argsort(np.argsort(flattened))

    # Reshape back to original shape
    ranks = ranks.reshape(samples.shape)

    # Compute average rank for each group
    group_averages = np.mean(ranks, axis=0)

    return {
        'ranks': ranks,
        'group_averages': group_averages
    }

def _compute_statistic(
    group_averages: np.ndarray,
    n_samples_per_group: Optional[np.ndarray] = None
) -> float:
    """
    Compute Conover test statistic.

    Parameters
    ----------
    group_averages : np.ndarray
        Average ranks for each group
    n_samples_per_group : Optional[np.ndarray]
        Number of samples in each group. If None, assumes equal sample sizes.

    Returns
    -------
    float
        Computed test statistic
    """
    n_groups = len(group_averages)

    if n_samples_per_group is None:
        # Assume equal sample sizes
        n = len(group_averages)
        n_samples_per_group = np.full(n_groups, n // n_groups)

    # Compute overall mean rank
    total_n = np.sum(n_samples_per_group)
    grand_mean = (total_n + 1) / 2

    # Compute numerator
    numerator = np.sum(n_samples_per_group * (group_averages - grand_mean) ** 2)

    # Compute denominator
    denominator = (total_n * (total_n + 1) / 12) * (1 - np.sum(n_samples_per_group ** 3) / total_n**3)

    # Compute statistic
    statistic = numerator / denominator

    return float(statistic)

def _compute_p_value(
    statistic: float,
    df: int
) -> float:
    """
    Compute p-value from test statistic.

    Parameters
    ----------
    statistic : float
        Computed test statistic
    df : int
        Degrees of freedom

    Returns
    -------
    float
        Computed p-value
    """
    from scipy.stats import f

    # Conover test uses F distribution
    p_value = 1 - f.cdf(statistic, df, total_n - n_groups)

    return float(p_value)

def test_conover_fit(
    samples: np.ndarray,
    alpha: float = 0.05,
    equal_sample_sizes: bool = True
) -> Dict[str, Union[float, np.ndarray, Dict]]:
    """
    Perform Conover test for equality of variances.

    Parameters
    ----------
    samples : np.ndarray
        Array of samples to test. Shape (n_samples, n_groups)
    alpha : float, optional
        Significance level, by default 0.05
    equal_sample_sizes : bool, optional
        Whether to assume equal sample sizes, by default True

    Returns
    -------
    dict
        Dictionary containing:
        - 'statistic': computed test statistic
        - 'p_value': computed p-value
        - 'ranks': array of ranks for each sample
        - 'group_averages': average rank for each group
        - 'params_used': parameters used in the test
        - 'warnings': any warnings generated

    Examples
    --------
    >>> samples = np.random.randn(30, 4)  # 30 samples, 4 groups
    >>> result = test_conover_fit(samples)
    """
    # Validate inputs
    _validate_inputs(samples, alpha)

    n_samples, n_groups = samples.shape

    # Compute rank averages
    rank_results = _compute_rank_averages(samples)
    ranks = rank_results['ranks']
    group_averages = rank_results['group_averages']

    # Compute test statistic
    if equal_sample_sizes:
        n_samples_per_group = None
    else:
        # Count samples per group (assuming no missing values)
        n_samples_per_group = np.sum(~np.isnan(samples), axis=0)

    statistic = _compute_statistic(group_averages, n_samples_per_group)
    df = n_groups - 1

    # Compute p-value
    p_value = _compute_p_value(statistic, df)

    # Prepare results
    result = {
        'statistic': statistic,
        'p_value': p_value,
        'ranks': ranks,
        'group_averages': group_averages,
        'params_used': {
            'alpha': alpha,
            'equal_sample_sizes': equal_sample_sizes
        },
        'warnings': []
    }

    return result

################################################################################
# test_ochi
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    x: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> None:
    """Validate input arrays and apply normalization if specified."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays")
    if x.shape != y.shape:
        raise ValueError("Inputs must have the same shape")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("Input x contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Input y contains NaN or infinite values")

    if normalizer is not None:
        x = normalizer(x)
        y = normalizer(y)

    return x, y

def _compute_statistic(
    x: np.ndarray,
    y: np.ndarray,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = np.linalg.norm
) -> float:
    """Compute the Ochi test statistic."""
    residuals = x - y
    return distance_metric(residuals, np.zeros_like(residuals))

def _estimate_parameters(
    x: np.ndarray,
    y: np.ndarray,
    solver: str = "closed_form"
) -> Dict[str, float]:
    """Estimate parameters using the specified solver."""
    if solver == "closed_form":
        params = {
            'mean_x': np.mean(x),
            'var_x': np.var(x, ddof=1),
            'mean_y': np.mean(y),
            'var_y': np.var(y, ddof=1)
        }
    else:
        raise ValueError(f"Solver {solver} not implemented")

    return params

def _compute_metrics(
    x: np.ndarray,
    y: np.ndarray,
    metric_functions: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Compute specified metrics."""
    if metric_functions is None:
        metric_functions = {
            'mse': lambda x, y: np.mean((x - y) ** 2),
            'mae': lambda x, y: np.mean(np.abs(x - y))
        }

    metrics = {}
    for name, func in metric_functions.items():
        metrics[name] = func(x, y)

    return metrics

def test_ochi_fit(
    x: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = np.linalg.norm,
    solver: str = "closed_form",
    metric_functions: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, float], list]]:
    """
    Perform the Ochi test for variance comparison.

    Parameters
    ----------
    x : np.ndarray
        First input array.
    y : np.ndarray
        Second input array.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Normalization function to apply to inputs, by default None.
    distance_metric : Callable[[np.ndarray, np.ndarray], float], optional
        Distance metric function, by default np.linalg.norm.
    solver : str, optional
        Solver method, by default "closed_form".
    metric_functions : Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]], optional
        Dictionary of metric functions to compute, by default None.

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], Dict[str, float], list]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Example
    -------
    >>> x = np.random.randn(100)
    >>> y = np.random.randn(100) * 2
    >>> result = test_ochi_fit(x, y)
    """
    # Validate inputs and apply normalization
    x_valid, y_valid = _validate_inputs(x, y, normalizer)

    # Compute test statistic
    statistic = _compute_statistic(x_valid, y_valid, distance_metric)

    # Estimate parameters
    params = _estimate_parameters(x_valid, y_valid, solver)

    # Compute metrics
    metrics = _compute_metrics(x_valid, y_valid, metric_functions)

    # Prepare output
    result = {
        "result": statistic,
        "metrics": metrics,
        "params_used": params,
        "warnings": []
    }

    return result

################################################################################
# test_welch
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    samples: list,
    alpha: float = 0.05
) -> None:
    """
    Validate input samples and alpha level.

    Parameters
    ----------
    samples : list of array_like
        List of sample arrays to test.
    alpha : float, optional
        Significance level (default is 0.05).

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not all(isinstance(sample, (np.ndarray, list)) for sample in samples):
        raise ValueError("All samples must be numpy arrays or lists.")
    if not all(len(sample) > 1 for sample in samples):
        raise ValueError("All samples must have at least two observations.")
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")

def _calculate_statistic(
    samples: list,
    equal_var: bool = False
) -> float:
    """
    Calculate Welch's t-statistic.

    Parameters
    ----------
    samples : list of array_like
        List of sample arrays to test.
    equal_var : bool, optional
        Whether to assume equal variances (default is False).

    Returns
    ------
    float
        Welch's t-statistic.
    """
    samples = [np.asarray(sample) for sample in samples]
    k = len(samples)
    means = [np.mean(sample) for sample in samples]
    vars_ = [np.var(sample, ddof=1) for sample in samples]
    ns = [len(sample) for sample in samples]

    if equal_var:
        pooled_var = np.sum([(n - 1) * var for n, var in zip(ns, vars_)]) / np.sum([n - 1 for n in ns])
        se = np.sqrt(pooled_var * np.sum([1 / n for n in ns]))
    else:
        se = np.sqrt(np.sum([var / n for var, n in zip(vars_, ns)]))

    grand_mean = np.mean(means)
    numerator = np.sum([(mean - grand_mean) * n for mean, n in zip(means, ns)])
    t_stat = numerator / se

    return float(t_stat)

def _calculate_p_value(
    t_stat: float,
    df: int
) -> float:
    """
    Calculate p-value from t-statistic and degrees of freedom.

    Parameters
    ----------
    t_stat : float
        Welch's t-statistic.
    df : int
        Degrees of freedom.

    Returns
    ------
    float
        Two-tailed p-value.
    """
    from scipy.stats import t
    return 2 * (1 - t.cdf(np.abs(t_stat), df))

def _calculate_degrees_freedom(
    samples: list
) -> int:
    """
    Calculate degrees of freedom for Welch's test.

    Parameters
    ----------
    samples : list of array_like
        List of sample arrays to test.

    Returns
    ------
    int
        Degrees of freedom.
    """
    samples = [np.asarray(sample) for sample in samples]
    vars_ = [np.var(sample, ddof=1) for sample in samples]
    ns = [len(sample) for sample in samples]

    numerator = np.sum([(var / n)**2 for var, n in zip(vars_, ns)])
    denominator = np.sum([(var / n)**2 for var, n in zip(vars_, ns)])**2
    df = numerator / denominator

    return int(np.round(df))

def test_welch_fit(
    samples: list,
    alpha: float = 0.05,
    equal_var: bool = False
) -> Dict[str, Union[float, Dict]]:
    """
    Perform Welch's t-test for comparing means of multiple samples.

    Parameters
    ----------
    samples : list of array_like
        List of sample arrays to test.
    alpha : float, optional
        Significance level (default is 0.05).
    equal_var : bool, optional
        Whether to assume equal variances (default is False).

    Returns
    ------
    dict
        Dictionary containing:
        - result: str ('reject' or 'fail to reject')
        - p_value: float
        - t_statistic: float
        - degrees_freedom: int
        - alpha_used: float
        - warnings: list of str

    Examples
    --------
    >>> samples = [np.random.normal(0, 1, 30), np.random.normal(1, 2, 40)]
    >>> result = test_welch_fit(samples)
    """
    _validate_inputs(samples, alpha)

    t_stat = _calculate_statistic(samples, equal_var)
    df = _calculate_degrees_freedom(samples)
    p_value = _calculate_p_value(t_stat, df)

    result = 'reject' if p_value < alpha else 'fail to reject'

    return {
        "result": result,
        "p_value": p_value,
        "t_statistic": t_stat,
        "degrees_freedom": df,
        "alpha_used": alpha,
        "warnings": []
    }
