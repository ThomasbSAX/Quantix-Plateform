"""
Quantix – Module comparaison_moyennes
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# test_t_student
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def _validate_inputs(
    sample1: np.ndarray,
    sample2: np.ndarray,
    equal_var: bool = True
) -> None:
    """
    Validate input samples for t-test.

    Parameters
    ----------
    sample1 : np.ndarray
        First sample data.
    sample2 : np.ndarray
        Second sample data.
    equal_var : bool, optional
        Whether to assume equal variances.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not isinstance(sample1, np.ndarray) or not isinstance(sample2, np.ndarray):
        raise ValueError("Inputs must be numpy arrays")
    if sample1.ndim != 1 or sample2.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional")
    if np.isnan(sample1).any() or np.isnan(sample2).any():
        raise ValueError("Inputs contain NaN values")
    if np.isinf(sample1).any() or np.isinf(sample2).any():
        raise ValueError("Inputs contain infinite values")
    if len(sample1) < 2 or len(sample2) < 2:
        raise ValueError("Samples must have at least 2 elements")

def _calculate_statistic(
    sample1: np.ndarray,
    sample2: np.ndarray,
    equal_var: bool = True
) -> float:
    """
    Calculate t-statistic for independent samples.

    Parameters
    ----------
    sample1 : np.ndarray
        First sample data.
    sample2 : np.ndarray
        Second sample data.
    equal_var : bool, optional
        Whether to assume equal variances.

    Returns
    -------
    float
        Calculated t-statistic.
    """
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
    """
    Calculate p-value from t-statistic and degrees of freedom.

    Parameters
    ----------
    t_stat : float
        Calculated t-statistic.
    df : int
        Degrees of freedom.

    Returns
    -------
    float
        Calculated p-value.
    """
    from scipy.stats import t as t_dist
    return 2 * (1 - t_dist.cdf(abs(t_stat), df=df))

def test_t_student_fit(
    sample1: np.ndarray,
    sample2: np.ndarray,
    equal_var: bool = True,
    alternative: str = 'two-sided',
    custom_statistic_func: Optional[Callable] = None,
    custom_p_value_func: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Perform independent t-test for means of two samples.

    Parameters
    ----------
    sample1 : np.ndarray
        First sample data.
    sample2 : np.ndarray
        Second sample data.
    equal_var : bool, optional
        Whether to assume equal variances (default True).
    alternative : str, optional
        Alternative hypothesis ('two-sided', 'less', or 'greater').
    custom_statistic_func : Callable, optional
        Custom function to calculate statistic.
    custom_p_value_func : Callable, optional
        Custom function to calculate p-value.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing test results and metadata.

    Examples
    --------
    >>> sample1 = np.random.normal(0, 1, 100)
    >>> sample2 = np.random.normal(0.5, 1, 100)
    >>> result = test_t_student_fit(sample1, sample2)
    """
    _validate_inputs(sample1, sample2, equal_var)

    n1, n2 = len(sample1), len(sample2)
    df = n1 + n2 - 2 if equal_var else min(n1, n2) - 1

    if custom_statistic_func is not None:
        t_stat = custom_statistic_func(sample1, sample2)
    else:
        t_stat = _calculate_statistic(sample1, sample2, equal_var)

    if custom_p_value_func is not None:
        p_value = custom_p_value_func(t_stat, df)
    else:
        p_value = _calculate_p_value(t_stat, df)

    if alternative == 'less':
        p_value /= 2
    elif alternative == 'greater':
        p_value = 1 - p_value / 2

    return {
        "result": {
            "t_statistic": t_stat,
            "p_value": p_value,
            "degrees_of_freedom": df
        },
        "params_used": {
            "equal_var": equal_var,
            "alternative_hypothesis": alternative
        },
        "warnings": []
    }

################################################################################
# anova
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    data: np.ndarray,
    groups: np.ndarray,
    normalization: Optional[str] = None
) -> Dict[str, Union[np.ndarray, str]]:
    """
    Validate and preprocess input data for ANOVA.

    Parameters
    ----------
    data : np.ndarray
        Array of response variable values.
    groups : np.ndarray
        Array of group labels.
    normalization : str, optional
        Type of normalization to apply (none, standard, minmax, robust).

    Returns
    -------
    dict
        Dictionary containing validated and normalized data.
    """
    if not isinstance(data, np.ndarray) or not isinstance(groups, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if data.ndim != 1 or groups.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays")
    if len(data) != len(groups):
        raise ValueError("Data and groups must have the same length")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values")

    # Normalization
    if normalization == "standard":
        data = (data - np.mean(data)) / np.std(data)
    elif normalization == "minmax":
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
    elif normalization == "robust":
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        data = (data - median) / iqr

    return {"data": data, "groups": groups}

def _calculate_anova_statistics(
    data: np.ndarray,
    groups: np.ndarray
) -> Dict[str, float]:
    """
    Calculate ANOVA statistics.

    Parameters
    ----------
    data : np.ndarray
        Normalized response variable values.
    groups : np.ndarray
        Array of group labels.

    Returns
    -------
    dict
        Dictionary containing ANOVA statistics.
    """
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    group_sizes = [np.sum(groups == g) for g in unique_groups]
    total_n = len(data)

    # Calculate overall mean
    grand_mean = np.mean(data)

    # Calculate between-group sum of squares (SSB)
    group_means = np.array([np.mean(data[groups == g]) for g in unique_groups])
    ssb = np.sum([group_sizes[i] * (group_means[i] - grand_mean)**2 for i in range(n_groups)])

    # Calculate within-group sum of squares (SSW)
    ssw = np.sum([np.sum((data[groups == g] - group_means[i])**2) for i, g in enumerate(unique_groups)])

    # Calculate degrees of freedom
    dfb = n_groups - 1
    dwf = total_n - n_groups

    # Calculate mean squares
    msb = ssb / dfb if dfb > 0 else 0
    msw = ssw / dwf if dwf > 0 else 0

    # Calculate F-statistic
    f_stat = msb / msw if msw > 0 else np.inf

    return {
        "ssb": ssb,
        "ssw": ssw,
        "msb": msb,
        "msw": msw,
        "f_statistic": f_stat,
        "dfb": dfb,
        "dwf": dwf
    }

def _calculate_metrics(
    data: np.ndarray,
    groups: np.ndarray,
    metrics: Optional[Union[str, Callable]] = None
) -> Dict[str, float]:
    """
    Calculate additional metrics for ANOVA.

    Parameters
    ----------
    data : np.ndarray
        Normalized response variable values.
    groups : np.ndarray
        Array of group labels.
    metrics : str or callable, optional
        Metrics to calculate (mse, mae, r2) or custom callable.

    Returns
    -------
    dict
        Dictionary containing calculated metrics.
    """
    unique_groups = np.unique(groups)
    group_sizes = [np.sum(groups == g) for g in unique_groups]
    total_n = len(data)

    # Calculate group means
    group_means = np.array([np.mean(data[groups == g]) for g in unique_groups])

    # Calculate overall mean
    grand_mean = np.mean(data)

    metrics_dict = {}

    if metrics == "mse" or (callable(metrics) and metrics.__name__ == "mse"):
        # Mean Squared Error
        mse = np.sum((data - group_means[groups])**2) / total_n
        metrics_dict["mse"] = mse

    if metrics == "mae" or (callable(metrics) and metrics.__name__ == "mae"):
        # Mean Absolute Error
        mae = np.mean(np.abs(data - group_means[groups]))
        metrics_dict["mae"] = mae

    if metrics == "r2" or (callable(metrics) and metrics.__name__ == "r2"):
        # R-squared
        ss_total = np.sum((data - grand_mean)**2)
        ss_residual = np.sum((data - group_means[groups])**2)
        r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
        metrics_dict["r2"] = r2

    if callable(metrics) and not any([metrics.__name__ == m for m in ["mse", "mae", "r2"]]):
        # Custom metric
        custom_metric = metrics(data, groups)
        metrics_dict["custom_metric"] = custom_metric

    return metrics_dict

def anova_fit(
    data: np.ndarray,
    groups: np.ndarray,
    normalization: Optional[str] = None,
    metrics: Optional[Union[str, Callable]] = None
) -> Dict[str, Union[Dict, float, str]]:
    """
    Perform ANOVA analysis.

    Parameters
    ----------
    data : np.ndarray
        Array of response variable values.
    groups : np.ndarray
        Array of group labels.
    normalization : str, optional
        Type of normalization to apply (none, standard, minmax, robust).
    metrics : str or callable, optional
        Metrics to calculate (mse, mae, r2) or custom callable.

    Returns
    -------
    dict
        Dictionary containing ANOVA results, metrics, parameters used, and warnings.

    Example
    -------
    >>> data = np.array([1.2, 2.3, 3.4, 4.5, 5.6])
    >>> groups = np.array([0, 0, 1, 1, 2])
    >>> result = anova_fit(data, groups, normalization="standard", metrics="mse")
    """
    # Validate and preprocess inputs
    validated_data = _validate_inputs(data, groups, normalization)
    data = validated_data["data"]
    groups = validated_data["groups"]

    # Calculate ANOVA statistics
    anova_stats = _calculate_anova_statistics(data, groups)

    # Calculate metrics if requested
    metrics_dict = {}
    if metrics is not None:
        metrics_dict = _calculate_metrics(data, groups, metrics)

    # Prepare output
    result = {
        "result": anova_stats,
        "metrics": metrics_dict,
        "params_used": {
            "normalization": normalization,
            "metrics": str(metrics) if metrics is not None else None
        },
        "warnings": []
    }

    return result

# Example of custom metric function
def custom_metric(data: np.ndarray, groups: np.ndarray) -> float:
    """
    Example of a custom metric function for ANOVA.
    """
    unique_groups = np.unique(groups)
    group_means = np.array([np.mean(data[groups == g]) for g in unique_groups])
    return np.mean(np.abs(group_means - np.mean(group_means)))

################################################################################
# test_welch
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    x1: np.ndarray,
    x2: np.ndarray,
    alpha: float = 0.05
) -> None:
    """Validate input arrays and significance level."""
    if not isinstance(x1, np.ndarray) or not isinstance(x2, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional")
    if np.isnan(x1).any() or np.isnan(x2).any():
        raise ValueError("Input arrays contain NaN values")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha must be between 0 and 1")

def _calculate_statistic(
    x1: np.ndarray,
    x2: np.ndarray
) -> float:
    """Calculate Welch's t-statistic."""
    n1, n2 = len(x1), len(x2)
    mean1, mean2 = np.mean(x1), np.mean(x2)
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    numerator = mean1 - mean2
    denominator = np.sqrt(var1/n1 + var2/n2)
    return numerator / denominator

def _calculate_p_value(
    t_stat: float,
    df: float
) -> float:
    """Calculate p-value from t-statistic and degrees of freedom."""
    # Using scipy's t distribution for illustration
    from scipy.stats import t
    return 2 * (1 - t.cdf(abs(t_stat), df))

def _calculate_degrees_freedom(
    x1: np.ndarray,
    x2: np.ndarray
) -> float:
    """Calculate Welch-Satterthwaite equation for degrees of freedom."""
    n1, n2 = len(x1), len(x2)
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    numerator = (var1/n1 + var2/n2)**2
    denominator = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
    return numerator / denominator

def test_welch_fit(
    x1: np.ndarray,
    x2: np.ndarray,
    alpha: float = 0.05,
    alternative: str = 'two-sided',
    equal_var: bool = False
) -> Dict[str, Union[float, Dict]]:
    """
    Perform Welch's t-test for comparing means of two independent samples.

    Parameters
    ----------
    x1 : np.ndarray
        First sample array.
    x2 : np.ndarray
        Second sample array.
    alpha : float, optional
        Significance level (default is 0.05).
    alternative : str, optional
        Alternative hypothesis ('two-sided', 'less', or 'greater').
    equal_var : bool, optional
        If True, perform Student's t-test assuming equal variances.

    Returns
    -------
    dict
        Dictionary containing test results, metrics, parameters used, and warnings.

    Example
    -------
    >>> x1 = np.random.normal(0, 1, 100)
    >>> x2 = np.random.normal(0.5, 1, 100)
    >>> result = test_welch_fit(x1, x2)
    """
    _validate_inputs(x1, x2, alpha)

    if equal_var:
        # Implement Student's t-test logic here
        raise NotImplementedError("Student's t-test not implemented yet")
    else:
        # Welch's t-test implementation
        t_stat = _calculate_statistic(x1, x2)
        df = _calculate_degrees_freedom(x1, x2)
        p_value = _calculate_p_value(t_stat, df)

    # Determine if we reject the null hypothesis
    if alternative == 'two-sided':
        reject = p_value < alpha
    elif alternative == 'less':
        reject = p_value / 2 < alpha
    elif alternative == 'greater':
        reject = p_value / 2 < alpha
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    return {
        "result": {
            "t_statistic": t_stat,
            "p_value": p_value,
            "degrees_freedom": df,
            "reject_null": reject
        },
        "metrics": {
            "mean1": np.mean(x1),
            "mean2": np.mean(x2),
            "var1": np.var(x1, ddof=1),
            "var2": np.var(x2, ddof=1)
        },
        "params_used": {
            "alpha": alpha,
            "alternative": alternative,
            "equal_var": equal_var
        },
        "warnings": []
    }

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
    """Validate input samples for Mann-Whitney U test."""
    if not isinstance(sample1, np.ndarray) or not isinstance(sample2, np.ndarray):
        raise TypeError("Samples must be numpy arrays")
    if sample1.ndim != 1 or sample2.ndim != 1:
        raise ValueError("Samples must be 1-dimensional")
    if len(sample1) == 0 or len(sample2) == 0:
        raise ValueError("Samples cannot be empty")
    if np.any(np.isnan(sample1)) or np.any(np.isnan(sample2)):
        raise ValueError("Samples contain NaN values")
    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError("Alternative must be 'two-sided', 'less' or 'greater'")

def _compute_rank_sums(
    sample1: np.ndarray,
    sample2: np.ndarray
) -> tuple:
    """Compute rank sums for Mann-Whitney U test."""
    combined = np.concatenate([sample1, sample2])
    ranks = np.argsort(np.argsort(combined))
    rank_sum1 = np.sum(ranks[:len(sample1)])
    rank_sum2 = np.sum(ranks[len(sample1):])
    return rank_sum1, rank_sum2

def _compute_u_statistic(
    sample1: np.ndarray,
    sample2: np.ndarray
) -> float:
    """Compute Mann-Whitney U statistic."""
    n1, n2 = len(sample1), len(sample2)
    rank_sum1, _ = _compute_rank_sums(sample1, sample2)
    u1 = rank_sum1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    return min(u1, u2)

def _compute_p_value(
    u_statistic: float,
    n1: int,
    n2: int,
    alternative: str
) -> float:
    """Compute p-value for Mann-Whitney U test."""
    from scipy.stats import mannwhitneyu
    _, p_value = mannwhitneyu(
        np.zeros(n1),
        np.zeros(n2),
        alternative=alternative,
        method='exact'
    )
    return p_value

def test_mann_whitney_fit(
    sample1: np.ndarray,
    sample2: np.ndarray,
    alternative: str = 'two-sided',
    correction: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Perform Mann-Whitney U test to compare two independent samples.

    Parameters:
    -----------
    sample1 : np.ndarray
        First sample of observations.
    sample2 : np.ndarray
        Second sample of observations.
    alternative : str, optional
        Alternative hypothesis ('two-sided', 'less' or 'greater'), default 'two-sided'.
    correction : Callable, optional
        Optional function to apply corrections to p-values.

    Returns:
    --------
    dict
        Dictionary containing test results, metrics and parameters used.
    """
    _validate_inputs(sample1, sample2, alternative)

    n1, n2 = len(sample1), len(sample2)
    u_statistic = _compute_u_statistic(sample1, sample2)
    p_value = _compute_p_value(u_statistic, n1, n2, alternative)

    if correction is not None:
        p_value = correction(p_value)

    result = {
        'statistic': u_statistic,
        'pvalue': p_value,
        'alternative': alternative
    }

    return {
        'result': result,
        'metrics': {},
        'params_used': {
            'alternative': alternative,
            'correction': correction.__name__ if correction else None
        },
        'warnings': []
    }

# Example usage:
"""
sample1 = np.random.normal(0, 1, 100)
sample2 = np.random.normal(0.5, 1, 100)
result = test_mann_whitney_fit(sample1, sample2)
"""

################################################################################
# test_kruskal_wallis
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def _validate_inputs(
    groups: List[np.ndarray],
    normalizations: Optional[List[str]] = None,
    metric: str = "euclidean",
    custom_metric: Optional[Callable] = None,
) -> Dict[str, np.ndarray]:
    """
    Validate and preprocess input data for Kruskal-Wallis test.

    Parameters
    ----------
    groups : List[np.ndarray]
        List of arrays containing the samples for each group.
    normalizations : Optional[List[str]], default=None
        List of normalization methods to apply to each group.
    metric : str, default="euclidean"
        Distance metric to use for ranking.
    custom_metric : Optional[Callable], default=None
        Custom distance metric function.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing validated and normalized groups.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not all(isinstance(group, np.ndarray) for group in groups):
        raise ValueError("All groups must be numpy arrays.")

    if len(groups) < 2:
        raise ValueError("At least two groups are required.")

    if normalizations is not None and len(normalizations) != len(groups):
        raise ValueError("Number of normalizations must match number of groups.")

    normalized_groups = []
    for i, group in enumerate(groups):
        if np.any(np.isnan(group)) or np.any(np.isinf(group)):
            raise ValueError(f"Group {i} contains NaN or infinite values.")

        if normalizations is not None and normalizations[i] == "standard":
            group = (group - np.mean(group)) / np.std(group)
        elif normalizations is not None and normalizations[i] == "minmax":
            group = (group - np.min(group)) / (np.max(group) - np.min(group))
        elif normalizations is not None and normalizations[i] == "robust":
            group = (group - np.median(group)) / (np.percentile(group, 75) - np.percentile(group, 25))

        normalized_groups.append(group)

    return {"groups": normalized_groups}

def _compute_ranks(
    groups: List[np.ndarray],
    metric: str = "euclidean",
    custom_metric: Optional[Callable] = None,
) -> np.ndarray:
    """
    Compute ranks for each sample across all groups.

    Parameters
    ----------
    groups : List[np.ndarray]
        List of arrays containing the samples for each group.
    metric : str, default="euclidean"
        Distance metric to use for ranking.
    custom_metric : Optional[Callable], default=None
        Custom distance metric function.

    Returns
    -------
    np.ndarray
        Array of ranks for all samples.
    """
    combined = np.concatenate(groups)
    n_samples = len(combined)

    if metric == "euclidean":
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                distances[i, j] = np.linalg.norm(combined[i] - combined[j])
    elif metric == "manhattan":
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                distances[i, j] = np.sum(np.abs(combined[i] - combined[j]))
    elif metric == "cosine":
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                distances[i, j] = 1 - np.dot(combined[i], combined[j]) / (np.linalg.norm(combined[i]) * np.linalg.norm(combined[j]))
    elif custom_metric is not None:
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                distances[i, j] = custom_metric(combined[i], combined[j])
    else:
        raise ValueError("Invalid metric specified.")

    ranks = np.zeros(n_samples)
    for i in range(n_samples):
        ranks[i] = np.sum(distances[i, :]) + 1

    return ranks

def _kruskal_wallis_statistic(
    groups: List[np.ndarray],
    ranks: np.ndarray,
) -> float:
    """
    Compute the Kruskal-Wallis test statistic.

    Parameters
    ----------
    groups : List[np.ndarray]
        List of arrays containing the samples for each group.
    ranks : np.ndarray
        Array of ranks for all samples.

    Returns
    -------
    float
        Kruskal-Wallis test statistic.
    """
    n_groups = len(groups)
    n_samples = sum(len(group) for group in groups)

    R_j = np.zeros(n_groups)
    for j in range(n_groups):
        R_j[j] = np.sum(ranks[:sum(len(g) for g in groups[:j+1])][-len(groups[j]):])

    H = (12 / (n_samples * (n_samples + 1))) * np.sum(R_j**2 / len(groups[j])) - 3 * (n_samples + 1)

    return H

def test_kruskal_wallis_fit(
    groups: List[np.ndarray],
    normalizations: Optional[List[str]] = None,
    metric: str = "euclidean",
    custom_metric: Optional[Callable] = None,
) -> Dict[str, Union[float, Dict, List]]:
    """
    Perform the Kruskal-Wallis test for comparing means across multiple groups.

    Parameters
    ----------
    groups : List[np.ndarray]
        List of arrays containing the samples for each group.
    normalizations : Optional[List[str]], default=None
        List of normalization methods to apply to each group.
    metric : str, default="euclidean"
        Distance metric to use for ranking.
    custom_metric : Optional[Callable], default=None
        Custom distance metric function.

    Returns
    -------
    Dict[str, Union[float, Dict, List]]
        Dictionary containing the test result, metrics, parameters used, and warnings.
    """
    validated = _validate_inputs(groups, normalizations, metric, custom_metric)
    groups_validated = validated["groups"]

    ranks = _compute_ranks(groups_validated, metric, custom_metric)
    H_statistic = _kruskal_wallis_statistic(groups_validated, ranks)

    result = {
        "result": H_statistic,
        "metrics": {},
        "params_used": {
            "normalizations": normalizations,
            "metric": metric,
            "custom_metric": custom_metric is not None,
        },
        "warnings": [],
    }

    return result

################################################################################
# ecart_type
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    normalize: str = "none",
    metric: Union[str, Callable] = "euclidean"
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("X contains infinite values")

    valid_normalizations = ["none", "standard", "minmax", "robust"]
    if normalize not in valid_normalizations:
        raise ValueError(f"normalize must be one of {valid_normalizations}")

    valid_metrics = ["euclidean", "manhattan", "cosine", "minkowski"]
    if isinstance(metric, str) and metric not in valid_metrics:
        raise ValueError(f"metric must be one of {valid_metrics} or a callable")

def _normalize_data(
    X: np.ndarray,
    normalize: str = "none"
) -> np.ndarray:
    """Normalize the input data."""
    if normalize == "standard":
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalize == "minmax":
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalize == "robust":
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    return X

def _compute_metric(
    X: np.ndarray,
    metric: Union[str, Callable] = "euclidean",
    p: float = 2.0
) -> np.ndarray:
    """Compute the specified metric."""
    if isinstance(metric, str):
        if metric == "euclidean":
            return np.sqrt(np.sum(X**2, axis=1))
        elif metric == "manhattan":
            return np.sum(np.abs(X), axis=1)
        elif metric == "cosine":
            norms = np.linalg.norm(X, axis=1)
            return 1 - (X @ X.T) / np.outer(norms, norms)
        elif metric == "minkowski":
            return np.sum(np.abs(X)**p, axis=1) ** (1/p)
    else:
        return metric(X)

def ecart_type_fit(
    X: np.ndarray,
    normalize: str = "none",
    metric: Union[str, Callable] = "euclidean",
    p: float = 2.0,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Compute the standard deviation of specified metrics for comparison of means.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix (n_samples, n_features).
    normalize : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    metric : str or callable, optional
        Metric to compute ("euclidean", "manhattan", "cosine", "minkowski") or custom callable.
    p : float, optional
        Power parameter for Minkowski distance (default 2.0).
    **kwargs : dict
        Additional keyword arguments for custom metrics.

    Returns
    -------
    result : Dict[str, Union[np.ndarray, Dict]]
        Dictionary containing:
            - "result": Computed standard deviations
            - "metrics": Metrics used
            - "params_used": Parameters actually used
            - "warnings": Any warnings generated

    Examples
    --------
    >>> X = np.random.randn(100, 5)
    >>> result = ecart_type_fit(X, normalize="standard", metric="euclidean")
    """
    # Validate inputs
    _validate_inputs(X, normalize, metric)

    warnings = []
    params_used = {
        "normalize": normalize,
        "metric": metric,
        "p": p
    }

    # Normalize data if required
    X_normalized = _normalize_data(X, normalize)

    # Compute metric
    metrics = _compute_metric(X_normalized, metric, p)

    # Calculate standard deviation
    result = np.std(metrics, axis=0) if metrics.ndim > 1 else np.std(metrics)

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

################################################################################
# variance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def variance_fit(
    data: np.ndarray,
    *,
    ddof: int = 1,
    axis: Optional[int] = None,
    weights: Optional[np.ndarray] = None,
    normalize: bool = False
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate the variance of a dataset with various options.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    ddof : int, optional
        Delta degrees of freedom. The divisor used in the calculation is N - ddof,
        where N is the number of elements. By default 1 for sample variance.
    axis : int, optional
        Axis along which to compute the variance. If None, compute over the whole array.
    weights : np.ndarray, optional
        Array of weights associated with the data. If None, uniform weights are used.
    normalize : bool, optional
        Whether to return the normalized variance (divided by mean of data).

    Returns
    -------
    Dict[str, Union[float, np.ndarray]]
        Dictionary containing:
        - "result": computed variance
        - "metrics": additional metrics (normalized variance if requested)
        - "params_used": parameters used in the computation
        - "warnings": any warnings encountered

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> variance_fit(data)
    {
        'result': 2.0,
        'metrics': {},
        'params_used': {'ddof': 1, 'axis': None, 'weights': None, 'normalize': False},
        'warnings': []
    }
    """
    # Validate inputs
    _validate_inputs(data, weights=weights, axis=axis)

    # Calculate variance
    if weights is None:
        result = np.var(data, ddof=ddof, axis=axis)
    else:
        if axis is None:
            result = np.average((data - np.mean(data, weights=weights))**2, weights=weights)
        else:
            result = np.apply_along_axis(
                lambda x: np.average((x - np.mean(x, weights=weights))**2, weights=weights),
                axis=axis,
                arr=data
            )

    # Calculate additional metrics if requested
    metrics = {}
    if normalize:
        mean_val = np.mean(data, axis=axis) if weights is None else np.average(data, weights=weights)
        metrics['normalized_variance'] = result / mean_val if mean_val != 0 else np.nan

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'ddof': ddof,
            'axis': axis,
            'weights': weights is not None,
            'normalize': normalize
        },
        'warnings': _check_warnings(data, result)
    }

    return output

def _validate_inputs(
    data: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    axis: Optional[int] = None
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")

    if weights is not None:
        if not isinstance(weights, np.ndarray):
            raise TypeError("Weights must be a numpy array")
        if weights.shape != data.shape:
            raise ValueError("Weights must have the same shape as data")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative")

    if axis is not None:
        if not isinstance(axis, int):
            raise TypeError("Axis must be an integer")
        if axis >= data.ndim or axis < -data.ndim:
            raise ValueError("Axis out of bounds")

    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def _check_warnings(
    data: np.ndarray,
    result: Union[float, np.ndarray]
) -> list:
    """Check for potential warnings in the computation."""
    warnings = []

    if np.any(np.isnan(result)):
        warnings.append("Result contains NaN values")
    if np.any(np.isinf(result)):
        warnings.append("Result contains infinite values")

    return warnings

################################################################################
# intervalle_confiance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def intervalle_confiance_fit(
    data: np.ndarray,
    alpha: float = 0.05,
    method: str = 'z',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    **kwargs
) -> Dict[str, Union[float, Dict, str]]:
    """
    Calculate confidence interval for the mean of a dataset.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    alpha : float, optional
        Significance level (default is 0.05).
    method : str, optional
        Method to calculate confidence interval ('z' or 't', default is 'z').
    normalizer : Callable, optional
        Function to normalize data before calculation.
    **kwargs :
        Additional parameters for the normalizer function.

    Returns
    -------
    Dict[str, Union[float, Dict, str]]
        Dictionary containing:
        - "result": tuple of (lower_bound, upper_bound)
        - "metrics": dictionary of metrics
        - "params_used": dictionary of parameters used
        - "warnings": list of warnings

    Examples
    --------
    >>> data = np.random.normal(0, 1, 100)
    >>> result = intervalle_confiance_fit(data)
    """
    # Validate inputs
    data = _validate_input(data)

    # Normalize data if normalizer is provided
    if normalizer is not None:
        try:
            data = normalizer(data, **kwargs)
        except Exception as e:
            return {
                "result": None,
                "metrics": {},
                "params_used": {"alpha": alpha, "method": method},
                "warnings": [f"Normalization failed: {str(e)}"]
            }

    # Calculate confidence interval
    if method == 'z':
        lower_bound, upper_bound = _calculate_z_interval(data, alpha)
    elif method == 't':
        lower_bound, upper_bound = _calculate_t_interval(data, alpha)
    else:
        return {
            "result": None,
            "metrics": {},
            "params_used": {"alpha": alpha, "method": method},
            "warnings": ["Invalid method specified"]
        }

    # Calculate metrics
    metrics = _calculate_metrics(data, lower_bound, upper_bound)

    return {
        "result": (lower_bound, upper_bound),
        "metrics": metrics,
        "params_used": {"alpha": alpha, "method": method},
        "warnings": []
    }

def _validate_input(data: np.ndarray) -> np.ndarray:
    """
    Validate input data.

    Parameters
    ----------
    data : np.ndarray
        Input data array.

    Returns
    -------
    np.ndarray
        Validated data array.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values")
    return data

def _calculate_z_interval(data: np.ndarray, alpha: float) -> tuple:
    """
    Calculate confidence interval using z-distribution.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    alpha : float
        Significance level.

    Returns
    -------
    tuple
        Tuple of (lower_bound, upper_bound).
    """
    mean = np.mean(data)
    std_err = np.std(data) / np.sqrt(len(data))
    z_score = _get_z_score(alpha)
    margin_of_error = z_score * std_err
    return (mean - margin_of_error, mean + margin_of_error)

def _calculate_t_interval(data: np.ndarray, alpha: float) -> tuple:
    """
    Calculate confidence interval using t-distribution.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    alpha : float
        Significance level.

    Returns
    -------
    tuple
        Tuple of (lower_bound, upper_bound).
    """
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(len(data))
    t_score = _get_t_score(alpha, len(data) - 1)
    margin_of_error = t_score * std_err
    return (mean - margin_of_error, mean + margin_of_error)

def _get_z_score(alpha: float) -> float:
    """
    Get z-score for given significance level.

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

def _get_t_score(alpha: float, df: int) -> float:
    """
    Get t-score for given significance level and degrees of freedom.

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

def _calculate_metrics(data: np.ndarray, lower_bound: float, upper_bound: float) -> Dict[str, float]:
    """
    Calculate metrics for the confidence interval.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    lower_bound : float
        Lower bound of the confidence interval.
    upper_bound : float
        Upper bound of the confidence interval.

    Returns
    -------
    Dict[str, float]
        Dictionary of metrics.
    """
    mean = np.mean(data)
    std = np.std(data)
    return {
        "mean": mean,
        "std_dev": std,
        "interval_width": upper_bound - lower_bound,
        "coverage_probability": 1 - _calculate_non_coverage_probability(data, lower_bound, upper_bound)
    }

def _calculate_non_coverage_probability(data: np.ndarray, lower_bound: float, upper_bound: float) -> float:
    """
    Calculate the probability that the true mean is not in the confidence interval.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    lower_bound : float
        Lower bound of the confidence interval.
    upper_bound : float
        Upper bound of the confidence interval.

    Returns
    -------
    float
        Non-coverage probability.
    """
    mean = np.mean(data)
    if mean < lower_bound or mean > upper_bound:
        return 1.0
    return 0.0

################################################################################
# effect_size
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    group1: np.ndarray,
    group2: np.ndarray,
    metric_func: Optional[Callable] = None
) -> None:
    """Validate input arrays and optional metric function."""
    if not isinstance(group1, np.ndarray) or not isinstance(group2, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if group1.ndim != 1 or group2.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional")
    if len(group1) != len(group2):
        raise ValueError("Input arrays must have the same length")
    if np.any(np.isnan(group1)) or np.any(np.isnan(group2)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(group1)) or np.any(np.isinf(group2)):
        raise ValueError("Input arrays must not contain infinite values")
    if metric_func is not None and not callable(metric_func):
        raise TypeError("Metric function must be callable if provided")

def _compute_statistic(
    group1: np.ndarray,
    group2: np.ndarray,
    statistic_type: str = "cohen_d"
) -> float:
    """Compute effect size statistic based on selected method."""
    if statistic_type == "cohen_d":
        pooled_std = np.sqrt(((len(group1)-1)*np.var(group1, ddof=1) +
                             (len(group2)-1)*np.var(group2, ddof=1)) /
                            (len(group1) + len(group2) - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    elif statistic_type == "hedges_g":
        cohen_d = _compute_statistic(group1, group2, "cohen_d")
        n_total = len(group1) + len(group2)
        return cohen_d * (1 - 3/(4*(n_total-9)))
    elif statistic_type == "glass_delta":
        return (np.mean(group1) - np.mean(group2)) / np.std(group2, ddof=1)
    else:
        raise ValueError(f"Unknown statistic type: {statistic_type}")

def _compute_metrics(
    group1: np.ndarray,
    group2: np.ndarray,
    metric_func: Optional[Callable] = None
) -> Dict[str, float]:
    """Compute additional metrics for effect size analysis."""
    metrics = {
        "mean_diff": np.mean(group1) - np.mean(group2),
        "std_diff": np.std(group1, ddof=1) - np.std(group2, ddof=1),
        "var_ratio": np.var(group1, ddof=1) / np.var(group2, ddof=1)
    }

    if metric_func is not None:
        metrics["custom_metric"] = metric_func(group1, group2)

    return metrics

def effect_size_fit(
    group1: np.ndarray,
    group2: np.ndarray,
    statistic_type: str = "cohen_d",
    metric_func: Optional[Callable] = None,
    normalize: bool = False
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Compute effect size between two groups with configurable options.

    Parameters:
    -----------
    group1 : np.ndarray
        First group of observations
    group2 : np.ndarray
        Second group of observations
    statistic_type : str, optional
        Type of effect size statistic to compute (default: "cohen_d")
    metric_func : Callable, optional
        Custom metric function that takes two arrays and returns a float
    normalize : bool, optional
        Whether to standardize the groups before computation (default: False)

    Returns:
    --------
    dict
        Dictionary containing:
        - "result": computed effect size statistic
        - "metrics": additional computed metrics
        - "params_used": parameters used in computation
        - "warnings": any warnings generated during computation

    Example:
    --------
    >>> group1 = np.random.normal(0, 1, 100)
    >>> group2 = np.random.normal(0.5, 1, 100)
    >>> result = effect_size_fit(group1, group2)
    """
    # Initialize output dictionary
    output: Dict[str, Union[float, Dict[str, float], str]] = {
        "result": None,
        "metrics": {},
        "params_used": {
            "statistic_type": statistic_type,
            "normalize": normalize
        },
        "warnings": []
    }

    # Validate inputs
    _validate_inputs(group1, group2, metric_func)

    # Normalize if requested
    if normalize:
        mean1, std1 = np.mean(group1), np.std(group1)
        mean2, std2 = np.mean(group2), np.std(group2)

        if std1 == 0 or std2 == 0:
            output["warnings"].append("Standard deviation of one group is zero - normalization skipped")
        else:
            group1 = (group1 - mean1) / std1
            group2 = (group2 - mean2) / std2

    # Compute effect size statistic
    try:
        output["result"] = _compute_statistic(group1, group2, statistic_type)
    except Exception as e:
        output["warnings"].append(f"Error computing statistic: {str(e)}")
        return output

    # Compute additional metrics
    try:
        output["metrics"] = _compute_metrics(group1, group2, metric_func)
    except Exception as e:
        output["warnings"].append(f"Error computing metrics: {str(e)}")

    return output

################################################################################
# p_value
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def p_value_fit(
    data: np.ndarray,
    group_labels: np.ndarray,
    test_type: str = 'ttest',
    normalization: Optional[str] = None,
    metric: Union[str, Callable] = 'mean_diff',
    distance: Optional[Union[str, Callable]] = None,
    solver: str = 'closed_form',
    alpha: float = 0.05,
    custom_test_func: Optional[Callable] = None
) -> Dict:
    """
    Compute p-value for comparing means between groups.

    Parameters:
    -----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features)
    group_labels : np.ndarray
        Group labels for each sample of shape (n_samples,)
    test_type : str
        Type of statistical test ('ttest', 'mannwhitney', 'anova')
    normalization : str or None
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable
        Metric to compare means ('mean_diff', 'median_diff', custom function)
    distance : str or callable or None
        Distance metric for non-parametric tests (None for parametric)
    solver : str
        Solver method ('closed_form', 'gradient_descent')
    alpha : float
        Significance level for p-value calculation
    custom_test_func : callable or None
        Custom test function that overrides default methods

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> data = np.random.randn(100, 5)
    >>> labels = np.random.randint(0, 2, size=100)
    >>> result = p_value_fit(data, labels, test_type='ttest')
    """
    # Validate inputs
    _validate_inputs(data, group_labels)

    # Normalize data if specified
    normalized_data = _apply_normalization(data, normalization)

    # Prepare groups
    groups = _prepare_groups(normalized_data, group_labels)

    # Choose test method
    if custom_test_func is not None:
        p_value = _custom_test(groups, custom_test_func)
    else:
        if test_type == 'ttest':
            p_value = _ttest(groups, metric)
        elif test_type == 'mannwhitney':
            p_value = _mann_whitney_u(groups, distance)
        elif test_type == 'anova':
            p_value = _anova(groups)
        else:
            raise ValueError(f"Unknown test type: {test_type}")

    # Calculate metrics
    metrics = _calculate_metrics(groups, metric)

    return {
        'result': {'p_value': p_value},
        'metrics': metrics,
        'params_used': {
            'test_type': test_type,
            'normalization': normalization,
            'metric': metric,
            'distance': distance,
            'solver': solver
        },
        'warnings': _check_warnings(groups)
    }

def _validate_inputs(data: np.ndarray, group_labels: np.ndarray) -> None:
    """Validate input data and labels."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if not isinstance(group_labels, np.ndarray):
        raise TypeError("Group labels must be a numpy array")
    if data.shape[0] != group_labels.shape[0]:
        raise ValueError("Data and labels must have same number of samples")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")

def _apply_normalization(
    data: np.ndarray,
    method: Optional[str]
) -> np.ndarray:
    """Apply specified normalization to data."""
    if method is None or method == 'none':
        return data
    elif method == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == 'robust':
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _prepare_groups(
    data: np.ndarray,
    group_labels: np.ndarray
) -> Dict[int, np.ndarray]:
    """Prepare data grouped by labels."""
    groups = {}
    unique_labels = np.unique(group_labels)
    for label in unique_labels:
        groups[label] = data[group_labels == label]
    return groups

def _ttest(
    groups: Dict[int, np.ndarray],
    metric: Union[str, Callable]
) -> float:
    """Perform t-test between groups."""
    from scipy.stats import ttest_ind
    group1 = groups[np.min(list(groups.keys()))]
    group2 = groups[np.max(list(groups.keys()))]

    if metric == 'mean_diff':
        return ttest_ind(group1, group2).pvalue
    elif callable(metric):
        return ttest_ind(group1, group2).pvalue
    else:
        raise ValueError(f"Unknown metric for t-test: {metric}")

def _mann_whitney_u(
    groups: Dict[int, np.ndarray],
    distance: Optional[Union[str, Callable]]
) -> float:
    """Perform Mann-Whitney U test between groups."""
    from scipy.stats import mannwhitneyu
    group1 = groups[np.min(list(groups.keys()))]
    group2 = groups[np.max(list(groups.keys()))]

    if distance is None:
        return mannwhitneyu(group1, group2).pvalue
    elif callable(distance):
        # Custom distance implementation would go here
        return mannwhitneyu(group1, group2).pvalue
    else:
        raise ValueError(f"Unknown distance metric: {distance}")

def _anova(
    groups: Dict[int, np.ndarray]
) -> float:
    """Perform ANOVA test between multiple groups."""
    from scipy.stats import f_oneway
    return f_oneway(*groups.values()).pvalue

def _custom_test(
    groups: Dict[int, np.ndarray],
    test_func: Callable
) -> float:
    """Apply custom test function to groups."""
    return test_func(groups)

def _calculate_metrics(
    groups: Dict[int, np.ndarray],
    metric: Union[str, Callable]
) -> Dict:
    """Calculate comparison metrics between groups."""
    group1 = groups[np.min(list(groups.keys()))]
    group2 = groups[np.max(list(groups.keys()))]

    metrics = {}

    if metric == 'mean_diff':
        metrics['mean_diff'] = np.mean(group1) - np.mean(group2)
    elif metric == 'median_diff':
        metrics['median_diff'] = np.median(group1) - np.median(group2)
    elif callable(metric):
        metrics['custom_metric'] = metric(group1, group2)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

def _check_warnings(
    groups: Dict[int, np.ndarray]
) -> list:
    """Check for potential issues with the data."""
    warnings = []

    for label, group in groups.items():
        if len(group) < 3:
            warnings.append(f"Group {label} has only {len(group)} samples (minimum 3 recommended)")
        if np.std(group) == 0:
            warnings.append(f"Group {label} has zero variance")

    return warnings
