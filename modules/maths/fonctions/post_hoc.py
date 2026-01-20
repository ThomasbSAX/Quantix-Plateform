"""
Quantix – Module post_hoc
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# tukey_test
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    y: np.ndarray,
    groups: np.ndarray,
    alpha: float = 0.05
) -> None:
    """Validate input data for Tukey's HSD test."""
    if not isinstance(y, np.ndarray) or not isinstance(groups, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y.ndim != 1 or groups.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional")
    if len(y) != len(groups):
        raise ValueError("y and groups must have the same length")
    if not np.isfinite(y).all():
        raise ValueError("y contains non-finite values")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be between 0 and 1")

def _compute_group_means(y: np.ndarray, groups: np.ndarray) -> Dict[str, Union[np.ndarray, int]]:
    """Compute group means and counts."""
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    means = np.array([np.mean(y[groups == g]) for g in unique_groups])
    counts = np.array([np.sum(groups == g) for g in unique_groups])

    return {
        "means": means,
        "counts": counts,
        "n_groups": n_groups
    }

def _compute_q_critical(n_groups: int, df_error: int) -> float:
    """Compute the critical value for Tukey's HSD test."""
    # This is a simplified version - in practice you'd use a proper q-distribution table
    return 3.5  # Example value, replace with actual calculation

def _compute_hsd(
    means: np.ndarray,
    counts: np.ndarray,
    n_groups: int,
    alpha: float = 0.05
) -> Dict[str, Union[float, np.ndarray]]:
    """Compute Tukey's HSD test statistics."""
    total_n = np.sum(counts)
    df_error = total_n - n_groups

    q_critical = _compute_q_critical(n_groups, df_error)
    se = np.sqrt(np.sum(counts) / (counts * total_n))

    hsd = q_critical * se
    pairwise_diffs = np.abs(means[:, None] - means[None, :])

    return {
        "hsd": hsd,
        "pairwise_diffs": pairwise_diffs,
        "q_critical": q_critical
    }

def tukey_test_fit(
    y: np.ndarray,
    groups: np.ndarray,
    alpha: float = 0.05,
    normalization: Optional[str] = None,
    metric: str = "mse",
    custom_metric: Optional[Callable] = None
) -> Dict[str, Union[Dict, np.ndarray]]:
    """
    Perform Tukey's Honestly Significant Difference (HSD) test.

    Parameters:
    -----------
    y : np.ndarray
        Response variable values.
    groups : np.ndarray
        Group labels for each observation in y.
    alpha : float, optional
        Significance level (default: 0.05).
    normalization : str or None, optional
        Normalization method (not implemented in this example).
    metric : str, optional
        Metric to use for evaluation (not implemented in this example).
    custom_metric : callable or None, optional
        Custom metric function (not implemented in this example).

    Returns:
    --------
    dict
        Dictionary containing test results, metrics, parameters used, and warnings.
    """
    _validate_inputs(y, groups, alpha)

    group_stats = _compute_group_means(y, groups)
    hsd_results = _compute_hsd(
        group_stats["means"],
        group_stats["counts"],
        group_stats["n_groups"],
        alpha
    )

    # Determine which pairwise comparisons are significant
    is_significant = hsd_results["pairwise_diffs"] > hsd_results["hsd"]

    return {
        "result": {
            "means": group_stats["means"],
            "pairwise_differences": hsd_results["pairwise_diffs"],
            "significant_comparisons": is_significant,
            "hsd_value": hsd_results["hsd"],
            "q_critical": hsd_results["q_critical"]
        },
        "metrics": {},
        "params_used": {
            "alpha": alpha,
            "normalization": normalization,
            "metric": metric
        },
        "warnings": []
    }

# Example usage:
"""
y = np.array([1.2, 2.3, 3.4, 4.5, 5.6, 1.1, 2.2, 3.3])
groups = np.array([0, 0, 0, 1, 1, 2, 2, 2])
result = tukey_test_fit(y, groups)
"""

################################################################################
# bonferroni_correction
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(p_values: np.ndarray) -> None:
    """Validate input p-values array."""
    if not isinstance(p_values, np.ndarray):
        raise TypeError("p_values must be a numpy array")
    if p_values.ndim != 1:
        raise ValueError("p_values must be a 1-dimensional array")
    if np.any(p_values < 0) or np.any(p_values > 1):
        raise ValueError("All p-values must be between 0 and 1")
    if np.isnan(p_values).any():
        raise ValueError("p_values contains NaN values")

def _compute_bonferroni_threshold(p_values: np.ndarray, alpha: float = 0.05) -> float:
    """Compute the Bonferroni corrected threshold."""
    return alpha / len(p_values)

def _apply_bonferroni_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
    method: str = 'simple'
) -> Dict[str, Union[np.ndarray, float]]:
    """Apply Bonferroni correction to p-values."""
    if method == 'simple':
        threshold = _compute_bonferroni_threshold(p_values, alpha)
        corrected_p_values = np.minimum(p_values * len(p_values), 1.0)
    else:
        raise ValueError(f"Unknown method: {method}")
    return {
        'corrected_p_values': corrected_p_values,
        'threshold': threshold
    }

def bonferroni_correction_fit(
    p_values: np.ndarray,
    alpha: float = 0.05,
    method: str = 'simple',
    custom_method: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, float, str]]:
    """
    Apply Bonferroni correction to p-values.

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values to correct.
    alpha : float, optional
        Significance level (default is 0.05).
    method : str, optional
        Correction method ('simple' or custom callable).
    custom_method : Callable, optional
        Custom correction function if method is 'custom'.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, str]]
        Dictionary containing:
        - 'result': corrected p-values
        - 'threshold': significance threshold
        - 'params_used': parameters used
        - 'warnings': any warnings

    Examples
    --------
    >>> p_values = np.array([0.1, 0.2, 0.3])
    >>> result = bonferroni_correction_fit(p_values)
    """
    _validate_inputs(p_values)

    if custom_method is not None:
        result = custom_method(p_values, alpha)
    else:
        result = _apply_bonferroni_correction(p_values, alpha, method)

    return {
        'result': result['corrected_p_values'],
        'threshold': result['threshold'],
        'params_used': {
            'alpha': alpha,
            'method': method
        },
        'warnings': []
    }

################################################################################
# scheffe_test
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray) -> None:
    """Validate input data for Scheffé test."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input X must be a 2-dimensional array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input X contains NaN or infinite values")

def _compute_scheffe_statistic(X: np.ndarray, group_indices: np.ndarray) -> float:
    """Compute the Scheffé test statistic."""
    n_groups = len(np.unique(group_indices))
    group_means = np.array([np.mean(X[group_indices == g], axis=0) for g in range(n_groups)])
    overall_mean = np.mean(X, axis=0)
    SST = np.sum((X - overall_mean) ** 2)
    SSW = np.sum([np.sum((X[group_indices == g] - group_means[g]) ** 2) for g in range(n_groups)])
    SSB = SST - SSW
    p = X.shape[1]
    statistic = (n_groups - 1) * SSB / (p * SSW)
    return statistic

def _compute_p_value(statistic: float, df1: int, df2: int) -> float:
    """Compute the p-value for the Scheffé test statistic."""
    from scipy.stats import f
    return 1 - f.cdf(statistic, df1, df2)

def scheffe_test_fit(
    X: np.ndarray,
    group_indices: np.ndarray,
    normalization: Optional[str] = None,
    metric: str = "mse",
    alpha: float = 0.05,
    custom_metric: Optional[Callable] = None
) -> Dict[str, Union[float, Dict]]:
    """
    Perform Scheffé's post-hoc test for multiple comparisons.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    group_indices : np.ndarray
        Array of group indices for each sample
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str, optional
        Metric to use ('mse', 'mae', 'r2')
    alpha : float, optional
        Significance level (default: 0.05)
    custom_metric : callable, optional
        Custom metric function

    Returns:
    --------
    dict
        Dictionary containing test results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.randn(100, 5)
    >>> groups = np.random.randint(0, 3, size=100)
    >>> result = scheffe_test_fit(X, groups)
    """
    _validate_inputs(X)

    # Normalization
    if normalization == "standard":
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalization == "minmax":
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalization == "robust":
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))

    # Compute test statistic
    statistic = _compute_scheffe_statistic(X, group_indices)
    n_groups = len(np.unique(group_indices))
    df1 = n_groups - 1
    df2 = X.shape[0] - n_groups

    # Compute p-value
    p_value = _compute_p_value(statistic, df1, df2)

    # Compute metrics
    if metric == "mse":
        mse = np.mean((X - np.mean(X, axis=0)) ** 2)
    elif metric == "mae":
        mse = np.mean(np.abs(X - np.mean(X, axis=0)))
    elif metric == "r2":
        mse = 1 - np.var(X) / np.mean((X - np.mean(X, axis=0)) ** 2)
    else:
        if custom_metric is None:
            raise ValueError("Custom metric function must be provided")
        mse = custom_metric(X)

    # Prepare results
    result = {
        "statistic": statistic,
        "p_value": p_value,
        "is_significant": p_value < alpha
    }

    metrics = {
        metric: mse
    }

    params_used = {
        "normalization": normalization,
        "metric": metric,
        "alpha": alpha
    }

    warnings = []

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

################################################################################
# dunnett_test
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    control_group: np.ndarray,
    treatment_groups: Dict[str, np.ndarray],
    alpha: float = 0.05,
) -> None:
    """Validate input data for Dunnett's test."""
    if not isinstance(control_group, np.ndarray) or control_group.ndim != 1:
        raise ValueError("Control group must be a 1D numpy array.")
    if not isinstance(treatment_groups, dict):
        raise ValueError("Treatment groups must be provided as a dictionary.")
    for name, group in treatment_groups.items():
        if not isinstance(group, np.ndarray) or group.ndim != 1:
            raise ValueError(f"Treatment group {name} must be a 1D numpy array.")
        if len(control_group) != len(group):
            raise ValueError("All groups must have the same number of samples.")
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")

def _calculate_statistics(
    control_group: np.ndarray,
    treatment_groups: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Calculate test statistics for Dunnett's test."""
    stats = {}
    mean_control = np.mean(control_group)
    for name, group in treatment_groups.items():
        mean_treatment = np.mean(group)
        std_control = np.std(control_group, ddof=1)
        std_treatment = np.std(group, ddof=1)
        n_control = len(control_group)
        n_treatment = len(group)

        # Calculate pooled standard deviation
        pooled_std = np.sqrt(
            ((n_control - 1) * std_control**2 + (n_treatment - 1) * std_treatment**2)
            / (n_control + n_treatment - 2)
        )

        # Calculate Dunnett's test statistic
        stat = (mean_treatment - mean_control) / pooled_std
        stats[name] = stat

    return stats

def _calculate_p_values(
    test_statistics: Dict[str, float],
    alpha: float,
) -> Dict[str, float]:
    """Calculate p-values for Dunnett's test using critical values."""
    # This is a simplified version; in practice, you would use a more accurate method
    critical_values = {
        'two_sided': 2.40,  # Example value for k=3 groups
    }
    p_values = {}
    for name, stat in test_statistics.items():
        if abs(stat) > critical_values['two_sided']:
            p_values[name] = 2 * (1 - alpha)
        else:
            p_values[name] = 1.0
    return p_values

def dunnett_test_fit(
    control_group: np.ndarray,
    treatment_groups: Dict[str, np.ndarray],
    alpha: float = 0.05,
    normalization: Optional[Callable] = None,
) -> Dict[str, Union[Dict[str, float], Dict[str, str]]]:
    """
    Perform Dunnett's test to compare multiple treatment groups against a control group.

    Parameters:
    -----------
    control_group : np.ndarray
        The control group data.
    treatment_groups : Dict[str, np.ndarray]
        Dictionary of treatment groups where keys are group names and values are data arrays.
    alpha : float, optional
        Significance level (default is 0.05).
    normalization : Callable, optional
        Function to normalize the data (default is None).

    Returns:
    --------
    Dict[str, Union[Dict[str, float], Dict[str, str]]]
        Dictionary containing:
        - "result": Dictionary of p-values for each treatment group.
        - "metrics": Dictionary of test statistics for each treatment group.
        - "params_used": Dictionary of parameters used in the test.
        - "warnings": List of any warnings generated during the test.

    Example:
    --------
    control = np.array([1.2, 1.5, 1.3, 1.4])
    treatments = {
        'A': np.array([1.6, 1.7, 1.8, 1.9]),
        'B': np.array([1.3, 1.4, 1.5, 1.6]),
    }
    result = dunnett_test_fit(control, treatments)
    """
    warnings = []

    # Validate inputs
    _validate_inputs(control_group, treatment_groups, alpha)

    # Normalize data if specified
    if normalization is not None:
        control_group = normalization(control_group)
        treatment_groups = {name: normalization(group) for name, group in treatment_groups.items()}

    # Calculate test statistics
    test_statistics = _calculate_statistics(control_group, treatment_groups)

    # Calculate p-values
    p_values = _calculate_p_values(test_statistics, alpha)

    return {
        "result": p_values,
        "metrics": test_statistics,
        "params_used": {
            "alpha": alpha,
            "normalization": normalization.__name__ if normalization else None,
        },
        "warnings": warnings,
    }

################################################################################
# holm_bonferroni_method
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(p_values: np.ndarray) -> None:
    """Validate input p-values array."""
    if not isinstance(p_values, np.ndarray):
        raise TypeError("p_values must be a numpy array")
    if len(p_values.shape) != 1:
        raise ValueError("p_values must be a 1-dimensional array")
    if np.any(p_values < 0) or np.any(p_values > 1):
        raise ValueError("p-values must be between 0 and 1")
    if np.isnan(p_values).any():
        raise ValueError("p-values must not contain NaN values")

def _holm_bonferroni_correction(p_values: np.ndarray, alpha: float = 0.05) -> Dict[str, Union[np.ndarray, float]]:
    """Perform Holm-Bonferroni correction on p-values."""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)[::-1]  # Sort in descending order
    sorted_p_values = p_values[sorted_indices]
    adjusted_p_values = np.zeros_like(p_values)

    for i, p_val in enumerate(sorted_p_values):
        adjusted_p = (i + 1) * p_val / n
        adjusted_p_values[sorted_indices[i]] = min(adjusted_p, 1.0)

    rejected = adjusted_p_values <= alpha
    return {
        "adjusted_p_values": adjusted_p_values,
        "rejected": rejected,
        "alpha": alpha
    }

def holm_bonferroni_method_fit(
    p_values: np.ndarray,
    alpha: float = 0.05,
    custom_correction: Optional[Callable[[np.ndarray, float], Dict[str, Union[np.ndarray, float]]]] = None
) -> Dict[str, Union[np.ndarray, float, dict]]:
    """
    Perform Holm-Bonferroni method for multiple hypothesis testing correction.

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values from multiple hypothesis tests.
    alpha : float, optional
        Significance level (default is 0.05).
    custom_correction : Callable, optional
        Custom correction function (must follow the same signature as _holm_bonferroni_correction).

    Returns
    -------
    dict
        Dictionary containing:
        - "result": Results of the correction (adjusted p-values and rejection status)
        - "metrics": Metrics related to the correction
        - "params_used": Parameters used in the correction
        - "warnings": Any warnings generated during the process

    Example
    -------
    >>> p_values = np.array([0.01, 0.02, 0.03, 0.04])
    >>> result = holm_bonferroni_method_fit(p_values)
    """
    _validate_inputs(p_values)

    if custom_correction is not None:
        correction_result = custom_correction(p_values, alpha)
    else:
        correction_result = _holm_bonferroni_correction(p_values, alpha)

    metrics = {
        "num_tests": len(p_values),
        "num_rejected": np.sum(correction_result["rejected"]),
    }

    return {
        "result": correction_result,
        "metrics": metrics,
        "params_used": {"alpha": alpha},
        "warnings": []
    }

################################################################################
# hochberg_method
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(p_values: np.ndarray) -> None:
    """Validate input p-values array."""
    if not isinstance(p_values, np.ndarray):
        raise TypeError("p_values must be a numpy array")
    if p_values.ndim != 1:
        raise ValueError("p_values must be a 1-dimensional array")
    if np.any(p_values < 0) or np.any(p_values > 1):
        raise ValueError("p-values must be between 0 and 1")
    if np.isnan(p_values).any():
        raise ValueError("p-values must not contain NaN values")

def _compute_critical_values(p_values: np.ndarray, method: str = 'bonferroni') -> np.ndarray:
    """Compute critical values based on the chosen method."""
    n = len(p_values)
    if method == 'bonferroni':
        return 1.0 / n
    elif method == 'holm':
        return np.arange(1, n + 1)[::-1] / n
    elif method == 'hochberg':
        return np.arange(1, n + 1) / n
    else:
        raise ValueError(f"Unknown method: {method}")

def _apply_correction(p_values: np.ndarray, critical_values: np.ndarray) -> np.ndarray:
    """Apply the correction to p-values."""
    corrected_p_values = np.zeros_like(p_values)
    sorted_indices = np.argsort(p_values)[::-1]
    for i, idx in enumerate(sorted_indices):
        if p_values[idx] > critical_values[i]:
            corrected_p_values[idx] = 1.0
        else:
            corrected_p_values[idx] = p_values[idx]
    return corrected_p_values

def hochberg_method_fit(
    p_values: np.ndarray,
    method: str = 'hochberg',
    alpha: float = 0.05,
    custom_critical_values: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Union[np.ndarray, Dict, str]]:
    """
    Apply the Hochberg method for multiple hypothesis testing correction.

    Parameters:
    -----------
    p_values : np.ndarray
        Array of p-values to correct.
    method : str, optional
        Correction method ('bonferroni', 'holm', 'hochberg'). Default is 'hochberg'.
    alpha : float, optional
        Significance level. Default is 0.05.
    custom_critical_values : Callable, optional
        Custom function to compute critical values.

    Returns:
    --------
    dict
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    _validate_inputs(p_values)

    if custom_critical_values is not None:
        critical_values = custom_critical_values(p_values)
    else:
        critical_values = _compute_critical_values(p_values, method)

    corrected_p_values = _apply_correction(p_values, critical_values)

    # Determine which hypotheses are rejected
    rejected = corrected_p_values < alpha

    return {
        "result": {
            "corrected_p_values": corrected_p_values,
            "rejected": rejected
        },
        "metrics": {
            "num_rejected": np.sum(rejected),
            "alpha_level": alpha
        },
        "params_used": {
            "method": method,
            "alpha": alpha,
            "custom_critical_values": custom_critical_values is not None
        },
        "warnings": []
    }

# Example usage:
# p_values = np.array([0.01, 0.02, 0.03, 0.04])
# result = hochberg_method_fit(p_values)

################################################################################
# hommel_method
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def hommel_method_fit(
    p_values: np.ndarray,
    alpha: float = 0.05,
    correction_method: str = 'hommel',
    normalization: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray], float]] = None
) -> Dict[str, Union[Dict[str, float], np.ndarray, str]]:
    """
    Applique la méthode de correction post-hoc de Hommel aux p-values.

    Parameters:
    -----------
    p_values : np.ndarray
        Tableau des p-values à corriger.
    alpha : float, optional
        Niveau de significativité (par défaut 0.05).
    correction_method : str, optional
        Méthode de correction à utiliser (par défaut 'hommel').
    normalization : str, optional
        Normalisation à appliquer aux p-values (par défaut None).
    custom_metric : Callable, optional
        Fonction personnalisée pour calculer une métrique (par défaut None).

    Returns:
    --------
    Dict[str, Union[Dict[str, float], np.ndarray, str]]
        Dictionnaire contenant les résultats, métriques et paramètres utilisés.
    """
    # Validation des entrées
    _validate_inputs(p_values, alpha)

    # Normalisation si nécessaire
    if normalization:
        p_values = _apply_normalization(p_values, normalization)

    # Calcul des p-values corrigées
    corrected_p_values = _compute_hommel_correction(p_values, alpha)

    # Calcul des métriques
    metrics = {}
    if custom_metric:
        metrics['custom'] = custom_metric(corrected_p_values)
    else:
        metrics.update({
            'mean_corrected_p': np.mean(corrected_p_values),
            'max_corrected_p': np.max(corrected_p_values)
        })

    # Retourne le résultat structuré
    return {
        'result': {'corrected_p_values': corrected_p_values},
        'metrics': metrics,
        'params_used': {
            'alpha': alpha,
            'correction_method': correction_method,
            'normalization': normalization
        },
        'warnings': _check_warnings(p_values, corrected_p_values)
    }

def _validate_inputs(
    p_values: np.ndarray,
    alpha: float
) -> None:
    """
    Valide les entrées pour la méthode de Hommel.

    Parameters:
    -----------
    p_values : np.ndarray
        Tableau des p-values à corriger.
    alpha : float
        Niveau de significativité.

    Raises:
    -------
    ValueError
        Si les entrées sont invalides.
    """
    if not isinstance(p_values, np.ndarray):
        raise ValueError("p_values doit être un tableau NumPy.")
    if not (0 <= alpha <= 1):
        raise ValueError("alpha doit être entre 0 et 1.")
    if np.any(p_values < 0) or np.any(p_values > 1):
        raise ValueError("Les p-values doivent être entre 0 et 1.")
    if np.isnan(p_values).any():
        raise ValueError("Les p-values ne doivent pas contenir de NaN.")

def _apply_normalization(
    p_values: np.ndarray,
    method: str
) -> np.ndarray:
    """
    Applique une normalisation aux p-values.

    Parameters:
    -----------
    p_values : np.ndarray
        Tableau des p-values à normaliser.
    method : str
        Méthode de normalisation à appliquer.

    Returns:
    --------
    np.ndarray
        Tableau des p-values normalisées.
    """
    if method == 'standard':
        return (p_values - np.mean(p_values)) / np.std(p_values)
    elif method == 'minmax':
        return (p_values - np.min(p_values)) / (np.max(p_values) - np.min(p_values))
    elif method == 'robust':
        median = np.median(p_values)
        mad = np.median(np.abs(p_values - median))
        return (p_values - median) / mad
    else:
        raise ValueError(f"Méthode de normalisation inconnue: {method}")

def _compute_hommel_correction(
    p_values: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Calcule les p-values corrigées selon la méthode de Hommel.

    Parameters:
    -----------
    p_values : np.ndarray
        Tableau des p-values à corriger.
    alpha : float
        Niveau de significativité.

    Returns:
    --------
    np.ndarray
        Tableau des p-values corrigées.
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)[::-1]
    p_sorted = np.sort(p_values)[::-1]

    corrected_p_values = np.zeros_like(p_values)
    for k in range(1, n + 1):
        subset = p_sorted[:k]
        if np.max(subset) <= (alpha * k / n):
            corrected_p_values[sorted_indices[:k]] = subset
        else:
            break

    return corrected_p_values

def _check_warnings(
    p_values: np.ndarray,
    corrected_p_values: np.ndarray
) -> Optional[str]:
    """
    Vérifie les avertissements potentiels.

    Parameters:
    -----------
    p_values : np.ndarray
        Tableau des p-values originales.
    corrected_p_values : np.ndarray
        Tableau des p-values corrigées.

    Returns:
    --------
    Optional[str]
        Avertissement si nécessaire, sinon None.
    """
    if np.any(corrected_p_values > 1):
        return "Certaines p-values corrigées sont supérieures à 1."
    if np.any(p_values == corrected_p_values):
        return "Certaines p-values n'ont pas été corrigées."
    return None

################################################################################
# sidak_correction
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(p_values: np.ndarray) -> None:
    """Validate input p-values array."""
    if not isinstance(p_values, np.ndarray):
        raise TypeError("p_values must be a numpy array")
    if p_values.ndim != 1:
        raise ValueError("p_values must be a 1-dimensional array")
    if np.any(p_values < 0) or np.any(p_values > 1):
        raise ValueError("All p-values must be between 0 and 1")
    if np.isnan(p_values).any():
        raise ValueError("p_values contains NaN values")

def _compute_sidak_correction(p_values: np.ndarray) -> np.ndarray:
    """Compute Sidak correction for given p-values."""
    n = len(p_values)
    corrected_p_values = 1 - (1 - p_values) ** n
    return corrected_p_values

def sidak_correction_fit(
    p_values: np.ndarray,
    method: str = "sidak",
    alpha: float = 0.05
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Apply Sidak correction to p-values.

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values to correct.
    method : str, optional
        Correction method (currently only "sidak" is supported).
    alpha : float, optional
        Significance level.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict]]
        Dictionary containing:
        - "result": corrected p-values
        - "metrics": dictionary with metrics information
        - "params_used": dictionary of parameters used
        - "warnings": list of warnings

    Examples
    --------
    >>> p_values = np.array([0.1, 0.2, 0.3])
    >>> result = sidak_correction_fit(p_values)
    """
    _validate_inputs(p_values)

    corrected_p_values = _compute_sidak_correction(p_values)

    metrics = {
        "method": method,
        "alpha": alpha
    }

    params_used = {
        "original_p_values_shape": p_values.shape,
        "correction_method": method
    }

    warnings = []

    return {
        "result": corrected_p_values,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

################################################################################
# false_discovery_rate
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def false_discovery_rate_fit(
    p_values: np.ndarray,
    method: str = 'benjamini_hochberg',
    alpha: float = 0.05,
    custom_method: Optional[Callable[[np.ndarray, float], np.ndarray]] = None
) -> Dict[str, Union[np.ndarray, Dict, str]]:
    """
    Compute false discovery rate corrected p-values.

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values to correct.
    method : str, optional
        Method for FDR correction. Options: 'benjamini_hochberg', 'by', 'storey'.
    alpha : float, optional
        Significance level.
    custom_method : Callable, optional
        Custom FDR correction method.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict, str]]
        Dictionary containing:
        - "result": Corrected p-values
        - "metrics": Metrics about the correction
        - "params_used": Parameters used
        - "warnings": Any warnings

    Example
    -------
    >>> p_values = np.array([0.1, 0.2, 0.3, 0.4])
    >>> result = false_discovery_rate_fit(p_values)
    """
    # Validate inputs
    _validate_inputs(p_values, alpha)

    params_used = {
        'method': method,
        'alpha': alpha
    }

    warnings = []

    # Choose the appropriate FDR correction method
    if custom_method is not None:
        corrected_p_values = custom_method(p_values, alpha)
    elif method == 'benjamini_hochberg':
        corrected_p_values = _benjamini_hochberg(p_values, alpha)
    elif method == 'by':
        corrected_p_values = _by(p_values, alpha)
    elif method == 'storey':
        corrected_p_values = _storey(p_values, alpha)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Calculate metrics
    metrics = _calculate_metrics(p_values, corrected_p_values)

    return {
        'result': corrected_p_values,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

def _validate_inputs(p_values: np.ndarray, alpha: float) -> None:
    """Validate input parameters."""
    if not isinstance(p_values, np.ndarray):
        raise TypeError("p_values must be a numpy array")
    if not np.all(p_values >= 0) or not np.all(p_values <= 1):
        raise ValueError("p_values must be between 0 and 1")
    if not np.isfinite(p_values).all():
        raise ValueError("p_values must be finite")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

def _benjamini_hochberg(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    p_sorted = p_values[sorted_indices]
    corrected_p_values = np.zeros_like(p_values)

    for i in range(n):
        j = n - i - 1
        p_value = p_sorted[j]
        corrected_p_values[sorted_indices[j]] = (n * p_value) / (j + 1)

    corrected_p_values = np.minimum(corrected_p_values, alpha)
    return corrected_p_values

def _by(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Yekutieli FDR correction."""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    p_sorted = p_values[sorted_indices]
    corrected_p_values = np.zeros_like(p_values)

    for i in range(n):
        j = n - i - 1
        p_value = p_sorted[j]
        corrected_p_values[sorted_indices[j]] = (p_value * n) / np.sum(1 / np.arange(1, j + 2))

    corrected_p_values = np.minimum(corrected_p_values, alpha)
    return corrected_p_values

def _storey(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """Storey's FDR correction."""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    p_sorted = p_values[sorted_indices]
    corrected_p_values = np.zeros_like(p_values)

    for i in range(n):
        j = n - i - 1
        p_value = p_sorted[j]
        corrected_p_values[sorted_indices[j]] = (p_value * n) / np.sum(p_sorted <= p_value)

    corrected_p_values = np.minimum(corrected_p_values, alpha)
    return corrected_p_values

def _calculate_metrics(p_values: np.ndarray, corrected_p_values: np.ndarray) -> Dict[str, float]:
    """Calculate metrics for FDR correction."""
    original_significant = np.sum(p_values < 0.05)
    corrected_significant = np.sum(corrected_p_values < 0.05)

    return {
        'original_significant': original_significant,
        'corrected_significant': corrected_significant,
        'significant_reduction': original_significant - corrected_significant
    }

################################################################################
# newman_keuls_test
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def newman_keuls_test_fit(
    data: np.ndarray,
    groups: np.ndarray,
    alpha: float = 0.05,
    normalization: Optional[str] = None,
    metric: str = 'mean',
    pairwise_comparison_func: Optional[Callable] = None,
    p_value_correction: str = 'bonferroni'
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Perform Newman-Keuls post-hoc test for multiple comparisons.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_samples, n_features) containing the data.
    groups : np.ndarray
        Array of shape (n_samples,) containing group labels.
    alpha : float, optional
        Significance level, by default 0.05.
    normalization : Optional[str], optional
        Normalization method, by default None. Options: 'standard', 'minmax'.
    metric : str, optional
        Metric to compare groups, by default 'mean'. Options: 'median', 'mean'.
    pairwise_comparison_func : Optional[Callable], optional
        Custom function for pairwise comparisons, by default None.
    p_value_correction : str, optional
        Method for correcting p-values, by default 'bonferroni'. Options: 'holm', 'fdr_bh'.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.random.rand(100, 3)
    >>> groups = np.random.randint(0, 5, size=100)
    >>> result = newman_keuls_test_fit(data, groups)
    """
    # Validate inputs
    _validate_inputs(data, groups)

    # Normalize data if required
    normalized_data = _apply_normalization(data, normalization)

    # Compute group statistics
    group_stats = _compute_group_statistics(normalized_data, groups, metric)

    # Perform pairwise comparisons
    if pairwise_comparison_func is None:
        pairwise_results = _perform_pairwise_comparisons(group_stats, alpha, p_value_correction)
    else:
        pairwise_results = _perform_custom_pairwise_comparisons(group_stats, alpha, p_value_correction, pairwise_comparison_func)

    # Prepare output
    result = {
        "result": pairwise_results,
        "metrics": {"alpha": alpha, "p_value_correction": p_value_correction},
        "params_used": {"normalization": normalization, "metric": metric},
        "warnings": []
    }

    return result

def _validate_inputs(data: np.ndarray, groups: np.ndarray) -> None:
    """Validate input data and groups."""
    if not isinstance(data, np.ndarray) or not isinstance(groups, np.ndarray):
        raise TypeError("Data and groups must be numpy arrays.")
    if data.shape[0] != groups.shape[0]:
        raise ValueError("Data and groups must have the same number of samples.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values.")
    if np.any(np.isnan(groups)) or np.any(np.isinf(groups)):
        raise ValueError("Groups contain NaN or infinite values.")

def _apply_normalization(data: np.ndarray, normalization: Optional[str]) -> np.ndarray:
    """Apply normalization to the data."""
    if normalization is None:
        return data
    elif normalization == 'standard':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)
    elif normalization == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / ((max_val - min_val) + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _compute_group_statistics(data: np.ndarray, groups: np.ndarray, metric: str) -> Dict:
    """Compute statistics for each group."""
    unique_groups = np.unique(groups)
    group_stats = {}
    for group in unique_groups:
        group_data = data[groups == group]
        if metric == 'mean':
            stat = np.mean(group_data, axis=0)
        elif metric == 'median':
            stat = np.median(group_data, axis=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        group_stats[group] = stat
    return group_stats

def _perform_pairwise_comparisons(group_stats: Dict, alpha: float, p_value_correction: str) -> np.ndarray:
    """Perform pairwise comparisons between groups."""
    groups = list(group_stats.keys())
    n_groups = len(groups)
    p_values = np.zeros((n_groups, n_groups))
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            # Placeholder for actual statistical test
            p_values[i, j] = 0.1  # Example p-value
    return _correct_p_values(p_values, alpha, p_value_correction)

def _perform_custom_pairwise_comparisons(group_stats: Dict, alpha: float, p_value_correction: str,
                                        pairwise_comparison_func: Callable) -> np.ndarray:
    """Perform custom pairwise comparisons between groups."""
    groups = list(group_stats.keys())
    n_groups = len(groups)
    p_values = np.zeros((n_groups, n_groups))
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            p_values[i, j] = pairwise_comparison_func(group_stats[groups[i]], group_stats[groups[j]])
    return _correct_p_values(p_values, alpha, p_value_correction)

def _correct_p_values(p_values: np.ndarray, alpha: float, method: str) -> np.ndarray:
    """Correct p-values for multiple comparisons."""
    if method == 'bonferroni':
        return np.minimum(p_values * len(p_values), alpha)
    elif method == 'holm':
        # Placeholder for Holm correction
        return np.minimum(p_values * (1 / np.arange(1, len(p_values) + 1)), alpha)
    elif method == 'fdr_bh':
        # Placeholder for Benjamini-Hochberg correction
        return np.minimum(p_values * (1 / np.arange(1, len(p_values) + 1)), alpha)
    else:
        raise ValueError(f"Unknown p-value correction method: {method}")
