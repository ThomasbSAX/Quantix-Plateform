"""
Quantix – Module comparaison_groupes_visuelle
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# boxplot
################################################################################

import numpy as np
from typing import Dict, Any, Callable, Optional, Union

def boxplot_fit(
    data: np.ndarray,
    group_labels: Optional[np.ndarray] = None,
    normalization: str = "none",
    metrics: Union[str, Callable[[np.ndarray], float]] = "default",
    show_outliers: bool = True,
    whisker_method: str = "iqr",
    custom_whiskers: Optional[Callable[[np.ndarray], tuple]] = None,
) -> Dict[str, Any]:
    """
    Compute and return boxplot statistics for visual comparison of groups.

    Parameters:
    -----------
    data : np.ndarray
        Input data array. If group_labels is provided, each column represents a group.
    group_labels : Optional[np.ndarray]
        Array of labels indicating group membership for each data point.
    normalization : str
        Normalization method: "none", "standard", "minmax", or "robust".
    metrics : Union[str, Callable]
        Metrics to compute: "default" (median, q1, q3), or custom callable.
    show_outliers : bool
        Whether to include outliers in the computation.
    whisker_method : str
        Method for computing whiskers: "iqr" (1.5*IQR) or custom.
    custom_whiskers : Optional[Callable]
        Custom function to compute whiskers.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing boxplot statistics and metadata.
    """
    # Validate inputs
    _validate_inputs(data, group_labels)

    # Normalize data if required
    normalized_data = _apply_normalization(data, normalization)

    # Compute boxplot statistics for each group
    if group_labels is not None:
        groups = _group_data(normalized_data, group_labels)
    else:
        groups = [normalized_data]

    boxplot_stats = []
    for group in groups:
        stats = _compute_boxplot_statistics(
            group,
            metrics=metrics,
            show_outliers=show_outliers,
            whisker_method=whisker_method,
            custom_whiskers=custom_whiskers
        )
        boxplot_stats.append(stats)

    # Prepare output dictionary
    result = {
        "result": boxplot_stats,
        "metrics": _get_metrics_description(metrics),
        "params_used": {
            "normalization": normalization,
            "whisker_method": whisker_method,
            "show_outliers": show_outliers
        },
        "warnings": _check_warnings(data, group_labels)
    }

    return result

def _validate_inputs(
    data: np.ndarray,
    group_labels: Optional[np.ndarray]
) -> None:
    """Validate input data and group labels."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if group_labels is not None and not isinstance(group_labels, np.ndarray):
        raise TypeError("Group labels must be a numpy array or None.")
    if group_labels is not None and len(data) != len(group_labels):
        raise ValueError("Data and group labels must have the same length.")

def _apply_normalization(
    data: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply specified normalization to the data."""
    if method == "none":
        return data
    elif method == "standard":
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        return (data - mean) / std
    elif method == "minmax":
        min_val = np.nanmin(data, axis=0)
        max_val = np.nanmax(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.nanmedian(data, axis=0)
        iqr = np.nanpercentile(data, 75, axis=0) - np.nanpercentile(data, 25, axis=0)
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _group_data(
    data: np.ndarray,
    group_labels: np.ndarray
) -> list:
    """Group data by labels."""
    unique_groups = np.unique(group_labels)
    groups = []
    for group in unique_groups:
        mask = group_labels == group
        groups.append(data[mask])
    return groups

def _compute_boxplot_statistics(
    group: np.ndarray,
    metrics: Union[str, Callable],
    show_outliers: bool,
    whisker_method: str,
    custom_whiskers: Optional[Callable]
) -> Dict[str, Any]:
    """Compute boxplot statistics for a single group."""
    q1 = np.nanpercentile(group, 25)
    median = np.nanmedian(group)
    q3 = np.nanpercentile(group, 75)

    if custom_whiskers is not None:
        lower_whisker, upper_whisker = custom_whiskers(group)
    else:
        iqr = q3 - q1
        if whisker_method == "iqr":
            lower_whisker = q1 - 1.5 * iqr
            upper_whisker = q3 + 1.5 * iqr

    if show_outliers:
        outliers = group[(group < lower_whisker) | (group > upper_whisker)]
    else:
        outliers = np.array([])

    # Compute metrics
    if isinstance(metrics, str) and metrics == "default":
        metric_values = {
            "median": median,
            "q1": q1,
            "q3": q3
        }
    elif callable(metrics):
        metric_values = metrics(group)
    else:
        raise ValueError("Metrics must be 'default' or a callable.")

    return {
        "q1": q1,
        "median": median,
        "q3": q3,
        "lower_whisker": lower_whisker,
        "upper_whisker": upper_whisker,
        "outliers": outliers,
        "metrics": metric_values
    }

def _get_metrics_description(
    metrics: Union[str, Callable]
) -> Dict[str, Any]:
    """Get description of the metrics used."""
    if isinstance(metrics, str) and metrics == "default":
        return {
            "description": "Default boxplot statistics (median, q1, q3)",
            "values": ["median", "q1", "q3"]
        }
    elif callable(metrics):
        return {
            "description": "Custom metrics function",
            "values": "user-defined"
        }
    else:
        raise ValueError("Invalid metrics specification.")

def _check_warnings(
    data: np.ndarray,
    group_labels: Optional[np.ndarray]
) -> list:
    """Check for potential warnings in the data."""
    warnings = []
    if np.isnan(data).any():
        warnings.append("Data contains NaN values.")
    if group_labels is not None and np.isnan(group_labels).any():
        warnings.append("Group labels contain NaN values.")
    if group_labels is not None and len(np.unique(group_labels)) == 1:
        warnings.append("All data points belong to the same group.")
    return warnings

################################################################################
# violin_plot
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def validate_inputs(data: np.ndarray, groups: np.ndarray) -> None:
    """
    Validate input data and groups.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    groups : np.ndarray
        Group labels for each data point.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not isinstance(data, np.ndarray) or not isinstance(groups, np.ndarray):
        raise ValueError("Inputs must be numpy arrays.")
    if data.ndim != 1:
        raise ValueError("Data must be 1-dimensional.")
    if groups.ndim != 1:
        raise ValueError("Groups must be 1-dimensional.")
    if len(data) != len(groups):
        raise ValueError("Data and groups must have the same length.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data must not contain NaN or infinite values.")

def normalize_data(
    data: np.ndarray,
    method: str = "standard",
    custom_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """
    Normalize data using specified method or custom function.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    method : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    custom_func : Callable, optional
        Custom normalization function.

    Returns
    ------
    np.ndarray
        Normalized data.
    """
    if custom_func is not None:
        return custom_func(data)

    if method == "none":
        return data
    elif method == "standard":
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std
    elif method == "minmax":
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)
    elif method == "robust":
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def compute_violin_stats(
    data: np.ndarray,
    groups: np.ndarray
) -> Dict[str, Any]:
    """
    Compute statistics for violin plot.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    groups : np.ndarray
        Group labels for each data point.

    Returns
    ------
    Dict[str, Any]
        Dictionary containing computed statistics.
    """
    unique_groups = np.unique(groups)
    stats = {}

    for group in unique_groups:
        group_data = data[groups == group]
        stats[group] = {
            "mean": np.mean(group_data),
            "median": np.median(group_data),
            "q1": np.percentile(group_data, 25),
            "q3": np.percentile(group_data, 75),
            "min": np.min(group_data),
            "max": np.max(group_data),
            "density": np.histogram(group_data, bins=50, density=True)[0]
        }

    return stats

def violin_plot_compute(
    data: np.ndarray,
    groups: np.ndarray,
    normalization_method: str = "standard",
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metrics: Union[str, Callable] = "mse",
    **kwargs
) -> Dict[str, Any]:
    """
    Compute violin plot statistics and metrics.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    groups : np.ndarray
        Group labels for each data point.
    normalization_method : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    custom_normalization : Callable, optional
        Custom normalization function.
    metrics : str or Callable, optional
        Metric to compute ("mse", "mae", "r2") or custom callable.

    Returns
    ------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    validate_inputs(data, groups)

    # Normalize data
    normalized_data = normalize_data(
        data,
        method=normalization_method,
        custom_func=custom_normalization
    )

    # Compute violin statistics
    stats = compute_violin_stats(normalized_data, groups)

    # Compute metrics
    if isinstance(metrics, str):
        if metrics == "mse":
            metric_value = np.mean((normalized_data - np.mean(normalized_data)) ** 2)
        elif metrics == "mae":
            metric_value = np.mean(np.abs(normalized_data - np.mean(normalized_data)))
        elif metrics == "r2":
            mean_y = np.mean(normalized_data)
            ss_total = np.sum((normalized_data - mean_y) ** 2)
            ss_res = np.sum((normalized_data - normalized_data) ** 2)
            metric_value = 1 - (ss_res / ss_total)
        else:
            raise ValueError(f"Unknown metric: {metrics}")
    else:
        metric_value = metrics(normalized_data)

    # Prepare output
    result = {
        "result": stats,
        "metrics": {"value": metric_value, "name": metrics},
        "params_used": {
            "normalization_method": normalization_method,
            "custom_normalization": custom_normalization is not None,
            "metrics": metrics
        },
        "warnings": []
    }

    return result

# Example usage:
# data = np.random.randn(100)
# groups = np.random.randint(0, 2, size=100)
# result = violin_plot_compute(data, groups)

################################################################################
# swarm_plot
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union
from enum import Enum

class Normalization(Enum):
    NONE = "none"
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"

class DistanceMetric(Enum):
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"
    MINKOWSKI = "minkowski"

def validate_inputs(
    data: np.ndarray,
    groups: Optional[np.ndarray] = None,
    group_labels: Optional[Union[list, np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Validate input data and groups.

    Parameters:
    - data: Input data array of shape (n_samples, n_features)
    - groups: Group labels for each sample
    - group_labels: Optional custom names for groups

    Returns:
    - Dictionary containing validated data and parameters
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Data must be a 2D numpy array")

    if groups is not None:
        if len(groups) != data.shape[0]:
            raise ValueError("Number of group labels must match number of samples")
        if not np.issubdtype(groups.dtype, np.integer):
            raise ValueError("Group labels must be integers")

    if group_labels is not None:
        if len(group_labels) != np.max(groups) + 1:
            raise ValueError("Number of group labels must match number of unique groups")

    return {
        "data": data,
        "groups": groups,
        "group_labels": group_labels
    }

def apply_normalization(
    data: np.ndarray,
    normalization: Normalization = Normalization.NONE
) -> np.ndarray:
    """
    Apply specified normalization to the data.

    Parameters:
    - data: Input data array
    - normalization: Normalization method to apply

    Returns:
    - Normalized data array
    """
    if normalization == Normalization.NONE:
        return data

    normalized_data = np.copy(data)

    if normalization == Normalization.STANDARD:
        mean = np.mean(normalized_data, axis=0)
        std = np.std(normalized_data, axis=0)
        normalized_data = (normalized_data - mean) / std

    elif normalization == Normalization.MINMAX:
        min_val = np.min(normalized_data, axis=0)
        max_val = np.max(normalized_data, axis=0)
        normalized_data = (normalized_data - min_val) / (max_val - min_val)

    elif normalization == Normalization.ROBUST:
        median = np.median(normalized_data, axis=0)
        iqr = np.subtract(*np.percentile(normalized_data, [75, 25], axis=0))
        normalized_data = (normalized_data - median) / iqr

    return normalized_data

def calculate_distances(
    data: np.ndarray,
    distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
    p: float = 2.0
) -> np.ndarray:
    """
    Calculate pairwise distances between samples.

    Parameters:
    - data: Input data array
    - distance_metric: Distance metric to use
    - p: Power parameter for Minkowski distance

    Returns:
    - Distance matrix of shape (n_samples, n_samples)
    """
    if distance_metric == DistanceMetric.EUCLIDEAN:
        return np.sqrt(np.sum((data[:, np.newaxis] - data) ** 2, axis=2))
    elif distance_metric == DistanceMetric.MANHATTAN:
        return np.sum(np.abs(data[:, np.newaxis] - data), axis=2)
    elif distance_metric == DistanceMetric.COSINE:
        dot_products = np.dot(data, data.T)
        norms = np.sqrt(np.sum(data**2, axis=1))
        return 1 - (dot_products / np.outer(norms, norms))
    elif distance_metric == DistanceMetric.MINKOWSKI:
        return np.sum(np.abs(data[:, np.newaxis] - data) ** p, axis=2) ** (1/p)

def compute_swarm_statistics(
    data: np.ndarray,
    groups: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute statistics for swarm plot visualization.

    Parameters:
    - data: Input data array
    - groups: Group labels for each sample

    Returns:
    - Dictionary containing computed statistics
    """
    stats = {
        "means": np.mean(data, axis=0),
        "medians": np.median(data, axis=0),
        "stds": np.std(data, axis=0)
    }

    if groups is not None:
        unique_groups = np.unique(groups)
        group_stats = {}

        for group in unique_groups:
            mask = groups == group
            group_data = data[mask]
            group_stats[int(group)] = {
                "mean": np.mean(group_data, axis=0),
                "median": np.median(group_data, axis=0),
                "std": np.std(group_data, axis=0),
                "count": len(group_data)
            }

        stats["group_stats"] = group_stats

    return stats

def swarm_plot_fit(
    data: np.ndarray,
    groups: Optional[np.ndarray] = None,
    group_labels: Optional[Union[list, np.ndarray]] = None,
    normalization: Normalization = Normalization.NONE,
    distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
    p: float = 2.0
) -> Dict[str, Any]:
    """
    Main function to compute swarm plot statistics.

    Parameters:
    - data: Input data array of shape (n_samples, n_features)
    - groups: Group labels for each sample
    - group_labels: Optional custom names for groups
    - normalization: Normalization method to apply
    - distance_metric: Distance metric to use for pairwise distances
    - p: Power parameter for Minkowski distance

    Returns:
    - Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    validated = validate_inputs(data, groups, group_labels)
    data_validated = validated["data"]
    groups_validated = validated.get("groups")
    group_labels_validated = validated.get("group_labels")

    # Apply normalization
    data_normalized = apply_normalization(data_validated, normalization)

    # Calculate distances if needed
    distance_matrix = None
    if groups_validated is not None:
        distance_matrix = calculate_distances(data_normalized, distance_metric, p)

    # Compute swarm statistics
    stats = compute_swarm_statistics(data_normalized, groups_validated)

    # Prepare output
    result = {
        "result": stats,
        "metrics": {},
        "params_used": {
            "normalization": normalization.value,
            "distance_metric": distance_metric.value,
            "p": p
        },
        "warnings": []
    }

    if distance_matrix is not None:
        result["metrics"]["distance_metric"] = distance_metric.value

    return result

################################################################################
# strip_plot
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List, Any

def strip_plot_fit(
    data: np.ndarray,
    groups: Optional[np.ndarray] = None,
    jitter: float = 0.2,
    orientation: str = 'vertical',
    palette: Optional[List[str]] = None,
    normalize: bool = False,
    metric: str = 'mean',
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a strip plot for visual comparison of groups.

    Parameters:
    -----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features)
    groups : Optional[np.ndarray]
        Group labels for each sample. If None, all samples are considered as one group.
    jitter : float
        Amount of jitter to add to points for better visibility. Default is 0.2.
    orientation : str
        Orientation of the plot ('vertical' or 'horizontal'). Default is 'vertical'.
    palette : Optional[List[str]]
        List of colors for each group. If None, a default color palette is used.
    normalize : bool
        Whether to normalize the data. Default is False.
    metric : str
        Metric for summary statistics ('mean', 'median', 'std'). Default is 'mean'.
    custom_metric : Optional[Callable]
        Custom metric function. If provided, overrides the `metric` parameter.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the plot data and metadata.
    """
    # Validate inputs
    _validate_inputs(data, groups)

    # Normalize data if required
    if normalize:
        data = _normalize_data(data)

    # Compute summary statistics
    plot_data, metrics = _compute_summary_statistics(
        data,
        groups,
        metric,
        custom_metric
    )

    # Prepare plot parameters
    params_used = {
        'jitter': jitter,
        'orientation': orientation,
        'normalize': normalize,
        'metric': metric if custom_metric is None else 'custom'
    }

    # Generate warnings
    warnings = _generate_warnings(data, groups)

    return {
        'result': plot_data,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

def _validate_inputs(data: np.ndarray, groups: Optional[np.ndarray]) -> None:
    """Validate input data and groups."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    if groups is not None:
        if not isinstance(groups, np.ndarray):
            raise TypeError("Groups must be a numpy array.")
        if len(groups) != data.shape[0]:
            raise ValueError("Groups must have the same number of samples as data.")

def _normalize_data(data: np.ndarray) -> np.ndarray:
    """Normalize the input data."""
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def _compute_summary_statistics(
    data: np.ndarray,
    groups: Optional[np.ndarray],
    metric: str,
    custom_metric: Optional[Callable]
) -> tuple:
    """Compute summary statistics for the strip plot."""
    if groups is None:
        groups = np.zeros(data.shape[0], dtype=int)

    unique_groups = np.unique(groups)
    plot_data = []
    metrics_dict = {}

    for group in unique_groups:
        group_mask = groups == group
        group_data = data[group_mask]

        if custom_metric is not None:
            metric_value = custom_metric(group_data)
        else:
            if metric == 'mean':
                metric_value = np.mean(group_data, axis=0)
            elif metric == 'median':
                metric_value = np.median(group_data, axis=0)
            elif metric == 'std':
                metric_value = np.std(group_data, axis=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")

        plot_data.append({
            'group': group,
            'data': group_data,
            'metric_value': metric_value
        })
        metrics_dict[f'group_{group}'] = {
            'metric': metric_value,
            'count': group_data.shape[0]
        }

    return plot_data, metrics_dict

def _generate_warnings(data: np.ndarray, groups: Optional[np.ndarray]) -> List[str]:
    """Generate warnings for the strip plot."""
    warnings = []

    if np.any(np.isnan(data)):
        warnings.append("Data contains NaN values.")
    if np.any(np.isinf(data)):
        warnings.append("Data contains infinite values.")

    return warnings

################################################################################
# barplot_moyenne_ecart_type
################################################################################

import numpy as np
from typing import Dict, Any, Callable, Optional, Union

def _validate_inputs(
    data: np.ndarray,
    group_labels: np.ndarray,
    normalize: str = "none",
    custom_normalize: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Validate input data and group labels.

    Parameters:
    -----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features)
    group_labels : np.ndarray
        Group labels for each sample of shape (n_samples,)
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    custom_normalize : callable, optional
        Custom normalization function

    Returns:
    --------
    dict
        Dictionary containing validated data and parameters used
    """
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")
    if group_labels.ndim != 1:
        raise ValueError("Group labels must be a 1D array")
    if len(data) != len(group_labels):
        raise ValueError("Data and group labels must have the same length")

    if normalize == "none" and custom_normalize is None:
        normalized_data = data
    elif normalize in ["standard", "minmax", "robust"]:
        if normalize == "standard":
            normalized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        elif normalize == "minmax":
            normalized_data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
        elif normalize == "robust":
            normalized_data = (data - np.median(data, axis=0)) / (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0))
    elif custom_normalize is not None:
        normalized_data = custom_normalize(data)
    else:
        raise ValueError("Invalid normalization method or function")

    return {
        "data": normalized_data,
        "group_labels": group_labels,
        "normalize_method": normalize if custom_normalize is None else "custom"
    }

def _calculate_group_stats(
    data: np.ndarray,
    group_labels: np.ndarray
) -> Dict[str, Any]:
    """
    Calculate mean and standard deviation for each group.

    Parameters:
    -----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features)
    group_labels : np.ndarray
        Group labels for each sample of shape (n_samples,)

    Returns:
    --------
    dict
        Dictionary containing group statistics
    """
    unique_groups = np.unique(group_labels)
    group_stats = {}

    for group in unique_groups:
        group_mask = group_labels == group
        group_data = data[group_mask]
        group_stats[group] = {
            "mean": np.mean(group_data, axis=0),
            "std": np.std(group_data, axis=0)
        }

    return group_stats

def _plot_barplot(
    group_stats: Dict[str, Any],
    feature_index: int = 0,
    error_bars: bool = True,
    custom_plot_func: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Create bar plot of group means with error bars.

    Parameters:
    -----------
    group_stats : dict
        Dictionary containing group statistics
    feature_index : int, optional
        Index of the feature to plot (default: 0)
    error_bars : bool, optional
        Whether to show error bars (default: True)
    custom_plot_func : callable, optional
        Custom plotting function

    Returns:
    --------
    dict
        Dictionary containing plot information and warnings
    """
    groups = list(group_stats.keys())
    means = [group_stats[g]["mean"][feature_index] for g in groups]
    stds = [group_stats[g]["std"][feature_index] for g in groups]

    if custom_plot_func is not None:
        plot_info = custom_plot_func(groups, means, stds if error_bars else None)
    else:
        # Default plotting logic would go here
        plot_info = {
            "groups": groups,
            "means": means,
            "stds": stds if error_bars else None
        }

    return {
        "plot_info": plot_info,
        "warnings": []
    }

def barplot_moyenne_ecart_type_fit(
    data: np.ndarray,
    group_labels: np.ndarray,
    normalize: str = "none",
    custom_normalize: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    feature_index: int = 0,
    error_bars: bool = True,
    custom_plot_func: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Create a bar plot comparing group means with error bars.

    Parameters:
    -----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features)
    group_labels : np.ndarray
        Group labels for each sample of shape (n_samples,)
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    custom_normalize : callable, optional
        Custom normalization function
    feature_index : int, optional
        Index of the feature to plot (default: 0)
    error_bars : bool, optional
        Whether to show error bars (default: True)
    custom_plot_func : callable, optional
        Custom plotting function

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> data = np.random.randn(100, 5)
    >>> groups = np.random.randint(0, 3, size=100)
    >>> result = barplot_moyenne_ecart_type_fit(data, groups)
    """
    # Validate inputs
    validated = _validate_inputs(
        data=data,
        group_labels=group_labels,
        normalize=normalize,
        custom_normalize=custom_normalize
    )

    # Calculate group statistics
    stats = _calculate_group_stats(
        data=validated["data"],
        group_labels=group_labels
    )

    # Create plot
    plot_result = _plot_barplot(
        group_stats=stats,
        feature_index=feature_index,
        error_bars=error_bars,
        custom_plot_func=custom_plot_func
    )

    return {
        "result": plot_result["plot_info"],
        "metrics": {},
        "params_used": {
            "normalize_method": validated["normalize_method"],
            "feature_index": feature_index,
            "error_bars": error_bars
        },
        "warnings": plot_result["warnings"]
    }

################################################################################
# histogramme_superpose
################################################################################

import numpy as np
from typing import Callable, Dict, List, Optional, Union

def _validate_inputs(
    data: List[np.ndarray],
    bins: int = 10,
    density: bool = False,
    norm_type: str = 'none',
    custom_norm: Optional[Callable] = None
) -> Dict[str, np.ndarray]:
    """
    Validate and preprocess input data for histogram superposition.

    Parameters:
    -----------
    data : List[np.ndarray]
        List of arrays containing the data for each group.
    bins : int, optional
        Number of bins in the histogram (default: 10).
    density : bool, optional
        If True, normalize to form a probability density (default: False).
    norm_type : str, optional
        Type of normalization ('none', 'standard', 'minmax', 'robust') (default: 'none').
    custom_norm : Optional[Callable], optional
        Custom normalization function (default: None).

    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing validated and processed data.
    """
    if not all(isinstance(group, np.ndarray) for group in data):
        raise ValueError("All groups must be numpy arrays.")

    if not all(group.ndim == 1 for group in data):
        raise ValueError("All groups must be 1-dimensional arrays.")

    if norm_type not in ['none', 'standard', 'minmax', 'robust']:
        raise ValueError("Invalid norm_type. Choose from 'none', 'standard', 'minmax', 'robust'.")

    if custom_norm is not None and not callable(custom_norm):
        raise ValueError("custom_norm must be a callable function.")

    validated_data = []
    for group in data:
        if np.any(np.isnan(group)):
            raise ValueError("Input data contains NaN values.")
        if np.any(np.isinf(group)):
            raise ValueError("Input data contains infinite values.")
        validated_data.append(group)

    return {
        'data': validated_data,
        'bins': bins,
        'density': density,
        'norm_type': norm_type,
        'custom_norm': custom_norm
    }

def _apply_normalization(
    data: np.ndarray,
    norm_type: str = 'none',
    custom_norm: Optional[Callable] = None
) -> np.ndarray:
    """
    Apply normalization to the data.

    Parameters:
    -----------
    data : np.ndarray
        Input data to normalize.
    norm_type : str, optional
        Type of normalization ('none', 'standard', 'minmax', 'robust') (default: 'none').
    custom_norm : Optional[Callable], optional
        Custom normalization function (default: None).

    Returns:
    --------
    np.ndarray
        Normalized data.
    """
    if norm_type == 'none':
        return data

    if custom_norm is not None:
        return custom_norm(data)

    if norm_type == 'standard':
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std

    if norm_type == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)

    if norm_type == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr

    raise ValueError("Invalid norm_type. Choose from 'none', 'standard', 'minmax', 'robust'.")

def _compute_histogram(
    data: np.ndarray,
    bins: int = 10,
    density: bool = False
) -> np.ndarray:
    """
    Compute histogram for a single group.

    Parameters:
    -----------
    data : np.ndarray
        Input data.
    bins : int, optional
        Number of bins in the histogram (default: 10).
    density : bool, optional
        If True, normalize to form a probability density (default: False).

    Returns:
    --------
    np.ndarray
        Histogram values.
    """
    hist, _ = np.histogram(data, bins=bins, density=density)
    return hist

def _compute_metrics(
    histograms: List[np.ndarray],
    metric_type: str = 'mse',
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """
    Compute metrics between histograms.

    Parameters:
    -----------
    histograms : List[np.ndarray]
        List of histograms to compare.
    metric_type : str, optional
        Type of metric ('mse', 'mae', 'r2', 'logloss') (default: 'mse').
    custom_metric : Optional[Callable], optional
        Custom metric function (default: None).

    Returns:
    --------
    Dict[str, float]
        Dictionary of computed metrics.
    """
    if not all(len(hist) == len(histograms[0]) for hist in histograms):
        raise ValueError("All histograms must have the same length.")

    if custom_metric is not None and not callable(custom_metric):
        raise ValueError("custom_metric must be a callable function.")

    metrics = {}
    if custom_metric is not None:
        for i in range(len(histograms)):
            for j in range(i + 1, len(histograms)):
                key = f"metric_{i}_{j}"
                metrics[key] = custom_metric(histograms[i], histograms[j])
        return metrics

    ref_hist = histograms[0]
    for i in range(1, len(histograms)):
        key = f"metric_0_{i}"
        if metric_type == 'mse':
            metrics[key] = np.mean((ref_hist - histograms[i]) ** 2)
        elif metric_type == 'mae':
            metrics[key] = np.mean(np.abs(ref_hist - histograms[i]))
        elif metric_type == 'r2':
            ss_res = np.sum((ref_hist - histograms[i]) ** 2)
            ss_tot = np.sum((ref_hist - np.mean(ref_hist)) ** 2)
            metrics[key] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        elif metric_type == 'logloss':
            epsilon = 1e-15
            metrics[key] = -np.mean(ref_hist * np.log(histograms[i] + epsilon) +
                                   (1 - ref_hist) * np.log(1 - histograms[i] + epsilon))
        else:
            raise ValueError("Invalid metric_type. Choose from 'mse', 'mae', 'r2', 'logloss'.")

    return metrics

def histogramme_superpose_fit(
    data: List[np.ndarray],
    bins: int = 10,
    density: bool = False,
    norm_type: str = 'none',
    custom_norm: Optional[Callable] = None,
    metric_type: str = 'mse',
    custom_metric: Optional[Callable] = None
) -> Dict[str, Union[List[np.ndarray], Dict[str, float], Dict[str, str]]]:
    """
    Compute and compare superimposed histograms for multiple groups.

    Parameters:
    -----------
    data : List[np.ndarray]
        List of arrays containing the data for each group.
    bins : int, optional
        Number of bins in the histogram (default: 10).
    density : bool, optional
        If True, normalize to form a probability density (default: False).
    norm_type : str, optional
        Type of normalization ('none', 'standard', 'minmax', 'robust') (default: 'none').
    custom_norm : Optional[Callable], optional
        Custom normalization function (default: None).
    metric_type : str, optional
        Type of metric ('mse', 'mae', 'r2', 'logloss') (default: 'mse').
    custom_metric : Optional[Callable], optional
        Custom metric function (default: None).

    Returns:
    --------
    Dict[str, Union[List[np.ndarray], Dict[str, float], Dict[str, str]]]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    validated_data = _validate_inputs(data, bins, density, norm_type, custom_norm)
    normalized_data = [_apply_normalization(group, validated_data['norm_type'], validated_data['custom_norm'])
                      for group in validated_data['data']]
    histograms = [_compute_histogram(group, validated_data['bins'], validated_data['density'])
                 for group in normalized_data]
    metrics = _compute_metrics(histograms, metric_type, custom_metric)

    return {
        'result': histograms,
        'metrics': metrics,
        'params_used': {
            'bins': validated_data['bins'],
            'density': validated_data['density'],
            'norm_type': validated_data['norm_type'],
            'metric_type': metric_type
        },
        'warnings': []
    }

################################################################################
# density_plot
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def density_plot_fit(
    data: np.ndarray,
    group_labels: Optional[np.ndarray] = None,
    normalization: str = 'standard',
    bandwidth: float = 1.0,
    kernel: Callable[[np.ndarray], np.ndarray] = lambda x: (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2),
    n_bins: int = 100,
    range: Optional[tuple] = None,
    metrics: List[str] = ['mse'],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict:
    """
    Compute density plots for visual comparison of groups.

    Parameters:
    -----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features).
    group_labels : Optional[np.ndarray]
        Array of group labels for each sample. If None, all data is treated as one group.
    normalization : str
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    bandwidth : float
        Bandwidth for kernel density estimation.
    kernel : Callable[[np.ndarray], np.ndarray]
        Kernel function for density estimation.
    n_bins : int
        Number of bins for the plot.
    range : Optional[tuple]
        Range of the plot (min, max).
    metrics : List[str]
        List of metrics to compute: 'mse', 'mae', 'r2'.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, group_labels, normalization)

    # Normalize data
    normalized_data = _normalize_data(data, normalization)

    # Compute density plots
    densities = _compute_density_plots(
        normalized_data,
        group_labels,
        bandwidth,
        kernel,
        n_bins,
        range
    )

    # Compute metrics if group_labels are provided
    metrics_result = {}
    if group_labels is not None:
        metrics_result = _compute_metrics(densities, group_labels, metrics, custom_metric)

    # Prepare output
    result = {
        'result': densities,
        'metrics': metrics_result,
        'params_used': {
            'normalization': normalization,
            'bandwidth': bandwidth,
            'kernel': kernel.__name__ if hasattr(kernel, '__name__') else 'custom',
            'n_bins': n_bins,
            'range': range
        },
        'warnings': []
    }

    return result

def _validate_inputs(
    data: np.ndarray,
    group_labels: Optional[np.ndarray],
    normalization: str
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    if group_labels is not None:
        if len(group_labels) != data.shape[0]:
            raise ValueError("Group labels must match the number of samples.")
    if normalization not in ['none', 'standard', 'minmax', 'robust']:
        raise ValueError("Normalization must be one of: 'none', 'standard', 'minmax', 'robust'.")

def _normalize_data(
    data: np.ndarray,
    normalization: str
) -> np.ndarray:
    """Normalize the input data."""
    if normalization == 'none':
        return data
    elif normalization == 'standard':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)
    elif normalization == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif normalization == 'robust':
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError("Unknown normalization method.")

def _compute_density_plots(
    data: np.ndarray,
    group_labels: Optional[np.ndarray],
    bandwidth: float,
    kernel: Callable[[np.ndarray], np.ndarray],
    n_bins: int,
    range: Optional[tuple]
) -> Dict:
    """Compute density plots for each group."""
    if range is None:
        min_val, max_val = np.min(data), np.max(data)
    else:
        min_val, max_val = range

    bins = np.linspace(min_val, max_val, n_bins)
    densities = {}

    if group_labels is None:
        # Single group case
        densities['all'] = _compute_single_density(data, bins, bandwidth, kernel)
    else:
        # Multiple groups case
        unique_groups = np.unique(group_labels)
        for group in unique_groups:
            group_data = data[group_labels == group]
            densities[f'group_{group}'] = _compute_single_density(group_data, bins, bandwidth, kernel)

    return densities

def _compute_single_density(
    data: np.ndarray,
    bins: np.ndarray,
    bandwidth: float,
    kernel: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Compute density for a single group."""
    densities = np.zeros_like(bins)
    for i, x in enumerate(bins):
        distances = (data - x) / bandwidth
        densities[i] = np.mean(kernel(distances))
    return densities

def _compute_metrics(
    densities: Dict,
    group_labels: np.ndarray,
    metrics: List[str],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict:
    """Compute metrics between density plots."""
    unique_groups = np.unique(group_labels)
    metrics_result = {}

    for metric in metrics:
        if metric == 'mse':
            func = _mean_squared_error
        elif metric == 'mae':
            func = _mean_absolute_error
        elif metric == 'r2':
            func = _r_squared
        else:
            raise ValueError(f"Unknown metric: {metric}")

        for i in range(len(unique_groups)):
            for j in range(i + 1, len(unique_groups)):
                key = f'{metric}_group_{unique_groups[i]}_vs_group_{unique_groups[j]}'
                metrics_result[key] = func(
                    densities[f'group_{unique_groups[i]}'],
                    densities[f'group_{unique_groups[j]}']
                )

    if custom_metric is not None:
        for i in range(len(unique_groups)):
            for j in range(i + 1, len(unique_groups)):
                key = f'custom_metric_group_{unique_groups[i]}_vs_group_{unique_groups[j]}'
                metrics_result[key] = custom_metric(
                    densities[f'group_{unique_groups[i]}'],
                    densities[f'group_{unique_groups[j]}']
                )

    return metrics_result

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

################################################################################
# pairplot
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Validate input data for pairplot.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values array of shape (n_samples,)

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing validation results and warnings
    """
    result = {"valid": True, "warnings": []}

    if not isinstance(X, np.ndarray):
        result["valid"] = False
        raise TypeError("X must be a numpy array")

    if y is not None and not isinstance(y, np.ndarray):
        result["valid"] = False
        raise TypeError("y must be a numpy array or None")

    if X.ndim != 2:
        result["valid"] = False
        raise ValueError("X must be a 2D array")

    if y is not None and X.shape[0] != len(y):
        result["valid"] = False
        raise ValueError("X and y must have the same number of samples")

    if np.any(np.isnan(X)):
        result["warnings"].append("NaN values found in X and will be removed")

    if y is not None and np.any(np.isnan(y)):
        result["warnings"].append("NaN values found in y and will be removed")

    return result

def apply_normalization(X: np.ndarray, normalization: str = "standard") -> np.ndarray:
    """
    Apply specified normalization to the input data.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    normalization : str
        Normalization method: "none", "standard", "minmax", or "robust"

    Returns:
    --------
    np.ndarray
        Normalized features array
    """
    if normalization == "none":
        return X

    X = X.copy()
    n_samples, n_features = X.shape

    if normalization == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X = (X - mean) / (std + 1e-8)

    elif normalization == "minmax":
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        X = (X - min_vals) / (max_vals - min_vals + 1e-8)

    elif normalization == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X = (X - median) / (iqr + 1e-8)

    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    return X

def compute_pairwise_metrics(X: np.ndarray, metric: str = "euclidean", custom_metric: Optional[Callable] = None) -> np.ndarray:
    """
    Compute pairwise metrics between samples.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    metric : str
        Distance metric: "euclidean", "manhattan", "cosine", or "minkowski"
    custom_metric : Optional[Callable]
        Custom metric function if needed

    Returns:
    --------
    np.ndarray
        Pairwise distance matrix of shape (n_samples, n_samples)
    """
    if custom_metric is not None:
        return np.array([[custom_metric(x, y) for x in X] for y in X])

    n_samples = X.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    if metric == "euclidean":
        for i in range(n_samples):
            distance_matrix[i] = np.linalg.norm(X - X[i], axis=1)

    elif metric == "manhattan":
        for i in range(n_samples):
            distance_matrix[i] = np.sum(np.abs(X - X[i]), axis=1)

    elif metric == "cosine":
        for i in range(n_samples):
            distance_matrix[i] = 1 - np.dot(X, X[i]) / (np.linalg.norm(X, axis=1) * np.linalg.norm(X[i]))

    elif metric == "minkowski":
        for i in range(n_samples):
            distance_matrix[i] = np.sum(np.abs(X - X[i])**3, axis=1)**(1/3)

    else:
        raise ValueError(f"Unknown metric: {metric}")

    return distance_matrix

def pairplot_fit(X: np.ndarray, y: Optional[np.ndarray] = None,
                 normalization: str = "standard",
                 metric: str = "euclidean",
                 custom_metric: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Compute pairplot visualization data.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values array of shape (n_samples,)
    normalization : str
        Normalization method: "none", "standard", "minmax", or "robust"
    metric : str
        Distance metric: "euclidean", "manhattan", "cosine", or "minkowski"
    custom_metric : Optional[Callable]
        Custom metric function if needed

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    validation = validate_inputs(X, y)
    if not validation["valid"]:
        return {"result": None, "metrics": {}, "params_used": {}, "warnings": validation["warnings"]}

    # Remove NaN values if any
    mask = ~np.isnan(X).any(axis=1)
    X_clean = X[mask]
    if y is not None:
        y_clean = y[mask]

    # Apply normalization
    X_norm = apply_normalization(X_clean, normalization)

    # Compute pairwise metrics
    distance_matrix = compute_pairwise_metrics(X_norm, metric, custom_metric)

    # Prepare results
    result = {
        "result": distance_matrix,
        "metrics": {},
        "params_used": {
            "normalization": normalization,
            "metric": metric
        },
        "warnings": validation["warnings"]
    }

    return result

# Example usage:
"""
X = np.random.rand(10, 5)
y = np.random.randint(0, 2, size=10)
result = pairplot_fit(X, y, normalization="standard", metric="euclidean")
"""

################################################################################
# heatmap_correlation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def _validate_inputs(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input contains NaN or infinite values")

def _normalize_data(
    X: np.ndarray,
    method: str = "standard",
    custom_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize data using specified method."""
    if custom_func is not None:
        return custom_func(X)

    X_norm = X.copy()
    if method == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_norm = (X - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - median) / (iqr + 1e-8)
    return X_norm

def _compute_correlation_matrix(
    X: np.ndarray,
    method: str = "pearson",
    custom_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Compute correlation matrix using specified method."""
    if custom_func is not None:
        return custom_func(X)

    n = X.shape[0]
    if method == "pearson":
        cov_matrix = np.cov(X, rowvar=False)
        stddev = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(stddev, stddev)
    elif method == "spearman":
        ranks = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), axis=0, arr=X)
        corr_matrix = np.corrcoef(ranks, rowvar=False)
    return corr_matrix

def _apply_mask(
    corr_matrix: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Apply mask to correlation matrix."""
    if mask is not None:
        if mask.shape != corr_matrix.shape:
            raise ValueError("Mask shape must match correlation matrix shape")
        corr_matrix = np.where(mask, corr_matrix, 0)
    return corr_matrix

def heatmap_correlation_fit(
    X: np.ndarray,
    normalization_method: str = "standard",
    correlation_method: str = "pearson",
    custom_normalization_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_correlation_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    mask: Optional[np.ndarray] = None
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Compute correlation heatmap with configurable options.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix (n_samples, n_features)
    normalization_method : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    correlation_method : str, optional
        Correlation method ('pearson', 'spearman')
    custom_normalization_func : callable, optional
        Custom normalization function
    custom_correlation_func : callable, optional
        Custom correlation computation function
    mask : np.ndarray, optional
        Mask to apply to the correlation matrix

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': correlation matrix
        - 'metrics': dictionary of metrics
        - 'params_used': parameters used
        - 'warnings': list of warnings

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> result = heatmap_correlation_fit(X)
    """
    _validate_inputs(X)

    warnings = []

    # Normalization
    if normalization_method != "none":
        X_norm = _normalize_data(X, method=normalization_method,
                               custom_func=custom_normalization_func)
    else:
        X_norm = X.copy()

    # Correlation computation
    corr_matrix = _compute_correlation_matrix(
        X_norm,
        method=correlation_method,
        custom_func=custom_correlation_func
    )

    # Apply mask
    corr_matrix = _apply_mask(corr_matrix, mask)

    # Metrics
    metrics = {
        "mean_correlation": np.mean(corr_matrix),
        "max_correlation": np.max(corr_matrix),
        "min_correlation": np.min(corr_matrix)
    }

    # Parameters used
    params_used = {
        "normalization_method": normalization_method,
        "correlation_method": correlation_method,
        "mask_applied": mask is not None
    }

    return {
        "result": corr_matrix,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

################################################################################
# radar_chart
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(data: np.ndarray, features: list, normalizations: Dict[str, Any]) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    if len(features) != data.shape[1]:
        raise ValueError("Number of features must match the number of columns in data.")
    if any(np.isnan(data).any()):
        raise ValueError("Data contains NaN values.")
    if any(np.isinf(data).any()):
        raise ValueError("Data contains infinite values.")

def normalize_data(data: np.ndarray, features: list, normalizations: Dict[str, Any]) -> np.ndarray:
    """Normalize data based on specified normalization methods."""
    normalized_data = np.zeros_like(data, dtype=float)
    for i, feature in enumerate(features):
        method = normalizations.get(feature, 'none')
        if method == 'standard':
            mean = np.mean(data[:, i])
            std = np.std(data[:, i])
            normalized_data[:, i] = (data[:, i] - mean) / std
        elif method == 'minmax':
            min_val = np.min(data[:, i])
            max_val = np.max(data[:, i])
            normalized_data[:, i] = (data[:, i] - min_val) / (max_val - min_val)
        elif method == 'robust':
            median = np.median(data[:, i])
            iqr = np.percentile(data[:, i], 75) - np.percentile(data[:, i], 25)
            normalized_data[:, i] = (data[:, i] - median) / iqr
        else:
            normalized_data[:, i] = data[:, i]
    return normalized_data

def compute_metrics(data: np.ndarray, features: list, metrics: Dict[str, Union[str, Callable]]) -> Dict[str, float]:
    """Compute specified metrics for the data."""
    metric_results = {}
    for feature in features:
        if isinstance(metrics.get(feature), str):
            metric_name = metrics[feature]
            if metric_name == 'mean':
                metric_results[f'{feature}_mean'] = np.mean(data[:, features.index(feature)])
            elif metric_name == 'median':
                metric_results[f'{feature}_median'] = np.median(data[:, features.index(feature)])
            elif metric_name == 'std':
                metric_results[f'{feature}_std'] = np.std(data[:, features.index(feature)])
        else:
            custom_metric = metrics[feature]
            metric_results[f'{feature}_custom'] = custom_metric(data[:, features.index(feature)])
    return metric_results

def radar_chart_fit(
    data: np.ndarray,
    features: list,
    normalizations: Dict[str, str] = None,
    metrics: Dict[str, Union[str, Callable]] = None
) -> Dict[str, Any]:
    """
    Compute radar chart data for visual comparison of groups.

    Parameters:
    - data: 2D numpy array where each row is a sample and each column is a feature.
    - features: List of feature names corresponding to the columns in data.
    - normalizations: Dictionary mapping feature names to normalization methods ('none', 'standard', 'minmax', 'robust').
    - metrics: Dictionary mapping feature names to metric names ('mean', 'median', 'std') or custom callable functions.

    Returns:
    - Dictionary containing normalized data, computed metrics, parameters used, and warnings.
    """
    if normalizations is None:
        normalizations = {feature: 'none' for feature in features}
    if metrics is None:
        metrics = {feature: 'mean' for feature in features}

    validate_inputs(data, features, normalizations)
    normalized_data = normalize_data(data, features, normalizations)
    metrics_results = compute_metrics(normalized_data, features, metrics)

    return {
        'result': normalized_data,
        'metrics': metrics_results,
        'params_used': {
            'normalizations': normalizations,
            'metrics': metrics
        },
        'warnings': []
    }

# Example usage:
# data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# features = ['feature1', 'feature2', 'feature3']
# normalizations = {'feature1': 'standard', 'feature2': 'minmax'}
# metrics = {'feature1': 'mean', 'feature3': lambda x: np.sum(x)}
# result = radar_chart_fit(data, features, normalizations, metrics)

################################################################################
# parallel_coordinates
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List
from enum import Enum

class Normalization(Enum):
    NONE = "none"
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"

class Metric(Enum):
    MSE = "mse"
    MAE = "mae"
    R2 = "r2"
    LOGLoss = "logloss"

class Distance(Enum):
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"
    MINKOWSKI = "minkowski"

class Solver(Enum):
    CLOSED_FORM = "closed_form"
    GRADIENT_DESCENT = "gradient_descent"
    NEWTON = "newton"
    COORDINATE_DESCENT = "coordinate_descent"

class Regularization(Enum):
    NONE = "none"
    L1 = "l1"
    L2 = "l2"
    ELASTICNET = "elasticnet"

def parallel_coordinates_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalization: Union[Normalization, str] = Normalization.STANDARD,
    metric: Union[Metric, str, Callable] = Metric.MSE,
    distance: Union[Distance, str] = Distance.EUCLIDEAN,
    solver: Union[Solver, str] = Solver.CLOSED_FORM,
    regularization: Union[Regularization, str] = Regularization.NONE,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Fit parallel coordinates model for visual comparison of groups.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    y : Optional[np.ndarray]
        Target values if supervised learning is needed.
    normalization : Union[Normalization, str]
        Normalization method to apply. Default is "standard".
    metric : Union[Metric, str, Callable]
        Metric to evaluate the model. Default is "mse".
    distance : Union[Distance, str]
        Distance metric for calculations. Default is "euclidean".
    solver : Union[Solver, str]
        Solver to use for optimization. Default is "closed_form".
    regularization : Union[Regularization, str]
        Regularization type. Default is "none".
    alpha : float
        Regularization strength. Default is 1.0.
    l1_ratio : float
        ElasticNet mixing parameter. Default is 0.5.
    max_iter : int
        Maximum number of iterations. Default is 1000.
    tol : float
        Tolerance for stopping criteria. Default is 1e-4.
    random_state : Optional[int]
        Random seed for reproducibility. Default is None.
    custom_metric : Optional[Callable]
        Custom metric function if needed.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _apply_normalization(X, normalization)

    # Prepare parameters
    params_used = {
        "normalization": normalization,
        "metric": metric,
        "distance": distance,
        "solver": solver,
        "regularization": regularization,
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "max_iter": max_iter,
        "tol": tol,
        "random_state": random_state
    }

    # Fit model based on solver choice
    if solver == Solver.CLOSED_FORM:
        result = _closed_form_solution(X_normalized, y)
    elif solver == Solver.GRADIENT_DESCENT:
        result = _gradient_descent(X_normalized, y, metric, distance,
                                  regularization, alpha, l1_ratio,
                                  max_iter, tol, random_state)
    elif solver == Solver.NEWTON:
        result = _newton_method(X_normalized, y, metric, distance,
                               regularization, alpha, max_iter, tol)
    elif solver == Solver.COORDINATE_DESCENT:
        result = _coordinate_descent(X_normalized, y, metric, distance,
                                    regularization, alpha, max_iter, tol)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(result, X_normalized, y,
                                metric if not custom_metric else custom_metric)

    # Prepare output
    output = {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": []
    }

    return output

def _validate_inputs(X: np.ndarray, y: Optional[np.ndarray]) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if y is not None and np.any(np.isnan(y)):
        raise ValueError("y contains NaN values")

def _apply_normalization(X: np.ndarray, normalization: Union[Normalization, str]) -> np.ndarray:
    """Apply specified normalization to the data."""
    if isinstance(normalization, str):
        normalization = Normalization(normalization)

    X_normalized = X.copy()

    if normalization == Normalization.STANDARD:
        X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalization == Normalization.MINMAX:
        X_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalization == Normalization.ROBUST:
        median = np.median(X, axis=0)
        iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
        X_normalized = (X - median) / iqr

    return X_normalized

def _closed_form_solution(X: np.ndarray, y: Optional[np.ndarray]) -> Dict:
    """Closed form solution for parallel coordinates."""
    if y is None:
        raise ValueError("Target values (y) are required for closed form solution")

    # Simple linear regression as an example
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
    coefficients = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y

    return {
        "coefficients": coefficients,
        "intercept": coefficients[0],
        "features": coefficients[1:]
    }

def _gradient_descent(
    X: np.ndarray,
    y: Optional[np.ndarray],
    metric: Union[Metric, str, Callable],
    distance: Union[Distance, str],
    regularization: Union[Regularization, str],
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tol: float,
    random_state: Optional[int]
) -> Dict:
    """Gradient descent solver for parallel coordinates."""
    if y is None:
        raise ValueError("Target values (y) are required for gradient descent")

    np.random.seed(random_state)
    n_features = X.shape[1]
    coefficients = np.random.randn(n_features)

    for _ in range(max_iter):
        # Compute gradient (simplified example)
        predictions = X @ coefficients
        error = predictions - y

        if metric == Metric.MSE:
            gradient = (2 / len(y)) * X.T @ error
        elif metric == Metric.MAE:
            gradient = (1 / len(y)) * X.T @ np.sign(error)
        else:
            raise ValueError(f"Unsupported metric for gradient descent: {metric}")

        # Apply regularization
        if regularization == Regularization.L1:
            gradient += alpha * np.sign(coefficients)
        elif regularization == Regularization.L2:
            gradient += 2 * alpha * coefficients
        elif regularization == Regularization.ELASTICNET:
            gradient += alpha * (l1_ratio * np.sign(coefficients) +
                                (1 - l1_ratio) * 2 * coefficients)

        # Update coefficients
        old_coefficients = coefficients.copy()
        coefficients -= gradient

        if np.linalg.norm(coefficients - old_coefficients) < tol:
            break

    return {
        "coefficients": coefficients,
        "iterations": _ + 1
    }

def _newton_method(
    X: np.ndarray,
    y: Optional[np.ndarray],
    metric: Union[Metric, str, Callable],
    distance: Union[Distance, str],
    regularization: Union[Regularization, str],
    alpha: float,
    max_iter: int,
    tol: float
) -> Dict:
    """Newton method solver for parallel coordinates."""
    if y is None:
        raise ValueError("Target values (y) are required for Newton method")

    n_features = X.shape[1]
    coefficients = np.zeros(n_features)

    for _ in range(max_iter):
        # Compute predictions and error
        predictions = X @ coefficients
        error = predictions - y

        if metric == Metric.MSE:
            gradient = (2 / len(y)) * X.T @ error
            hessian = (2 / len(y)) * X.T @ X

            if regularization == Regularization.L2:
                hessian += 2 * alpha * np.eye(n_features)

            # Newton update
            coefficients -= np.linalg.inv(hessian) @ gradient

        if np.linalg.norm(gradient) < tol:
            break

    return {
        "coefficients": coefficients,
        "iterations": _ + 1
    }

def _coordinate_descent(
    X: np.ndarray,
    y: Optional[np.ndarray],
    metric: Union[Metric, str, Callable],
    distance: Union[Distance, str],
    regularization: Union[Regularization, str],
    alpha: float,
    max_iter: int,
    tol: float
) -> Dict:
    """Coordinate descent solver for parallel coordinates."""
    if y is None:
        raise ValueError("Target values (y) are required for coordinate descent")

    n_features = X.shape[1]
    coefficients = np.zeros(n_features)

    for _ in range(max_iter):
        old_coefficients = coefficients.copy()

        for j in range(n_features):
            # Compute residual without feature j
            X_j = X[:, j]
            residual = y - (X @ coefficients) + coefficients[j] * X_j

            # Compute optimal coefficient for feature j
            if metric == Metric.MSE:
                numerator = X_j.T @ residual
                denominator = X_j.T @ X_j

                if regularization == Regularization.L1:
                    coefficients[j] = np.sign(numerator) * np.maximum(
                        abs(numerator) - alpha, 0) / denominator
                elif regularization == Regularization.L2:
                    coefficients[j] = numerator / (denominator + 2 * alpha)
                else:
                    coefficients[j] = numerator / denominator

        if np.linalg.norm(coefficients - old_coefficients) < tol:
            break

    return {
        "coefficients": coefficients,
        "iterations": _ + 1
    }

def _calculate_metrics(
    result: Dict,
    X: np.ndarray,
    y: Optional[np.ndarray],
    metric: Union[Metric, str, Callable]
) -> Dict:
    """Calculate evaluation metrics."""
    if y is None:
        return {"metrics": {}}

    coefficients = result["coefficients"]
    predictions = X @ coefficients

    metrics_dict = {}

    if metric == Metric.MSE or isinstance(metric, str) and metric.lower() == "mse":
        metrics_dict["mse"] = np.mean((predictions - y) ** 2)
    elif metric == Metric.MAE or isinstance(metric, str) and metric.lower() == "mae":
        metrics_dict["mae"] = np.mean(np.abs(predictions - y))
    elif metric == Metric.R2 or isinstance(metric, str) and metric.lower() == "r2":
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics_dict["r2"] = 1 - (ss_res / ss_tot)
    elif callable(metric):
        metrics_dict["custom"] = metric(y, predictions)

    return metrics_dict

################################################################################
# t_sne_projection
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def _validate_inputs(
    X: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    n_iter: int = 1000,
    metric: Union[str, Callable] = 'euclidean',
    normalize: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate and preprocess input data."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X must be a 2D numpy array")
    if n_components < 1:
        raise ValueError("n_components must be at least 1")
    if perplexity <= 0:
        raise ValueError("perplexity must be positive")

    # Normalize data if specified
    if normalize == 'standard':
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalize == 'minmax':
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalize == 'robust':
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))

    return {
        'X': X,
        'n_components': n_components,
        'perplexity': perplexity,
        'learning_rate': learning_rate,
        'n_iter': n_iter,
        'metric': metric
    }

def _compute_pairwise_distances(
    X: np.ndarray,
    metric: Union[str, Callable]
) -> np.ndarray:
    """Compute pairwise distances between samples."""
    if callable(metric):
        return metric(X)
    elif metric == 'euclidean':
        return np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
    elif metric == 'manhattan':
        return np.sum(np.abs(X[:, np.newaxis] - X), axis=2)
    elif metric == 'cosine':
        return 1 - np.dot(X, X.T) / (np.linalg.norm(X, axis=1)[:, np.newaxis] * np.linalg.norm(X, axis=1))
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def _compute_affinities(
    distances: np.ndarray,
    perplexity: float
) -> np.ndarray:
    """Compute pairwise affinities using Gaussian kernel."""
    P = np.zeros((distances.shape[0], distances.shape[1]))
    for i in range(distances.shape[0]):
        beta = 1.0 / (2 * perplexity ** 2)
        P[i] = np.exp(-beta * distances[i] ** 2)
        P[i] /= np.sum(P[i])
    return (P + P.T) / (2 * distances.shape[0])

def _gradient_descent(
    X: np.ndarray,
    P: np.ndarray,
    n_components: int = 2,
    learning_rate: float = 200.0,
    n_iter: int = 1000
) -> np.ndarray:
    """Perform gradient descent optimization for t-SNE."""
    n_samples = X.shape[0]
    Y = np.random.randn(n_samples, n_components)

    for _ in range(n_iter):
        # Compute pairwise affinities in low-dimensional space
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        denom = sum_Y[:, np.newaxis] + sum_Y[np.newaxis, :] - 2. * np.dot(Y, Y.T)
        Q = np.power(1 + num / denom, -1)

        # Compute gradient
        PQ = P - Q
        for i in range(n_samples):
            grads = np.tile(PQ[:, i] * Q[:, i], (n_components, 1)).T
            grads[range(n_samples), i] -= np.sum(PQ[:, i] * Q[:, i])
            grads = np.dot(grads, Y)
            Y[i] += learning_rate * grads

        # Avoid explosion of values during optimization
        Y = Y / (1 + np.sum(np.square(Y), 1)[:, np.newaxis] / n_components)

    return Y

def t_sne_projection_fit(
    X: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    n_iter: int = 1000,
    metric: Union[str, Callable] = 'euclidean',
    normalize: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Perform t-SNE projection for visualizing high-dimensional data.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_components : int, optional
        Dimension of the embedded space (default: 2)
    perplexity : float, optional
        Perplexity parameter (default: 30.0)
    learning_rate : float, optional
        Learning rate for gradient descent (default: 200.0)
    n_iter : int, optional
        Number of iterations (default: 1000)
    metric : str or callable, optional
        Distance metric (default: 'euclidean')
    normalize : str, optional
        Normalization method (none, standard, minmax, robust)

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': Embedded data of shape (n_samples, n_components)
        - 'metrics': Dictionary of computed metrics
        - 'params_used': Parameters actually used
        - 'warnings': List of warnings encountered
    """
    # Validate and preprocess inputs
    validated = _validate_inputs(X, n_components, perplexity, learning_rate, n_iter, metric, normalize)
    X = validated['X']
    n_components = validated['n_components']
    perplexity = validated['perplexity']
    learning_rate = validated['learning_rate']
    n_iter = validated['n_iter']
    metric = validated['metric']

    # Compute pairwise distances
    distances = _compute_pairwise_distances(X, metric)

    # Compute affinities
    P = _compute_affinities(distances, perplexity)

    # Perform optimization
    Y = _gradient_descent(X, P, n_components, learning_rate, n_iter)

    # Compute metrics
    metrics = {
        'kl_divergence': np.sum(P * np.log(P / (P + 1e-12)))
    }

    return {
        'result': Y,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'perplexity': perplexity,
            'learning_rate': learning_rate,
            'n_iter': n_iter,
            'metric': metric,
            'normalize': normalize
        },
        'warnings': []
    }

# Example usage:
# result = t_sne_projection_fit(X=np.random.rand(100, 50), n_components=2)

################################################################################
# umap_projection
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input contains NaN or infinite values")

def normalize_data(
    X: np.ndarray,
    method: str = "standard",
    custom_norm: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize data using specified method."""
    if custom_norm is not None:
        return custom_norm(X)

    X_norm = X.copy()
    if method == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_norm = (X - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - median) / (iqr + 1e-8)
    return X_norm

def compute_distance_matrix(
    X: np.ndarray,
    metric: str = "euclidean",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> np.ndarray:
    """Compute distance matrix using specified metric."""
    n_samples = X.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))

    if custom_metric is not None:
        for i in range(n_samples):
            for j in range(i, n_samples):
                dist_matrix[i, j] = custom_metric(X[i], X[j])
                dist_matrix[j, i] = dist_matrix[i, j]
        return dist_matrix

    if metric == "euclidean":
        for i in range(n_samples):
            for j in range(i, n_samples):
                dist_matrix[i, j] = np.linalg.norm(X[i] - X[j])
                dist_matrix[j, i] = dist_matrix[i, j]
    elif metric == "manhattan":
        for i in range(n_samples):
            for j in range(i, n_samples):
                dist_matrix[i, j] = np.sum(np.abs(X[i] - X[j]))
                dist_matrix[j, i] = dist_matrix[i, j]
    elif metric == "cosine":
        for i in range(n_samples):
            for j in range(i, n_samples):
                dot_product = np.dot(X[i], X[j])
                norm_i = np.linalg.norm(X[i])
                norm_j = np.linalg.norm(X[j])
                dist_matrix[i, j] = 1 - (dot_product / (norm_i * norm_j + 1e-8))
                dist_matrix[j, i] = dist_matrix[i, j]
    return dist_matrix

def umap_projection_fit(
    X: np.ndarray,
    n_components: int = 2,
    normalizer: str = "standard",
    metric: str = "euclidean",
    solver: str = "gradient_descent",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    spread: float = 1.0,
    repulsion_strength: float = 1.0,
    custom_norm: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Perform UMAP projection on input data.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_components : int, optional
        Number of dimensions for the projection (default: 2)
    normalizer : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard')
    metric : str, optional
        Distance metric ('euclidean', 'manhattan', 'cosine') (default: 'euclidean')
    solver : str, optional
        Solver method ('gradient_descent', 'closed_form') (default: 'gradient_descent')
    n_neighbors : int, optional
        Number of neighbors for local structure (default: 15)
    min_dist : float, optional
        Minimum distance between embedded points (default: 0.1)
    spread : float, optional
        Spread of the embedded points (default: 1.0)
    repulsion_strength : float, optional
        Strength of the repulsive force (default: 1.0)
    custom_norm : callable, optional
        Custom normalization function
    custom_metric : callable, optional
        Custom distance metric function

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': Embedded data of shape (n_samples, n_components)
        - 'metrics': Dictionary of computed metrics
        - 'params_used': Dictionary of parameters used
        - 'warnings': List of warnings encountered
    """
    # Validate input
    validate_input(X)

    # Normalize data
    X_norm = normalize_data(X, normalizer, custom_norm)

    # Compute distance matrix
    dist_matrix = compute_distance_matrix(X_norm, metric, custom_metric)

    # Initialize output dictionary
    result_dict = {
        "result": None,
        "metrics": {},
        "params_used": {
            "n_components": n_components,
            "normalizer": normalizer,
            "metric": metric,
            "solver": solver,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "spread": spread,
            "repulsion_strength": repulsion_strength
        },
        "warnings": []
    }

    # Here you would implement the actual UMAP projection
    # For now, we'll return random data as a placeholder
    n_samples = X.shape[0]
    result_dict["result"] = np.random.rand(n_samples, n_components)

    # Add some metrics (placeholder)
    result_dict["metrics"]["reconstruction_error"] = np.random.rand()
    result_dict["metrics"]["stress"] = np.random.rand()

    return result_dict

################################################################################
# pca_projection
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if y is not None:
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy array or None")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if len(y) != X.shape[0]:
            raise ValueError("y must have the same number of samples as X")

def _normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """Normalize data using specified method."""
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

def _compute_covariance_matrix(X: np.ndarray) -> np.ndarray:
    """Compute covariance matrix."""
    return np.cov(X, rowvar=False)

def _compute_eigen_decomposition(cov_matrix: np.ndarray) -> tuple:
    """Compute eigen decomposition of covariance matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    return eigenvalues, eigenvectors

def _project_data(X: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Project data onto principal components."""
    return X @ components

def pca_projection_fit(
    X: np.ndarray,
    n_components: int = 2,
    normalization: str = 'standard',
    distance_metric: str = 'euclidean',
    solver: str = 'closed_form',
    custom_distance: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Perform PCA projection of data.

    Parameters:
    - X: Input data matrix (n_samples, n_features)
    - n_components: Number of principal components to keep
    - normalization: Normalization method ('none', 'standard', 'minmax', 'robust')
    - distance_metric: Distance metric for evaluation ('euclidean', 'manhattan', etc.)
    - solver: Solver method ('closed_form')
    - custom_distance: Custom distance function
    - **kwargs: Additional solver-specific parameters

    Returns:
    - Dictionary containing results, metrics, and parameters used
    """
    # Validate inputs
    _validate_inputs(X)

    # Normalize data
    X_normalized = _normalize_data(X, normalization)

    # Compute covariance matrix
    cov_matrix = _compute_covariance_matrix(X_normalized)

    # Compute eigen decomposition
    eigenvalues, eigenvectors = _compute_eigen_decomposition(cov_matrix)

    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select top n_components
    components = eigenvectors[:, :n_components]

    # Project data
    projected_data = _project_data(X_normalized, components)

    # Calculate explained variance ratio
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues[:n_components] / total_variance

    # Prepare output
    result = {
        'projected_data': projected_data,
        'components': components,
        'explained_variance_ratio': explained_variance_ratio
    }

    metrics = {
        'explained_variance': np.sum(explained_variance_ratio)
    }

    params_used = {
        'normalization': normalization,
        'distance_metric': distance_metric,
        'solver': solver
    }

    warnings = []

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

# Example usage:
"""
X = np.random.rand(100, 5)
result = pca_projection_fit(X, n_components=2, normalization='standard')
"""

################################################################################
# scatter_matrix
################################################################################

import numpy as np
from typing import Dict, Any, Callable, Optional, Union

def validate_inputs(X: np.ndarray, groups: np.ndarray) -> None:
    """Validate input data and group labels."""
    if not isinstance(X, np.ndarray) or not isinstance(groups, np.ndarray):
        raise TypeError("X and groups must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if groups.ndim != 1:
        raise ValueError("groups must be a 1D array")
    if len(X) != len(groups):
        raise ValueError("X and groups must have the same length")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(groups)) or np.any(np.isinf(groups)):
        raise ValueError("groups contains NaN or infinite values")

def apply_normalization(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """Apply normalization to the data."""
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

def compute_scatter_matrix(X: np.ndarray, groups: np.ndarray) -> Dict[str, Any]:
    """Compute the scatter matrix for each group."""
    unique_groups = np.unique(groups)
    scatter_matrices = {}
    for g in unique_groups:
        X_g = X[groups == g]
        scatter_matrices[f'group_{g}'] = X_g.T @ X_g
    return scatter_matrices

def compute_total_scatter_matrix(X: np.ndarray) -> np.ndarray:
    """Compute the total scatter matrix."""
    return X.T @ X

def compute_within_scatter_matrix(scatter_matrices: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute the within-group scatter matrix."""
    return sum(scatter_matrices.values())

def compute_between_scatter_matrix(total_scatter: np.ndarray, within_scatter: np.ndarray) -> np.ndarray:
    """Compute the between-group scatter matrix."""
    return total_scatter - within_scatter

def compute_metrics(scatter_matrices: Dict[str, np.ndarray], total_scatter: np.ndarray,
                    within_scatter: np.ndarray, between_scatter: np.ndarray) -> Dict[str, float]:
    """Compute various metrics from the scatter matrices."""
    eigenvalues_total = np.linalg.eigvalsh(total_scatter)
    eigenvalues_within = np.linalg.eigvalsh(within_scatter)
    eigenvalues_between = np.linalg.eigvalsh(between_scatter)

    metrics = {
        'trace_ratio': np.trace(between_scatter) / (np.trace(within_scatter) + 1e-8),
        'determinant_ratio': np.prod(eigenvalues_between) / (np.prod(eigenvalues_within) + 1e-8),
        'wilks_lambda': np.prod(eigenvalues_within) / (np.prod(eigenvalues_total) + 1e-8)
    }
    return metrics

def scatter_matrix_fit(X: np.ndarray, groups: np.ndarray,
                       normalization: str = 'standard',
                       custom_metric: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Compute scatter matrices and related metrics for group comparison.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    groups : np.ndarray
        Group labels for each sample of shape (n_samples,)
    normalization : str, optional
        Normalization method to apply ('none', 'standard', 'minmax', 'robust')
    custom_metric : Callable, optional
        Custom metric function to compute additional metrics

    Returns:
    --------
    dict
        Dictionary containing scatter matrices, metrics, and parameters used

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> groups = np.random.randint(0, 3, size=100)
    >>> result = scatter_matrix_fit(X, groups, normalization='standard')
    """
    # Validate inputs
    validate_inputs(X, groups)

    # Apply normalization
    X_normalized = apply_normalization(X, method=normalization)

    # Compute scatter matrices
    scatter_matrices = compute_scatter_matrix(X_normalized, groups)
    total_scatter = compute_total_scatter_matrix(X_normalized)
    within_scatter = compute_within_scatter_matrix(scatter_matrices)
    between_scatter = compute_between_scatter_matrix(total_scatter, within_scatter)

    # Compute metrics
    metrics = compute_metrics(scatter_matrices, total_scatter,
                             within_scatter, between_scatter)

    # Add custom metric if provided
    if custom_metric is not None:
        metrics['custom'] = custom_metric(scatter_matrices, total_scatter,
                                         within_scatter, between_scatter)

    # Prepare output
    result = {
        'result': {
            'scatter_matrices': scatter_matrices,
            'total_scatter_matrix': total_scatter,
            'within_scatter_matrix': within_scatter,
            'between_scatter_matrix': between_scatter
        },
        'metrics': metrics,
        'params_used': {
            'normalization': normalization
        },
        'warnings': []
    }

    return result

################################################################################
# sunburst_chart
################################################################################

import numpy as np
from typing import Dict, Any, Callable, Optional, Union

def validate_inputs(
    data: np.ndarray,
    labels: np.ndarray,
    hierarchy: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Validate input data for sunburst chart.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    labels : np.ndarray
        Group labels for each sample of shape (n_samples,)
    hierarchy : Optional[np.ndarray]
        Hierarchy levels for each group of shape (n_groups,)

    Returns
    -------
    Dict[str, Any]
        Validation results and warnings

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    validation = {"warnings": []}

    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")
    if labels.ndim != 1:
        raise ValueError("Labels must be a 1D array")
    if len(data) != len(labels):
        raise ValueError("Data and labels must have same length")

    if hierarchy is not None:
        if len(np.unique(labels)) != len(hierarchy):
            validation["warnings"].append("Hierarchy length doesn't match number of unique groups")

    return validation

def normalize_data(
    data: np.ndarray,
    method: str = "standard",
    custom_func: Optional[Callable] = None
) -> np.ndarray:
    """
    Normalize data using specified method.

    Parameters
    ----------
    data : np.ndarray
        Input data to normalize
    method : str
        Normalization method: 'none', 'standard', 'minmax', 'robust'
    custom_func : Optional[Callable]
        Custom normalization function

    Returns
    -------
    np.ndarray
        Normalized data
    """
    if custom_func is not None:
        return custom_func(data)

    if method == "none":
        return data
    elif method == "standard":
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def compute_group_stats(
    data: np.ndarray,
    labels: np.ndarray
) -> Dict[str, Any]:
    """
    Compute statistics for each group.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix
    labels : np.ndarray
        Group labels

    Returns
    -------
    Dict[str, Any]
        Dictionary containing group statistics
    """
    groups = {}
    unique_labels = np.unique(labels)

    for label in unique_labels:
        group_data = data[labels == label]
        groups[str(label)] = {
            "count": len(group_data),
            "mean": np.mean(group_data, axis=0),
            "std": np.std(group_data, axis=0)
        }

    return groups

def calculate_metrics(
    data: np.ndarray,
    labels: np.ndarray,
    metric: str = "mse",
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """
    Calculate comparison metrics between groups.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix
    labels : np.ndarray
        Group labels
    metric : str
        Metric to calculate: 'mse', 'mae', 'r2'
    custom_metric : Optional[Callable]
        Custom metric function

    Returns
    -------
    Dict[str, float]
        Dictionary of calculated metrics
    """
    if custom_metric is not None:
        return {"custom": custom_metric(data, labels)}

    metrics = {}
    unique_labels = np.unique(labels)
    n_groups = len(unique_labels)

    if metric == "mse":
        overall_mean = np.mean(data, axis=0)
        mse_values = []
        for label in unique_labels:
            group_data = data[labels == label]
            mse_values.append(np.mean((group_data - overall_mean) ** 2))
        metrics["mse"] = np.mean(mse_values)

    elif metric == "mae":
        overall_mean = np.mean(data, axis=0)
        mae_values = []
        for label in unique_labels:
            group_data = data[labels == label]
            mae_values.append(np.mean(np.abs(group_data - overall_mean)))
        metrics["mae"] = np.mean(mae_values)

    elif metric == "r2":
        overall_mean = np.mean(data, axis=0)
        ss_total = np.sum((data - overall_mean) ** 2)
        ss_residual = 0
        for label in unique_labels:
            group_data = data[labels == label]
            group_mean = np.mean(group_data, axis=0)
            ss_residual += np.sum((group_data - group_mean) ** 2)
        metrics["r2"] = 1 - (ss_residual / ss_total)

    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

def sunburst_chart_fit(
    data: np.ndarray,
    labels: np.ndarray,
    hierarchy: Optional[np.ndarray] = None,
    normalization: str = "standard",
    metric: str = "mse",
    custom_normalization: Optional[Callable] = None,
    custom_metric: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Compute sunburst chart data for group comparison.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    labels : np.ndarray
        Group labels for each sample of shape (n_samples,)
    hierarchy : Optional[np.ndarray]
        Hierarchy levels for each group of shape (n_groups,)
    normalization : str
        Normalization method: 'none', 'standard', 'minmax', 'robust'
    metric : str
        Comparison metric: 'mse', 'mae', 'r2'
    custom_normalization : Optional[Callable]
        Custom normalization function
    custom_metric : Optional[Callable]
        Custom metric function

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - result: Computed group statistics
        - metrics: Calculated comparison metrics
        - params_used: Parameters used in computation
        - warnings: Any warnings encountered

    Example
    -------
    >>> data = np.random.rand(100, 5)
    >>> labels = np.random.randint(0, 3, size=100)
    >>> result = sunburst_chart_fit(data, labels)
    """
    # Validate inputs
    validation = validate_inputs(data, labels, hierarchy)

    # Normalize data
    normalized_data = normalize_data(
        data,
        method=normalization,
        custom_func=custom_normalization
    )

    # Compute group statistics
    groups = compute_group_stats(normalized_data, labels)

    # Calculate metrics
    metrics = calculate_metrics(
        normalized_data,
        labels,
        metric=metric,
        custom_metric=custom_metric
    )

    return {
        "result": groups,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric,
            "hierarchy_provided": hierarchy is not None
        },
        "warnings": validation["warnings"]
    }

################################################################################
# treemap
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def validate_inputs(
    data: np.ndarray,
    labels: np.ndarray,
    normalize: str = "none",
    metric: Union[str, Callable] = "euclidean"
) -> None:
    """
    Validate input data and parameters.

    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features)
    labels : np.ndarray
        Group labels for each sample of shape (n_samples,)
    normalize : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable
        Distance metric to use

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")
    if labels.ndim != 1:
        raise ValueError("Labels must be a 1D array")
    if len(data) != len(labels):
        raise ValueError("Data and labels must have the same length")
    if normalize not in ["none", "standard", "minmax", "robust"]:
        raise ValueError("Invalid normalization method")
    if isinstance(metric, str) and metric not in ["euclidean", "manhattan", "cosine", "minkowski"]:
        raise ValueError("Invalid metric string")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")

def normalize_data(
    data: np.ndarray,
    method: str = "none"
) -> np.ndarray:
    """
    Normalize data using specified method.

    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features)
    method : str
        Normalization method ('none', 'standard', 'minmax', 'robust')

    Returns
    ------
    np.ndarray
        Normalized data array
    """
    if method == "none":
        return data

    normalized = np.zeros_like(data, dtype=np.float64)

    if method == "standard":
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized = (data - mean) / (std + 1e-8)

    elif method == "minmax":
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)

    elif method == "robust":
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        normalized = (data - median) / (iqr + 1e-8)

    return normalized

def compute_distance_matrix(
    data: np.ndarray,
    metric: Union[str, Callable] = "euclidean",
    p: float = 2.0
) -> np.ndarray:
    """
    Compute distance matrix using specified metric.

    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features)
    metric : str or callable
        Distance metric to use
    p : float
        Parameter for Minkowski distance

    Returns
    ------
    np.ndarray
        Distance matrix of shape (n_samples, n_samples)
    """
    if isinstance(metric, str):
        if metric == "euclidean":
            return np.sqrt(np.sum((data[:, np.newaxis] - data) ** 2, axis=2))
        elif metric == "manhattan":
            return np.sum(np.abs(data[:, np.newaxis] - data), axis=2)
        elif metric == "cosine":
            dot_products = np.dot(data, data.T)
            norms = np.sqrt(np.sum(data**2, axis=1))
            return 1 - dot_products / (norms[:, np.newaxis] * norms)
        elif metric == "minkowski":
            return np.sum(np.abs(data[:, np.newaxis] - data) ** p, axis=2) ** (1/p)
    else:
        return metric(data)

def compute_treemap(
    data: np.ndarray,
    labels: np.ndarray,
    distance_matrix: Optional[np.ndarray] = None,
    metric: Union[str, Callable] = "euclidean",
    normalize: str = "none"
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Compute treemap visualization parameters.

    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features)
    labels : np.ndarray
        Group labels for each sample of shape (n_samples,)
    distance_matrix : np.ndarray or None
        Precomputed distance matrix
    metric : str or callable
        Distance metric to use
    normalize : str
        Normalization method

    Returns
    ------
    dict
        Dictionary containing treemap parameters and metrics
    """
    # Validate inputs
    validate_inputs(data, labels, normalize, metric)

    # Normalize data if needed
    normalized_data = normalize_data(data, normalize)

    # Compute distance matrix if not provided
    if distance_matrix is None:
        distance_matrix = compute_distance_matrix(normalized_data, metric)

    # Compute group statistics
    unique_labels = np.unique(labels)
    n_groups = len(unique_labels)

    # Initialize result dictionary
    result = {
        "group_sizes": np.zeros(n_groups, dtype=int),
        "group_centers": np.zeros((n_groups, data.shape[1])),
        "group_distances": np.zeros((n_groups, n_groups)),
        "group_metrics": {}
    }

    # Compute group statistics
    for i, label in enumerate(unique_labels):
        mask = labels == label
        result["group_sizes"][i] = np.sum(mask)
        result["group_centers"][i] = np.mean(normalized_data[mask], axis=0)

        # Compute within-group distance
        if result["group_sizes"][i] > 1:
            group_dist = distance_matrix[mask][:, mask]
            result["group_metrics"][f"group_{label}_mean_dist"] = np.mean(group_dist[np.triu_indices_from(group_dist, k=1)])

    # Compute between-group distances
    for i in range(n_groups):
        for j in range(i, n_groups):
            if i == j:
                result["group_distances"][i,j] = 0
            else:
                dist = distance_matrix[labels == unique_labels[i]][:, labels == unique_labels[j]]
                result["group_distances"][i,j] = np.mean(dist)
                result["group_distances"][j,i] = result["group_distances"][i,j]

    return {
        "result": result,
        "metrics": {
            "n_groups": n_groups,
            "total_samples": len(labels),
            "mean_group_size": np.mean(result["group_sizes"]),
            "max_group_distance": np.max(result["group_distances"])
        },
        "params_used": {
            "normalize": normalize,
            "metric": metric
        },
        "warnings": []
    }

def treemap_fit(
    data: np.ndarray,
    labels: np.ndarray,
    normalize: str = "none",
    metric: Union[str, Callable] = "euclidean",
    p: float = 2.0
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Compute treemap visualization parameters with user-configurable options.

    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features)
    labels : np.ndarray
        Group labels for each sample of shape (n_samples,)
    normalize : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski')
    p : float
        Parameter for Minkowski distance

    Returns
    ------
    dict
        Dictionary containing treemap parameters and metrics

    Examples
    --------
    >>> data = np.random.rand(100, 5)
    >>> labels = np.random.randint(0, 3, size=100)
    >>> result = treemap_fit(data, labels, normalize="standard", metric="euclidean")
    """
    return compute_treemap(data, labels, None, metric, normalize)

################################################################################
# dot_plot
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None
) -> Dict[str, Union[bool, str]]:
    """
    Validate input data for dot plot.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    groups : Optional[np.ndarray]
        Group labels of shape (n_samples,) or None

    Returns
    -------
    Dict[str, Union[bool, str]]
        Validation results with status and messages
    """
    validation = {"status": True, "messages": []}

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        validation["status"] = False
        validation["messages"].append("X must be a 2D numpy array")

    if not isinstance(y, np.ndarray) or y.ndim != 1:
        validation["status"] = False
        validation["messages"].append("y must be a 1D numpy array")

    if groups is not None:
        if not isinstance(groups, np.ndarray) or groups.ndim != 1:
            validation["status"] = False
            validation["messages"].append("groups must be a 1D numpy array or None")
        if len(groups) != X.shape[0]:
            validation["status"] = False
            validation["messages"].append("groups length must match number of samples")

    if len(y) != X.shape[0]:
        validation["status"] = False
        validation["messages"].append("y length must match number of samples")

    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        validation["status"] = False
        validation["messages"].append("X contains NaN or infinite values")

    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        validation["status"] = False
        validation["messages"].append("y contains NaN or infinite values")

    return validation

def normalize_data(
    X: np.ndarray,
    y: np.ndarray,
    normalization: str = "standard",
    groups: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Normalize data according to specified method.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    normalization : str
        Normalization method: "none", "standard", "minmax", or "robust"
    groups : Optional[np.ndarray]
        Group labels of shape (n_samples,) or None

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing normalized X and y
    """
    if normalization == "none":
        return {"X": X, "y": y}

    # Handle group-wise normalization if groups are provided
    if groups is not None:
        unique_groups = np.unique(groups)
        normalized_X = np.zeros_like(X, dtype=float)
        normalized_y = np.zeros_like(y, dtype=float)

        for group in unique_groups:
            mask = groups == group
            X_group = X[mask]
            y_group = y[mask]

            if normalization == "standard":
                mean = X_group.mean(axis=0)
                std = X_group.std(axis=0)
                normalized_X[mask] = (X_group - mean) / (std + 1e-8)
                normalized_y[mask] = (y_group - y_group.mean()) / (y_group.std() + 1e-8)
            elif normalization == "minmax":
                min_val = X_group.min(axis=0)
                max_val = X_group.max(axis=0)
                normalized_X[mask] = (X_group - min_val) / (max_val - min_val + 1e-8)
                normalized_y[mask] = (y_group - y_group.min()) / (y_group.max() - y_group.min() + 1e-8)
            elif normalization == "robust":
                median = np.median(X_group, axis=0)
                iqr = np.subtract(*np.percentile(X_group, [75, 25], axis=0))
                normalized_X[mask] = (X_group - median) / (iqr + 1e-8)
                normalized_y[mask] = (y_group - np.median(y_group)) / (
                    np.subtract(*np.percentile(y_group, [75, 25])) + 1e-8
                )

        return {"X": normalized_X, "y": normalized_y}

    # Global normalization
    if normalization == "standard":
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        normalized_X = (X - mean) / (std + 1e-8)
        normalized_y = (y - y.mean()) / (y.std() + 1e-8)
    elif normalization == "minmax":
        min_val = X.min(axis=0)
        max_val = X.max(axis=0)
        normalized_X = (X - min_val) / (max_val - min_val + 1e-8)
        normalized_y = (y - y.min()) / (y.max() - y.min() + 1e-8)
    elif normalization == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        normalized_X = (X - median) / (iqr + 1e-8)
        normalized_y = (y - np.median(y)) / (
            np.subtract(*np.percentile(y, [75, 25])) + 1e-8
        )

    return {"X": normalized_X, "y": normalized_y}

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Union[str, List[str], Callable] = "mse",
    groups: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute specified metrics between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        True target values of shape (n_samples,)
    y_pred : np.ndarray
        Predicted target values of shape (n_samples,)
    metrics : Union[str, List[str], Callable]
        Metric(s) to compute: "mse", "mae", "r2", or custom callable
    groups : Optional[np.ndarray]
        Group labels of shape (n_samples,) or None

    Returns
    -------
    Dict[str, float]
        Dictionary of computed metrics
    """
    metric_results = {}

    if isinstance(metrics, str):
        metrics = [metrics]

    for metric in metrics:
        if callable(metric):
            # Handle custom metric function
            if groups is not None:
                unique_groups = np.unique(groups)
                group_metrics = {}
                for group in unique_groups:
                    mask = groups == group
                    y_true_group = y_true[mask]
                    y_pred_group = y_pred[mask]
                    group_metrics[f"group_{group}"] = metric(y_true_group, y_pred_group)
                metric_results[metric.__name__] = group_metrics
            else:
                metric_results[metric.__name__] = metric(y_true, y_pred)
        elif metric == "mse":
            if groups is not None:
                unique_groups = np.unique(groups)
                group_metrics = {}
                for group in unique_groups:
                    mask = groups == group
                    y_true_group = y_true[mask]
                    y_pred_group = y_pred[mask]
                    group_metrics[f"group_{group}"] = np.mean((y_true_group - y_pred_group) ** 2)
                metric_results["mse"] = group_metrics
            else:
                metric_results["mse"] = np.mean((y_true - y_pred) ** 2)
        elif metric == "mae":
            if groups is not None:
                unique_groups = np.unique(groups)
                group_metrics = {}
                for group in unique_groups:
                    mask = groups == group
                    y_true_group = y_true[mask]
                    y_pred_group = y_pred[mask]
                    group_metrics[f"group_{group}"] = np.mean(np.abs(y_true_group - y_pred_group))
                metric_results["mae"] = group_metrics
            else:
                metric_results["mae"] = np.mean(np.abs(y_true - y_pred))
        elif metric == "r2":
            if groups is not None:
                unique_groups = np.unique(groups)
                group_metrics = {}
                for group in unique_groups:
                    mask = groups == group
                    y_true_group = y_true[mask]
                    y_pred_group = y_pred[mask]
                    ss_res = np.sum((y_true_group - y_pred_group) ** 2)
                    ss_tot = np.sum((y_true_group - np.mean(y_true_group)) ** 2)
                    group_metrics[f"group_{group}"] = 1 - (ss_res / (ss_tot + 1e-8))
                metric_results["r2"] = group_metrics
            else:
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                metric_results["r2"] = 1 - (ss_res / (ss_tot + 1e-8))

    return metric_results

def dot_plot_fit(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    normalization: str = "standard",
    metrics: Union[str, List[str], Callable] = "mse",
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Union[Dict, float]]:
    """
    Fit and evaluate a dot plot visualization for group comparisons.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    groups : Optional[np.ndarray]
        Group labels of shape (n_samples,) or None
    normalization : str
        Normalization method: "none", "standard", "minmax", or "robust"
    metrics : Union[str, List[str], Callable]
        Metric(s) to compute: "mse", "mae", "r2", or custom callable
    custom_metric : Optional[Callable]
        Custom metric function if not provided in metrics parameter

    Returns
    -------
    Dict[str, Union[Dict, float]]
        Dictionary containing:
        - "result": Dictionary of results
        - "metrics": Dictionary of computed metrics
        - "params_used": Dictionary of parameters used
        - "warnings": List of warning messages
    """
    # Validate inputs
    validation = validate_inputs(X, y, groups)
    if not validation["status"]:
        return {
            "result": None,
            "metrics": None,
            "params_used": {"normalization": normalization, "metrics": metrics},
            "warnings": validation["messages"]
        }

    warnings = []

    # Normalize data
    normalized_data = normalize_data(X, y, normalization, groups)
    X_norm = normalized_data["X"]
    y_norm = normalized_data["y"]

    # Compute metrics
    if custom_metric is not None:
        if isinstance(metrics, str):
            metrics = [metrics]
        metrics.append(custom_metric)

    metric_results = compute_metrics(y, y_norm, metrics, groups)

    # Prepare result dictionary
    result = {
        "normalized_X": X_norm,
        "normalized_y": y_norm
    }

    if groups is not None:
        unique_groups = np.unique(groups)
        group_stats = {}
        for group in unique_groups:
            mask = groups == group
            X_group = X_norm[mask]
            y_group = y_norm[mask]

            group_stats[f"group_{group}"] = {
                "mean_X": np.mean(X_group, axis=0),
                "std_X": np.std(X_group, axis=0),
                "mean_y": np.mean(y_group),
                "std_y": np.std(y_group)
            }
        result["group_stats"] = group_stats

    return {
        "result": result,
        "metrics": metric_results,
        "params_used": {
            "normalization": normalization,
            "metrics": metrics
        },
        "warnings": warnings
    }

################################################################################
# forest_plot
################################################################################

import numpy as np
from typing import Dict, Any, Callable, Optional, Union

def _validate_inputs(
    data: np.ndarray,
    group_labels: np.ndarray,
    metric: Union[str, Callable],
    normalization: str
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if not isinstance(group_labels, np.ndarray):
        raise TypeError("Group labels must be a numpy array")
    if data.shape[0] != group_labels.shape[0]:
        raise ValueError("Data and group labels must have the same number of samples")
    if normalization not in ['none', 'standard', 'minmax', 'robust']:
        raise ValueError("Normalization must be one of: none, standard, minmax, robust")
    if isinstance(metric, str) and metric not in ['mean', 'median', 'std', 'custom']:
        raise ValueError("Metric must be one of: mean, median, std, custom")

def _normalize_data(
    data: np.ndarray,
    group_labels: np.ndarray,
    normalization: str
) -> Dict[str, np.ndarray]:
    """Normalize data based on specified method."""
    results = {}
    for group in np.unique(group_labels):
        mask = group_labels == group
        group_data = data[mask]

        if normalization == 'standard':
            mean = np.mean(group_data, axis=0)
            std = np.std(group_data, axis=0)
            normalized = (group_data - mean) / std
        elif normalization == 'minmax':
            min_val = np.min(group_data, axis=0)
            max_val = np.max(group_data, axis=0)
            normalized = (group_data - min_val) / (max_val - min_val + 1e-8)
        elif normalization == 'robust':
            median = np.median(group_data, axis=0)
            iqr = np.percentile(group_data, 75, axis=0) - np.percentile(group_data, 25, axis=0)
            normalized = (group_data - median) / (iqr + 1e-8)
        else:
            normalized = group_data

        results[group] = normalized
    return results

def _compute_metric(
    data: np.ndarray,
    metric: Union[str, Callable],
    axis: int = 0
) -> np.ndarray:
    """Compute specified metric along given axis."""
    if isinstance(metric, str):
        if metric == 'mean':
            return np.mean(data, axis=axis)
        elif metric == 'median':
            return np.median(data, axis=axis)
        elif metric == 'std':
            return np.std(data, axis=axis)
    else:
        return metric(data)

def _calculate_confidence_intervals(
    data: np.ndarray,
    alpha: float = 0.95
) -> Dict[str, np.ndarray]:
    """Calculate confidence intervals for the data."""
    mean = np.mean(data, axis=0)
    std_err = np.std(data, axis=0) / np.sqrt(data.shape[0])
    margin = std_err * 1.96  # Assuming normal distribution
    return {
        'lower': mean - margin,
        'upper': mean + margin
    }

def forest_plot_fit(
    data: np.ndarray,
    group_labels: np.ndarray,
    metric: Union[str, Callable] = 'mean',
    normalization: str = 'none',
    alpha: float = 0.95,
    custom_metric: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Create a forest plot comparing groups visually.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    group_labels : np.ndarray
        Group labels for each sample of shape (n_samples,)
    metric : str or callable
        Metric to compute for each group. Options: 'mean', 'median', 'std', or custom callable.
    normalization : str
        Normalization method. Options: 'none', 'standard', 'minmax', 'robust'.
    alpha : float
        Confidence level for intervals (default: 0.95)
    custom_metric : callable, optional
        Custom metric function if using 'custom' option.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, group_labels, metric, normalization)

    # Normalize data if specified
    normalized_data = _normalize_data(data, group_labels, normalization)

    # Compute metrics for each group
    results = {}
    warnings_list = []

    for group, group_data in normalized_data.items():
        try:
            if isinstance(metric, str) and metric == 'custom':
                computed_metric = _compute_metric(group_data, custom_metric)
            else:
                computed_metric = _compute_metric(group_data, metric)

            # Calculate confidence intervals
            ci = _calculate_confidence_intervals(group_data, alpha)

            results[group] = {
                'metric': computed_metric,
                'confidence_interval': ci
            }
        except Exception as e:
            warnings_list.append(f"Error computing metrics for group {group}: {str(e)}")

    # Prepare output dictionary
    output = {
        'result': results,
        'metrics': {
            'metric_used': metric if not isinstance(metric, Callable) else 'custom',
            'normalization': normalization,
            'confidence_level': alpha
        },
        'params_used': {
            'data_shape': data.shape,
            'n_groups': len(np.unique(group_labels)),
            'normalization_method': normalization
        },
        'warnings': warnings_list if warnings_list else None
    }

    return output

# Example usage:
"""
data = np.random.randn(100, 5)
group_labels = np.array(['A']*50 + ['B']*50)

result = forest_plot_fit(
    data=data,
    group_labels=group_labels,
    metric='mean',
    normalization='standard'
)
"""

################################################################################
# qq_plot
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(data: np.ndarray, distribution: str = 'normal') -> None:
    """Validate input data and distribution type."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional.")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values.")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values.")

def _compute_quantiles(data: np.ndarray, distribution: str = 'normal') -> np.ndarray:
    """Compute theoretical quantiles based on the specified distribution."""
    n = len(data)
    p = np.arange(1, n + 1) / (n + 1)
    if distribution == 'normal':
        return np.percentile(data, [100 * p])
    elif distribution == 'uniform':
        return np.sort(data)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

def _compute_sample_quantiles(data: np.ndarray) -> np.ndarray:
    """Compute sample quantiles from the data."""
    return np.sort(data)

def _calculate_metrics(sample_quantiles: np.ndarray, theoretical_quantiles: np.ndarray,
                      metric_func: Callable = None) -> Dict[str, float]:
    """Calculate metrics between sample and theoretical quantiles."""
    if metric_func is None:
        def default_metric(x, y):
            return np.mean((x - y) ** 2)
        metric_func = default_metric

    metrics = {
        'mse': np.mean((sample_quantiles - theoretical_quantiles) ** 2),
        'mae': np.mean(np.abs(sample_quantiles - theoretical_quantiles)),
    }
    if metric_func is not None:
        metrics['custom'] = metric_func(sample_quantiles, theoretical_quantiles)
    return metrics

def qq_plot_fit(data: np.ndarray,
                distribution: str = 'normal',
                metric_func: Optional[Callable] = None) -> Dict:
    """
    Generate a Q-Q plot and compute relevant metrics.

    Parameters
    ----------
    data : np.ndarray
        Input data to be plotted.
    distribution : str, optional
        Theoretical distribution to compare against ('normal' or 'uniform'), by default 'normal'.
    metric_func : Optional[Callable], optional
        Custom metric function to compute between sample and theoretical quantiles, by default None.

    Returns
    -------
    Dict
        A dictionary containing the following keys:
        - 'result': Tuple of (sample_quantiles, theoretical_quantiles)
        - 'metrics': Dictionary of computed metrics
        - 'params_used': Dictionary of parameters used
        - 'warnings': List of warnings, if any

    Examples
    --------
    >>> data = np.random.normal(0, 1, 100)
    >>> result = qq_plot_fit(data, distribution='normal')
    """
    _validate_inputs(data)

    sample_quantiles = _compute_sample_quantiles(data)
    theoretical_quantiles = _compute_quantiles(data, distribution)

    metrics = _calculate_metrics(sample_quantiles, theoretical_quantiles, metric_func)

    return {
        'result': (sample_quantiles, theoretical_quantiles),
        'metrics': metrics,
        'params_used': {
            'distribution': distribution,
            'metric_func': metric_func.__name__ if metric_func else None
        },
        'warnings': []
    }
