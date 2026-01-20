"""
Quantix – Module courbes_roc_pr
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# courbe_roc
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(y_true: np.ndarray, y_scores: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_scores, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_scores.shape:
        raise ValueError("y_true and y_scores must have the same shape")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or Inf values")
    if np.any(np.isnan(y_scores)) or np.any(np.isinf(y_scores)):
        raise ValueError("y_scores contains NaN or Inf values")
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0 and 1")

def _compute_tpr_fpr(y_true: np.ndarray, y_scores: np.ndarray) -> tuple:
    """Compute true positive rate and false positive rate."""
    thresholds = np.unique(y_scores)
    tpr = np.zeros_like(thresholds)
    fpr = np.zeros_like(thresholds)

    for i, threshold in enumerate(thresholds):
        tp = np.sum((y_scores >= threshold) & (y_true == 1))
        fp = np.sum((y_scores >= threshold) & (y_true == 0))
        tn = np.sum((y_scores < threshold) & (y_true == 0))
        fn = np.sum((y_scores < threshold) & (y_true == 1))

        tpr[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr[i] = fp / (fp + tn) if (fp + tn) > 0 else 0

    return tpr, fpr

def _compute_auc(tpr: np.ndarray, fpr: np.ndarray) -> float:
    """Compute Area Under the Curve."""
    return np.trapz(tpr, fpr)

def _compute_metrics(y_true: np.ndarray, y_scores: np.ndarray,
                     tpr: np.ndarray, fpr: np.ndarray) -> Dict[str, float]:
    """Compute various metrics."""
    auc = _compute_auc(tpr, fpr)
    return {
        "auc": auc,
        "max_tpr_at_min_fpr": np.max(tpr[fpr == np.min(fpr)]),
        "min_fpr_at_max_tpr": np.min(fpr[tpr == np.max(tpr)])
    }

def courbe_roc_fit(y_true: np.ndarray, y_scores: np.ndarray,
                   normalization: str = "none",
                   metric: Union[str, Callable] = "auc") -> Dict:
    """
    Compute ROC curve and related metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_scores : np.ndarray
        Predicted scores.
    normalization : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    metric : str or callable, optional
        Metric to compute ("auc" or custom callable).

    Returns
    -------
    dict
        Dictionary containing results, metrics, parameters used and warnings.
    """
    _validate_inputs(y_true, y_scores)

    # Normalization
    if normalization == "standard":
        y_scores = (y_scores - np.mean(y_scores)) / np.std(y_scores)
    elif normalization == "minmax":
        y_scores = (y_scores - np.min(y_scores)) / (np.max(y_scores) - np.min(y_scores))
    elif normalization == "robust":
        y_scores = (y_scores - np.median(y_scores)) / (np.percentile(y_scores, 75) - np.percentile(y_scores, 25))

    tpr, fpr = _compute_tpr_fpr(y_true, y_scores)

    # Metrics
    if metric == "auc":
        metrics = _compute_metrics(y_true, y_scores, tpr, fpr)
    else:
        if callable(metric):
            metrics = {"custom_metric": metric(y_true, y_scores)}
        else:
            raise ValueError("Invalid metric specified")

    return {
        "result": {"tpr": tpr, "fpr": fpr},
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric
        },
        "warnings": []
    }

# Example usage:
"""
y_true = np.array([0, 1, 1, 0, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.9])
result = courbe_roc_fit(y_true, y_scores)
"""

################################################################################
# courbe_pr
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(y_true: np.ndarray, y_scores: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_scores, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_scores.shape:
        raise ValueError("y_true and y_scores must have the same shape")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or inf values")
    if np.any(np.isnan(y_scores)) or np.any(np.isinf(y_scores)):
        raise ValueError("y_scores contains NaN or inf values")
    if not np.issubdtype(y_true.dtype, np.integer):
        raise ValueError("y_true must contain integer values")
    if not (np.all(y_true == 0) or np.all(y_true == 1)):
        raise ValueError("y_true must be binary (0 or 1)")

def _compute_precision_recall(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Dict[str, np.ndarray]:
    """Compute precision and recall values."""
    # Sort scores in descending order
    desc_score_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[desc_score_indices]
    y_scores_sorted = y_scores[desc_score_indices]

    # Compute precision and recall
    pos_count = np.sum(y_true == 1)
    precisions = []
    recalls = []

    for i in range(len(y_true_sorted)):
        tp = np.sum(y_true_sorted[:i+1] == 1)
        fp = i + 1 - tp
        fn = pos_count - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / pos_count if pos_count > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    return {
        'precisions': np.array(precisions),
        'recalls': np.array(recalls)
    }

def _apply_normalization(
    y_scores: np.ndarray,
    normalization: Optional[str] = None
) -> np.ndarray:
    """Apply specified normalization to scores."""
    if normalization is None:
        return y_scores

    normalized = np.array(y_scores, dtype=np.float64)

    if normalization == 'standard':
        mean = np.mean(normalized)
        std = np.std(normalized)
        if std > 0:
            normalized = (normalized - mean) / std
    elif normalization == 'minmax':
        min_val = np.min(normalized)
        max_val = np.max(normalized)
        if min_val != max_val:
            normalized = (normalized - min_val) / (max_val - min_val)
    elif normalization == 'robust':
        median = np.median(normalized)
        iqr = np.percentile(normalized, 75) - np.percentile(normalized, 25)
        if iqr > 0:
            normalized = (normalized - median) / iqr

    return normalized

def _compute_metrics(
    precisions: np.ndarray,
    recalls: np.ndarray
) -> Dict[str, float]:
    """Compute various metrics from precision-recall curve."""
    # Compute average precision
    ap = np.trapz(precisions, recalls) / (recalls[-1] + 1e-8)

    # Find point closest to (0,1) corner
    distances = np.sqrt((precisions - 1)**2 + (recalls - 0)**2)
    best_idx = np.argmin(distances)

    return {
        'average_precision': ap,
        'max_f1_score': np.max(2 * precisions * recalls / (precisions + recalls + 1e-8)),
        'closest_to_corner_precision': precisions[best_idx],
        'closest_to_corner_recall': recalls[best_idx]
    }

def courbe_pr_fit(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    normalization: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict[str, np.ndarray], Dict[str, float], Dict[str, str], list]]:
    """
    Compute Precision-Recall curve and associated metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true binary labels (0 or 1)
    y_scores : np.ndarray
        Array of predicted scores/probabilities
    normalization : str, optional
        Normalization method for scores ('none', 'standard', 'minmax', 'robust')
    custom_metric : callable, optional
        Custom metric function taking (precisions, recalls) and returning a float

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': precision and recall values
        - 'metrics': computed metrics
        - 'params_used': parameters used in computation
        - 'warnings': any warnings generated

    Example
    -------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_scores = np.array([0.1, 0.9, 0.8, 0.2])
    >>> result = courbe_pr_fit(y_true, y_scores)
    """
    # Validate inputs
    _validate_inputs(y_true, y_scores)

    warnings = []

    # Apply normalization if specified
    normalized_scores = _apply_normalization(y_scores, normalization)
    if normalization:
        warnings.append(f"Applied {normalization} normalization to scores")

    # Compute precision-recall values
    pr_values = _compute_precision_recall(y_true, normalized_scores)

    # Compute metrics
    metrics = _compute_metrics(pr_values['precisions'], pr_values['recalls'])

    # Add custom metric if provided
    if custom_metric is not None:
        try:
            metrics['custom_metric'] = custom_metric(pr_values['precisions'], pr_values['recalls'])
        except Exception as e:
            warnings.append(f"Custom metric computation failed: {str(e)}")

    return {
        'result': pr_values,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'custom_metric': custom_metric.__name__ if custom_metric else None
        },
        'warnings': warnings
    }

################################################################################
# taux_vrai_positif
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(y_true: np.ndarray, y_scores: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_scores, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_scores.shape:
        raise ValueError("y_true and y_scores must have the same shape")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or Inf values")
    if np.any(np.isnan(y_scores)) or np.any(np.isinf(y_scores)):
        raise ValueError("y_scores contains NaN or Inf values")
    if not np.issubdtype(y_true.dtype, np.integer):
        raise ValueError("y_true must contain integer values")

def _compute_tpr(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float
) -> float:
    """Compute True Positive Rate (TPR) for a given threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return tpr

def taux_vrai_positif_fit(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, float], list]]:
    """
    Compute True Positive Rate (TPR) for given thresholds.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true binary labels (0 or 1).
    y_scores : np.ndarray
        Array of predicted scores/probabilities.
    thresholds : Optional[np.ndarray], default=None
        Array of thresholds to evaluate. If None, uses 100 linearly spaced values.
    custom_metric : Optional[Callable], default=None
        Custom metric function to compute alongside TPR.

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], Dict[str, float], list]]
        Dictionary containing:
        - "result": List of TPR values
        - "metrics": Dictionary of additional metrics (if custom_metric provided)
        - "params_used": Dictionary of parameters used
        - "warnings": List of warnings (if any)

    Examples
    --------
    >>> y_true = np.array([1, 0, 1, 1])
    >>> y_scores = np.array([0.9, 0.2, 0.8, 0.7])
    >>> result = taux_vrai_positif_fit(y_true, y_scores)
    """
    _validate_inputs(y_true, y_scores)

    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)

    tpr_values = [_compute_tpr(y_true, y_scores, threshold) for threshold in thresholds]
    metrics = {}
    if custom_metric is not None:
        metrics["custom"] = [custom_metric(y_true, (y_scores >= threshold).astype(int)) for threshold in thresholds]

    return {
        "result": tpr_values,
        "metrics": metrics,
        "params_used": {
            "thresholds": thresholds.tolist(),
            "custom_metric_provided": custom_metric is not None
        },
        "warnings": []
    }

################################################################################
# taux_faux_positif
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(y_true: np.ndarray, y_scores: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_scores, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_scores.shape:
        raise ValueError("y_true and y_scores must have the same shape")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or inf values")
    if np.any(np.isnan(y_scores)) or np.any(np.isinf(y_scores)):
        raise ValueError("y_scores contains NaN or inf values")
    if not np.issubdtype(y_true.dtype, np.integer):
        raise ValueError("y_true must contain integer values")

def _compute_fpr(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    normalize: str = "none",
    custom_normalize: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Union[np.ndarray, float]]:
    """Compute false positive rate."""
    if normalize == "none":
        pass
    elif normalize == "standard":
        y_scores = (y_scores - np.mean(y_scores)) / np.std(y_scores)
    elif normalize == "minmax":
        y_scores = (y_scores - np.min(y_scores)) / (np.max(y_scores) - np.min(y_scores))
    elif normalize == "robust":
        y_scores = (y_scores - np.median(y_scores)) / (np.percentile(y_scores, 75) - np.percentile(y_scores, 25))
    elif custom_normalize is not None:
        y_scores = custom_normalize(y_scores)
    else:
        raise ValueError("Invalid normalization method")

    # Sort scores and get corresponding true labels
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]

    # Compute cumulative sums
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)

    # Compute FPR
    fpr = fp / np.sum(1 - y_true)

    return {
        "fpr": fpr,
        "tp": tp,
        "fp": fp
    }

def taux_faux_positif_fit(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    normalize: str = "none",
    custom_normalize: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Compute false positive rate for ROC curve.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true binary labels (0 or 1)
    y_scores : np.ndarray
        Array of predicted scores/probabilities
    normalize : str, optional
        Normalization method for scores ("none", "standard", "minmax", "robust")
    custom_normalize : callable, optional
        Custom normalization function

    Returns
    -------
    dict
        Dictionary containing:
        - "fpr": False positive rate values
        - "tp": True positives at each threshold
        - "fp": False positives at each threshold

    Examples
    --------
    >>> y_true = np.array([1, 0, 1, 1, 0])
    >>> y_scores = np.array([0.9, 0.2, 0.8, 0.7, 0.1])
    >>> result = taux_faux_positif_fit(y_true, y_scores)
    """
    _validate_inputs(y_true, y_scores)

    result = _compute_fpr(
        y_true=y_true,
        y_scores=y_scores,
        normalize=normalize,
        custom_normalize=custom_normalize
    )

    return {
        "result": result["fpr"],
        "metrics": {},
        "params_used": {
            "normalize": normalize,
            "custom_normalize": custom_normalize is not None
        },
        "warnings": []
    }

################################################################################
# precision
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def precision_fit(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    *,
    normalization: str = 'none',
    metric: Union[str, Callable] = 'binary_crossentropy',
    solver: str = 'closed_form',
    custom_metric: Optional[Callable] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Compute precision curve and related metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_scores : np.ndarray
        Predicted scores/probabilities.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to optimize ('binary_crossentropy', 'mse', etc.) or custom callable.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent').
    custom_metric : callable, optional
        Custom metric function.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': Computed precision values
        - 'metrics': Calculated metrics
        - 'params_used': Parameters used in computation
        - 'warnings': Any warnings generated

    Example
    -------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_scores = np.array([0.1, 0.9, 0.8, 0.3])
    >>> result = precision_fit(y_true, y_scores)
    """
    # Validate inputs
    _validate_inputs(y_true, y_scores)

    # Normalize scores if needed
    normalized_scores = _apply_normalization(y_scores, normalization)

    # Choose metric function
    if isinstance(metric, str):
        metric_func = _get_metric_function(metric)
    else:
        metric_func = metric

    # Choose solver
    if solver == 'closed_form':
        precision_values, params = _closed_form_solver(y_true, normalized_scores)
    else:
        raise ValueError(f"Solver {solver} not implemented")

    # Calculate metrics
    metrics = _calculate_metrics(y_true, y_scores, precision_values, metric_func)

    # Prepare output
    result = {
        'result': precision_values,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'solver': solver
        },
        'warnings': []
    }

    return result

def _validate_inputs(y_true: np.ndarray, y_scores: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_scores, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_scores.shape:
        raise ValueError("y_true and y_scores must have the same shape")
    if np.any((y_true != 0) & (y_true != 1)):
        raise ValueError("y_true must contain only 0s and 1s")
    if np.any((y_scores < 0) | (y_scores > 1)):
        raise ValueError("y_scores must be between 0 and 1")

def _apply_normalization(scores: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to scores."""
    if method == 'none':
        return scores
    elif method == 'standard':
        return (scores - np.mean(scores)) / np.std(scores)
    elif method == 'minmax':
        return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    elif method == 'robust':
        return (scores - np.median(scores)) / (np.percentile(scores, 75) - np.percentile(scores, 25))
    else:
        raise ValueError(f"Normalization method {method} not recognized")

def _get_metric_function(metric_name: str) -> Callable:
    """Get metric function based on name."""
    metrics = {
        'binary_crossentropy': _binary_crossentropy,
        'mse': _mean_squared_error
    }
    if metric_name not in metrics:
        raise ValueError(f"Metric {metric_name} not recognized")
    return metrics[metric_name]

def _closed_form_solver(y_true: np.ndarray, y_scores: np.ndarray) -> tuple:
    """Closed form solution for precision calculation."""
    # Sort scores in descending order
    sorted_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]

    # Calculate precision at each threshold
    precision_values = []
    true_positives = 0
    false_positives = 0

    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            true_positives += 1
        else:
            false_positives += 1

        if (true_positives + false_positives) > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0
        precision_values.append(precision)

    return np.array(precision_values), {}

def _calculate_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    precision_values: np.ndarray,
    metric_func: Callable
) -> Dict[str, float]:
    """Calculate various metrics."""
    metrics = {}

    # Calculate metric
    if callable(metric_func):
        metrics['custom_metric'] = metric_func(y_true, y_scores)

    # Calculate AUC (Area Under Curve)
    thresholds = np.unique(y_scores)[::-1]
    thresholds = np.concatenate(([1], thresholds, [0]))

    # Calculate precision at each threshold
    precisions = []
    for i in range(len(thresholds)-1):
        mask = y_scores >= thresholds[i+1]
        if np.sum(mask) > 0:
            precisions.append(np.mean(y_true[mask] == 1))
        else:
            precisions.append(0)

    # Calculate AUC using trapezoidal rule
    auc = np.trapz(precisions, x=thresholds[:-1])
    metrics['auc'] = auc

    return metrics

def _binary_crossentropy(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Binary cross entropy metric."""
    epsilon = 1e-15
    y_scores = np.clip(y_scores, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_scores) + (1 - y_true) * np.log(1 - y_scores))

def _mean_squared_error(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Mean squared error metric."""
    return np.mean((y_true - y_scores) ** 2)

################################################################################
# rappel
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Inputs must not contain NaN values")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Inputs must not contain infinite values")

def _compute_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> float:
    """Compute recall metric."""
    y_pred_binary = (y_pred >= threshold).astype(int)
    true_positives = np.sum((y_true == 1) & (y_pred_binary == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred_binary == 0))
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    return recall

def _compute_custom_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> float:
    """Compute custom metric."""
    return metric_func(y_true, y_pred)

def rappel_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    metric: str = "recall",
    custom_metric_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute recall and optionally other metrics.

    Parameters:
    - y_true: True binary labels
    - y_pred: Predicted probabilities or scores
    - threshold: Decision threshold for binary classification
    - metric: Metric to compute ("recall" or "custom")
    - custom_metric_func: Custom metric function if metric="custom"

    Returns:
    Dictionary containing results, metrics, parameters used and warnings
    """
    _validate_inputs(y_true, y_pred)

    result = {}
    metrics = {}

    # Compute recall
    recall_value = _compute_recall(y_true, y_pred, threshold)
    result["recall"] = recall_value

    # Compute additional metrics if specified
    if metric == "custom" and custom_metric_func is not None:
        custom_value = _compute_custom_metric(y_true, y_pred, custom_metric_func)
        metrics["custom"] = custom_value
    elif metric == "recall":
        pass  # recall already computed

    params_used = {
        "threshold": threshold,
        "metric": metric
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
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.2, 0.8, 0.7, 0.1])
result = rappel_fit(y_true, y_pred)
"""

################################################################################
# f1_score
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Inputs must not contain NaN values")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Inputs must not contain infinite values")

def _compute_precision_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute precision and recall."""
    y_pred_binary = (y_pred >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true == 1) & (y_pred_binary == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {"precision": precision, "recall": recall}

def f1_score_compute(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    beta: float = 1.0
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute the F1 score.

    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth (binary) labels.
    y_pred : np.ndarray
        Predicted probabilities or binary labels.
    threshold : float, optional
        Threshold for converting probabilities to binary predictions (default: 0.5).
    beta : float, optional
        Weighting factor for recall in the F-score (default: 1.0).

    Returns:
    --------
    dict
        Dictionary containing the F1 score, precision, recall, and other metadata.
    """
    _validate_inputs(y_true, y_pred)

    pr_metrics = _compute_precision_recall(y_true, y_pred, threshold)
    precision = pr_metrics["precision"]
    recall = pr_metrics["recall"]

    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

    return {
        "result": f1,
        "metrics": pr_metrics,
        "params_used": {
            "threshold": threshold,
            "beta": beta
        },
        "warnings": []
    }

# Example usage:
# f1_score_compute(np.array([0, 1, 1]), np.array([0.1, 0.6, 0.9]))

################################################################################
# seuil_classification
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
) -> None:
    """Validate input arrays and metric function."""
    if y_true.ndim != 1 or y_scores.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays")
    if len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must have the same length")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or infinite values")
    if np.any(np.isnan(y_scores)) or np.any(np.isinf(y_scores)):
        raise ValueError("y_scores contains NaN or infinite values")

    # Test metric function with sample data
    try:
        test_y = np.array([0, 1])
        test_scores = np.array([0.2, 0.8])
        metric_func(test_y, test_scores)
    except Exception as e:
        raise ValueError(f"Metric function failed validation: {str(e)}")

def _compute_threshold_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
) -> Dict[str, Union[float, np.ndarray]]:
    """Compute metrics for a given threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    return {
        "threshold": threshold,
        "metric_value": metric_func(y_true, y_pred),
        "y_pred": y_pred,
    }

def _find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    """Find the optimal threshold that maximizes/minimizes the metric."""
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)

    results = []
    for threshold in thresholds:
        metrics = _compute_threshold_metrics(y_true, y_scores, threshold, metric_func)
        results.append(metrics)

    # Convert to numpy array for easier manipulation
    results_array = np.array([r["metric_value"] for r in results])

    # Find optimal threshold based on metric direction
    if "loss" in metric_func.__name__.lower() or "error" in metric_func.__name__.lower():
        optimal_idx = np.argmin(results_array)
    else:
        optimal_idx = np.argmax(results_array)

    return results[optimal_idx]

def seuil_classification_fit(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    thresholds: Optional[np.ndarray] = None,
    normalize_scores: bool = False,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Find the optimal classification threshold that optimizes a given metric.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of true binary labels (0 or 1)
    y_scores : np.ndarray
        Array of predicted scores/probabilities (between 0 and 1)
    metric_func : callable
        Function that takes (y_true, y_pred) and returns a scalar metric value
    thresholds : np.ndarray, optional
        Array of threshold values to evaluate. If None, uses 100 evenly spaced values.
    normalize_scores : bool, optional
        Whether to normalize the scores before finding threshold

    Returns:
    --------
    dict
        Dictionary containing:
        - "result": optimal threshold value
        - "metrics": dictionary of computed metrics
        - "params_used": parameters used in the computation
        - "warnings": any warnings generated during computation

    Example:
    --------
    >>> def accuracy_metric(y_true, y_pred):
    ...     return np.mean(y_true == y_pred)
    ...
    >>> result = seuil_classification_fit(
    ...     y_true=np.array([0, 1, 0, 1]),
    ...     y_scores=np.array([0.1, 0.9, 0.2, 0.8]),
    ...     metric_func=accuracy_metric
    ... )
    """
    # Initialize output dictionary
    output = {
        "result": None,
        "metrics": {},
        "params_used": {
            "thresholds_provided": thresholds is not None,
            "normalize_scores": normalize_scores,
        },
        "warnings": [],
    }

    # Validate inputs
    _validate_inputs(y_true, y_scores, metric_func)

    # Normalize scores if requested
    if normalize_scores:
        min_score = np.min(y_scores)
        max_score = np.max(y_scores)
        if min_score == max_score:
            output["warnings"].append("All scores are identical - normalization skipped")
        else:
            y_scores = (y_scores - min_score) / (max_score - min_score)

    # Find optimal threshold
    try:
        optimal_result = _find_optimal_threshold(y_true, y_scores, metric_func, thresholds)
    except Exception as e:
        raise RuntimeError(f"Failed to find optimal threshold: {str(e)}")

    # Prepare output
    output["result"] = optimal_result["threshold"]
    output["metrics"]["optimal_metric_value"] = optimal_result["metric_value"]

    return output

################################################################################
# auc_roc
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def auc_roc_compute(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    normalization: str = 'none',
    metric: Union[str, Callable] = 'auc',
    solver: str = 'trapezoidal',
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Union[float, Dict]]:
    """
    Compute the Area Under the ROC Curve (AUC-ROC).

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_scores : np.ndarray
        Predicted scores or probabilities.
    normalization : str, optional
        Normalization method for the scores ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to compute ('auc', 'pr_auc') or a custom callable.
    solver : str, optional
        Solver method ('trapezoidal', 'riemann').
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    **kwargs :
        Additional keyword arguments for custom functions.

    Returns
    -------
    Dict[str, Union[float, Dict]]
        Dictionary containing:
        - 'result': Computed AUC value.
        - 'metrics': Additional metrics if applicable.
        - 'params_used': Parameters used in the computation.
        - 'warnings': Any warnings generated during computation.

    Examples
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> result = auc_roc_compute(y_true, y_scores)
    """
    # Validate inputs
    _validate_inputs(y_true, y_scores)

    # Normalize scores if required
    normalized_scores = _normalize_scores(y_scores, method=normalization)

    # Compute AUC based on the chosen solver and metric
    if isinstance(metric, str):
        if metric == 'auc':
            auc_value = _compute_auc(y_true, normalized_scores, solver=solver)
        elif metric == 'pr_auc':
            auc_value = _compute_pr_auc(y_true, normalized_scores)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        auc_value = metric(y_true, normalized_scores, **kwargs)
    else:
        raise ValueError("Metric must be a string or callable.")

    # Prepare the output dictionary
    result_dict = {
        'result': auc_value,
        'metrics': {},
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'solver': solver
        },
        'warnings': []
    }

    return result_dict

def _validate_inputs(y_true: np.ndarray, y_scores: np.ndarray) -> None:
    """
    Validate the input arrays.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_scores : np.ndarray
        Predicted scores or probabilities.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if y_true.shape != y_scores.shape:
        raise ValueError("y_true and y_scores must have the same shape.")
    if np.any((y_true != 0) & (y_true != 1)):
        raise ValueError("y_true must contain only 0s and 1s.")
    if np.any(np.isnan(y_scores)):
        raise ValueError("y_scores must not contain NaN values.")
    if np.any(np.isinf(y_scores)):
        raise ValueError("y_scores must not contain infinite values.")

def _normalize_scores(scores: np.ndarray, method: str = 'none') -> np.ndarray:
    """
    Normalize the scores using the specified method.

    Parameters
    ----------
    scores : np.ndarray
        Predicted scores or probabilities.
    method : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').

    Returns
    -------
    np.ndarray
        Normalized scores.
    """
    if method == 'none':
        return scores
    elif method == 'standard':
        mean = np.mean(scores)
        std = np.std(scores)
        if std == 0:
            return scores
        return (scores - mean) / std
    elif method == 'minmax':
        min_val = np.min(scores)
        max_val = np.max(scores)
        if max_val == min_val:
            return scores
        return (scores - min_val) / (max_val - min_val)
    elif method == 'robust':
        median = np.median(scores)
        iqr = np.percentile(scores, 75) - np.percentile(scores, 25)
        if iqr == 0:
            return scores
        return (scores - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_auc(y_true: np.ndarray, y_scores: np.ndarray, solver: str = 'trapezoidal') -> float:
    """
    Compute the Area Under the ROC Curve (AUC-ROC).

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_scores : np.ndarray
        Predicted scores or probabilities.
    solver : str, optional
        Solver method ('trapezoidal', 'riemann').

    Returns
    -------
    float
        Computed AUC value.
    """
    # Sort the scores and get corresponding true labels
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_scores = y_scores[sorted_indices]
    sorted_labels = y_true[sorted_indices]

    # Compute the ROC curve
    tpr = np.cumsum(sorted_labels) / np.sum(sorted_labels)
    fpr = np.cumsum(1 - sorted_labels) / np.sum(1 - sorted_labels)

    # Add (0, 0) and (1, 1) points
    fpr = np.concatenate(([0], fpr, [1]))
    tpr = np.concatenate(([0], tpr, [1]))

    # Compute AUC using the specified solver
    if solver == 'trapezoidal':
        auc = np.trapz(tpr, fpr)
    elif solver == 'riemann':
        auc = np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    return auc

def _compute_pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute the Area Under the Precision-Recall Curve (PR-AUC).

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_scores : np.ndarray
        Predicted scores or probabilities.

    Returns
    -------
    float
        Computed PR-AUC value.
    """
    # Sort the scores and get corresponding true labels
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_scores = y_scores[sorted_indices]
    sorted_labels = y_true[sorted_indices]

    # Compute precision and recall
    precision = np.cumsum(sorted_labels) / np.arange(1, len(y_true) + 1)
    recall = np.cumsum(sorted_labels) / np.sum(sorted_labels)

    # Add (0, 1) point
    precision = np.concatenate(([1], precision))
    recall = np.concatenate(([0], recall))

    # Compute PR-AUC using the trapezoidal rule
    pr_auc = np.trapz(precision, recall)

    return pr_auc

################################################################################
# auc_pr
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def auc_pr_fit(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    normalization: str = 'none',
    metric: Union[str, Callable] = 'average_precision',
    solver: str = 'trapezoidal',
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Union[float, Dict]]:
    """
    Compute the Area Under the Precision-Recall Curve (AUC-PR).

    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth binary labels (0 or 1).
    y_scores : np.ndarray
        Predicted scores or probabilities.
    normalization : str, optional (default='none')
        Type of normalization to apply. Options: 'none', 'minmax'.
    metric : str or callable, optional (default='average_precision')
        Metric to compute. Options: 'average_precision', custom callable.
    solver : str, optional (default='trapezoidal')
        Solver to compute the AUC. Options: 'trapezoidal', 'rectangle'.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    **kwargs :
        Additional keyword arguments for the solver or metric.

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': float, the computed AUC-PR.
        - 'metrics': dict, additional metrics if applicable.
        - 'params_used': dict, parameters used in the computation.
        - 'warnings': list, any warnings generated.

    Example:
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_scores = np.array([0.1, 0.9, 0.8, 0.2])
    >>> result = auc_pr_fit(y_true, y_scores)
    """
    # Validate inputs
    _validate_inputs(y_true, y_scores)

    # Normalize scores if required
    if normalization != 'none':
        y_scores = _normalize(y_scores, method=normalization)

    # Compute precision-recall curve
    precision, recall = _compute_precision_recall(y_true, y_scores)

    # Compute AUC-PR
    auc_pr = _compute_auc(precision, recall, solver=solver)

    # Compute additional metrics if required
    metrics = {}
    if metric == 'average_precision':
        metrics['average_precision'] = _compute_average_precision(precision, recall)
    elif callable(metric):
        metrics['custom_metric'] = metric(precision, recall)

    # Prepare output
    result_dict = {
        'result': auc_pr,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'solver': solver
        },
        'warnings': []
    }

    return result_dict

def _validate_inputs(y_true: np.ndarray, y_scores: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_scores, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if y_true.shape != y_scores.shape:
        raise ValueError("y_true and y_scores must have the same shape.")
    if np.any((y_true != 0) & (y_true != 1)):
        raise ValueError("y_true must contain only 0s and 1s.")
    if np.any(np.isnan(y_scores)) or np.any(np.isinf(y_scores)):
        raise ValueError("y_scores must not contain NaN or Inf values.")

def _normalize(y_scores: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """Normalize the scores."""
    if method == 'minmax':
        return (y_scores - np.min(y_scores)) / (np.max(y_scores) - np.min(y_scores))
    elif method == 'standard':
        return (y_scores - np.mean(y_scores)) / np.std(y_scores)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_precision_recall(y_true: np.ndarray, y_scores: np.ndarray) -> tuple:
    """Compute precision and recall values."""
    # Sort scores in descending order
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]

    # Compute precision and recall
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)
    precision = tp / (tp + fp)
    recall = tp / np.sum(y_true)

    # Ensure precision and recall are decreasing
    precision = np.concatenate(([1], precision[:-1]))
    recall = np.concatenate(([0], recall[:-1]))

    return precision, recall

def _compute_auc(precision: np.ndarray, recall: np.ndarray, solver: str = 'trapezoidal') -> float:
    """Compute AUC using the specified solver."""
    if solver == 'trapezoidal':
        return np.trapz(precision, recall)
    elif solver == 'rectangle':
        return np.sum((recall[1:] - recall[:-1]) * precision[:-1])
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _compute_average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
    """Compute average precision."""
    return _compute_auc(precision, recall)
