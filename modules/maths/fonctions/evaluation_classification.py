"""
Quantix – Module evaluation_classification
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# accuracy
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays for accuracy computation."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Inputs must not contain NaN values")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Inputs must not contain infinite values")

def _compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy score."""
    return np.mean(y_true == y_pred)

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Optional[Dict[str, Callable]] = None
) -> Dict[str, float]:
    """Compute additional metrics if provided."""
    result_metrics = {}
    if metrics is not None:
        for name, metric_func in metrics.items():
            try:
                result_metrics[name] = metric_func(y_true, y_pred)
            except Exception as e:
                result_metrics[name] = np.nan
    return result_metrics

def accuracy_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Optional[Dict[str, Callable]] = None
) -> Dict[str, Any]:
    """
    Compute accuracy and optionally other metrics for classification.

    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth (correct) target values.
    y_pred : np.ndarray
        Estimated targets as returned by a classifier.
    metrics : Optional[Dict[str, Callable]], default=None
        Dictionary of additional metric functions to compute.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - "result": accuracy score
        - "metrics": additional metrics if provided
        - "params_used": parameters used in computation
        - "warnings": any warnings generated

    Example:
    --------
    >>> y_true = np.array([0, 1, 2, 2])
    >>> y_pred = np.array([0, 1, 1, 2])
    >>> accuracy_fit(y_true, y_pred)
    {
        'result': 0.75,
        'metrics': {},
        'params_used': {'metrics': None},
        'warnings': []
    }
    """
    _validate_inputs(y_true, y_pred)

    result = _compute_accuracy(y_true, y_pred)
    computed_metrics = _compute_metrics(y_true, y_pred, metrics)

    return {
        "result": result,
        "metrics": computed_metrics,
        "params_used": {"metrics": metrics},
        "warnings": []
    }

################################################################################
# precision
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> None:
    """Validate input arrays."""
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-dimensional arrays")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("y_true and y_pred must not contain NaN values")
    if sample_weight is not None:
        if len(sample_weight) != len(y_true):
            raise ValueError("sample_weight must have the same length as y_true")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight must be non-negative")

def _compute_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: int = 1,
    sample_weight: Optional[np.ndarray] = None
) -> float:
    """Compute precision metric."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    # Calculate true positives and false positives
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))

    if sample_weight is not None:
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label) * sample_weight)
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label) * sample_weight)
        total = np.sum(sample_weight)
    else:
        total = len(y_true)

    if tp + fp == 0:
        return np.nan
    precision = tp / (tp + fp)
    return precision

def precision_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: int = 1,
    sample_weight: Optional[np.ndarray] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute precision metric for classification.

    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth (correct) target values.
    y_pred : np.ndarray
        Estimated targets as returned by a classifier.
    pos_label : int, optional (default=1)
        The class label to consider as positive.
    sample_weight : np.ndarray, optional
        Sample weights.

    Returns:
    --------
    dict
        Dictionary containing:
        - "result": computed precision value
        - "metrics": dictionary with metric name and value
        - "params_used": dictionary of parameters used
        - "warnings": dictionary of warnings (empty if none)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred, sample_weight)

    # Compute precision
    result = _compute_precision(y_true, y_pred, pos_label, sample_weight)

    # Prepare output
    output = {
        "result": result,
        "metrics": {"precision": result},
        "params_used": {
            "pos_label": pos_label,
            "sample_weight": sample_weight is not None
        },
        "warnings": {}
    }

    if np.isnan(result):
        output["warnings"]["no_positive_predictions"] = (
            "Precision is undefined (no positive predictions)"
        )

    return output

# Example usage:
"""
y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 0, 0, 1])
result = precision_fit(y_true, y_pred)
print(result)
"""

################################################################################
# recall
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays for recall calculation."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Inputs must not contain NaN values")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Inputs must not contain infinite values")

def _calculate_recall(y_true: np.ndarray, y_pred: np.ndarray,
                     threshold: float = 0.5) -> float:
    """Calculate recall score."""
    y_true = np.array(y_true, dtype=bool)
    y_pred = np.array(y_pred >= threshold, dtype=bool)

    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))

    recall = true_positives / (true_positives + false_negatives)
    return recall if not np.isnan(recall) else 0.0

def recall_fit(y_true: np.ndarray, y_pred: np.ndarray,
               threshold: float = 0.5) -> Dict[str, Union[float, Dict]]:
    """
    Calculate recall score for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true binary labels (0 or 1)
    y_pred : np.ndarray
        Array of predicted probabilities or binary labels (0 or 1)
    threshold : float, optional
        Threshold for converting probabilities to binary predictions (default: 0.5)

    Returns
    -------
    dict
        Dictionary containing:
        - result: float, the recall score
        - metrics: dict with additional metrics if needed
        - params_used: dict of parameters used in calculation
        - warnings: list of warning messages

    Example
    -------
    >>> y_true = np.array([1, 0, 1, 1])
    >>> y_pred = np.array([0.9, 0.2, 0.8, 0.3])
    >>> recall_fit(y_true, y_pred)
    {
        'result': 0.666...,
        'metrics': {},
        'params_used': {'threshold': 0.5},
        'warnings': []
    }
    """
    _validate_inputs(y_true, y_pred)

    result = _calculate_recall(y_true, y_pred, threshold)

    return {
        'result': result,
        'metrics': {},
        'params_used': {'threshold': threshold},
        'warnings': []
    }

################################################################################
# f1_score
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays for F1 score calculation."""
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if np.any((y_true != 0) & (y_true != 1)):
        raise ValueError("y_true must contain only 0 and 1.")
    if np.any((y_pred != 0) & (y_pred != 1)):
        raise ValueError("y_pred must contain only 0 and 1.")

def _calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Calculate confusion matrix components."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

def _calculate_f1_score(tp: int, fp: int, tn: int, fn: int) -> float:
    """Calculate F1 score from confusion matrix components."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def _calculate_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    """Calculate additional metrics from confusion matrix."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy
    }

def f1_score_compute(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    custom_metric: Optional[Callable[[Dict[str, int]], float]] = None
) -> Dict[str, Any]:
    """
    Compute F1 score and additional metrics for binary classification.

    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth (binary) target values.
    y_pred : np.ndarray
        Estimated targets as returned by a classifier.
    custom_metric : Optional[Callable]
        Custom metric function that takes confusion matrix components and returns a float.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - "result": F1 score
        - "metrics": Additional metrics (precision, recall, accuracy)
        - "params_used": Parameters used in calculation
        - "warnings": Any warnings generated

    Example:
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> result = f1_score_compute(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Calculate confusion matrix components
    cm = _calculate_confusion_matrix(y_true, y_pred)

    # Calculate F1 score
    f1 = _calculate_f1_score(cm["tp"], cm["fp"], cm["tn"], cm["fn"])

    # Calculate additional metrics
    metrics = _calculate_metrics(cm["tp"], cm["fp"], cm["tn"], cm["fn"])

    # Calculate custom metric if provided
    custom_result = None
    if custom_metric is not None:
        try:
            custom_result = custom_metric(cm)
        except Exception as e:
            warnings.append(f"Custom metric calculation failed: {str(e)}")

    # Prepare output
    result = {
        "result": f1,
        "metrics": metrics,
        "params_used": {
            "custom_metric": custom_metric.__name__ if custom_metric else None
        },
        "warnings": []
    }

    return result

################################################################################
# roc_auc
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def roc_auc_fit(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    *,
    normalize: str = "none",
    metric: Union[str, Callable] = "auc",
    solver: str = "trapezoidal",
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Union[float, Dict]]:
    """
    Compute the ROC AUC score for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_scores : np.ndarray
        Predicted scores or probabilities.
    normalize : str, optional
        Normalization method for the scores. Options: "none", "standard", "minmax".
    metric : str or callable, optional
        Metric to compute. Options: "auc" (default), custom callable.
    solver : str, optional
        Solver method. Options: "trapezoidal" (default).
    custom_metric : callable, optional
        Custom metric function if not using default.
    **kwargs :
        Additional keyword arguments for the solver or custom functions.

    Returns
    -------
    Dict[str, Union[float, Dict]]
        Dictionary containing:
        - "result": float, the ROC AUC score.
        - "metrics": Dict, additional metrics if computed.
        - "params_used": Dict, parameters used in the computation.
        - "warnings": List[str], any warnings generated.

    Examples
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_scores = np.array([0.1, 0.9, 0.8, 0.2])
    >>> result = roc_auc_fit(y_true, y_scores)
    """
    # Validate inputs
    _validate_inputs(y_true, y_scores)

    # Normalize scores if required
    normalized_scores = _normalize_scores(y_scores, method=normalize)

    # Compute ROC AUC
    auc_score = _compute_roc_auc(
        y_true, normalized_scores, solver=solver, custom_metric=custom_metric, **kwargs
    )

    # Prepare output
    result = {
        "result": auc_score,
        "metrics": {},
        "params_used": {
            "normalize": normalize,
            "metric": metric if isinstance(metric, str) else "custom",
            "solver": solver,
        },
        "warnings": [],
    }

    return result

def _validate_inputs(y_true: np.ndarray, y_scores: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_scores, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if y_true.shape != y_scores.shape:
        raise ValueError("y_true and y_scores must have the same shape.")
    if np.any((y_true != 0) & (y_true != 1)):
        raise ValueError("y_true must contain only 0s and 1s.")
    if np.any(np.isnan(y_scores)) or np.any(np.isinf(y_scores)):
        raise ValueError("y_scores must not contain NaN or inf values.")

def _normalize_scores(scores: np.ndarray, method: str = "none") -> np.ndarray:
    """Normalize scores using the specified method."""
    if method == "none":
        return scores
    elif method == "standard":
        mean = np.mean(scores)
        std = np.std(scores)
        if std == 0:
            return scores
        return (scores - mean) / std
    elif method == "minmax":
        min_val = np.min(scores)
        max_val = np.max(scores)
        if max_val == min_val:
            return scores
        return (scores - min_val) / (max_val - min_val)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_roc_auc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    solver: str = "trapezoidal",
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> float:
    """Compute ROC AUC using the specified solver."""
    if custom_metric is not None:
        return _compute_custom_roc_auc(y_true, y_scores, custom_metric, **kwargs)

    if solver == "trapezoidal":
        return _compute_roc_auc_trapezoidal(y_true, y_scores)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _compute_roc_auc_trapezoidal(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute ROC AUC using the trapezoidal rule."""
    # Sort scores and get corresponding true labels
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_scores = y_scores[sorted_indices]
    sorted_labels = y_true[sorted_indices]

    # Compute TPR and FPR
    tpr = np.cumsum(sorted_labels) / np.sum(sorted_labels)
    fpr = np.cumsum(1 - sorted_labels) / np.sum(1 - sorted_labels)

    # Compute AUC using trapezoidal rule
    auc = np.trapz(tpr, fpr)
    return auc

def _compute_custom_roc_auc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    custom_metric: Callable,
    **kwargs
) -> float:
    """Compute ROC AUC using a custom metric."""
    return custom_metric(y_true, y_scores, **kwargs)

################################################################################
# confusion_matrix
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

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

def _compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute the confusion matrix."""
    if labels is None:
        unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    else:
        unique_labels = labels

    n_classes = len(unique_labels)
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    for i, true_label in enumerate(unique_labels):
        for j, pred_label in enumerate(unique_labels):
            confusion_matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))

    return confusion_matrix

def _normalize_confusion_matrix(
    cm: np.ndarray,
    norm: str = 'none'
) -> np.ndarray:
    """Normalize the confusion matrix."""
    if norm == 'none':
        return cm
    elif norm == 'true':
        return cm / cm.sum(axis=1, keepdims=True)
    elif norm == 'pred':
        return cm / cm.sum(axis=0, keepdims=True)
    elif norm == 'all':
        return cm / cm.sum()
    else:
        raise ValueError("Invalid normalization option")

def _compute_metrics(
    cm: np.ndarray,
    metrics: Optional[Dict[str, Callable]] = None
) -> Dict[str, float]:
    """Compute metrics from the confusion matrix."""
    if metrics is None:
        return {}

    result = {}
    for name, func in metrics.items():
        try:
            result[name] = func(cm)
        except Exception as e:
            result[name] = np.nan
    return result

def confusion_matrix_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[np.ndarray] = None,
    norm: str = 'none',
    metrics: Optional[Dict[str, Callable]] = None
) -> Dict[str, Any]:
    """
    Compute the confusion matrix and associated metrics.

    Parameters:
    - y_true: Ground truth (correct) target values.
    - y_pred: Estimated targets as returned by a classifier.
    - labels: List of labels to index the matrix. This may be used to reorder or select a subset of labels.
    - norm: Normalization method ('none', 'true', 'pred', 'all').
    - metrics: Dictionary of metric names and their corresponding functions.

    Returns:
    - A dictionary containing the confusion matrix, computed metrics, used parameters, and warnings.
    """
    _validate_inputs(y_true, y_pred)

    cm = _compute_confusion_matrix(y_true, y_pred, labels)
    norm_cm = _normalize_confusion_matrix(cm, norm)
    computed_metrics = _compute_metrics(norm_cm, metrics)

    return {
        'result': norm_cm,
        'metrics': computed_metrics,
        'params_used': {
            'norm': norm,
            'labels': labels
        },
        'warnings': []
    }

# Example usage:
"""
y_true = np.array([0, 1, 2, 0, 1])
y_pred = np.array([0, 1, 1, 0, 2])

def accuracy(cm):
    return np.trace(cm) / cm.sum()

metrics = {
    'accuracy': accuracy
}

result = confusion_matrix_fit(y_true, y_pred, metrics=metrics)
"""

################################################################################
# precision_recall_curve
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def precision_recall_curve_fit(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    pos_label: Union[int, str] = 1,
    sample_weight: Optional[np.ndarray] = None,
    normalize: bool = True
) -> Dict[str, Union[Tuple[np.ndarray, np.ndarray], Dict[str, float]]]:
    """
    Compute precision-recall pairs for different probability thresholds.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_scores : array-like of shape (n_samples,)
        Target scores, can either be probability estimates or non-thresholded
        decision function.
    pos_label : int or str, default=1
        The class label of the positive class.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    normalize : bool, default=True
        Whether to normalize the precision and recall values.

    Returns
    -------
    dict
        A dictionary containing:
        - 'precision_recall_curve': Tuple of precision and recall values
        - 'metrics': Dictionary containing additional metrics like average_precision

    Examples
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> result = precision_recall_curve_fit(y_true, y_scores)
    """
    # Validate inputs
    _validate_inputs(y_true, y_scores, sample_weight)

    # Compute precision-recall curve
    precision, recall, thresholds = _compute_precision_recall_curve(
        y_true, y_scores, pos_label, sample_weight
    )

    # Compute additional metrics
    average_precision = _compute_average_precision(precision, recall)

    # Normalize if required
    if normalize:
        precision, recall = _normalize_precision_recall(precision, recall)

    return {
        'result': (precision, recall),
        'metrics': {'average_precision': average_precision},
        'params_used': {
            'pos_label': pos_label,
            'sample_weight': sample_weight is not None,
            'normalize': normalize
        },
        'warnings': _check_warnings(y_true, y_scores)
    }

def _validate_inputs(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    sample_weight: Optional[np.ndarray]
) -> None:
    """Validate input arrays."""
    if y_true.shape != y_scores.shape:
        raise ValueError("y_true and y_scores must have the same shape")
    if sample_weight is not None and y_true.shape != sample_weight.shape:
        raise ValueError("y_true and sample_weight must have the same shape")
    if np.any(np.isnan(y_scores)):
        raise ValueError("y_scores contains NaN values")
    if np.any(np.isnan(y_true)):
        raise ValueError("y_true contains NaN values")

def _compute_precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    pos_label: Union[int, str],
    sample_weight: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision-recall curve."""
    # Sort scores and corresponding true labels
    desc_score_indices = np.argsort(y_scores, kind="mergesort")[::-1]
    y_scores_sorted = y_scores[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]

    if sample_weight is not None:
        sample_weight_sorted = sample_weight[desc_score_indices]
    else:
        sample_weight_sorted = None

    # Compute precision and recall for each threshold
    unique_scores = np.unique(y_scores_sorted)
    precision = np.zeros_like(unique_scores)
    recall = np.zeros_like(unique_scores)

    for i, threshold in enumerate(unique_scores):
        tp = np.sum((y_true_sorted >= threshold) & (y_true_sorted == pos_label))
        fp = np.sum((y_true_sorted >= threshold) & (y_true_sorted != pos_label))
        fn = np.sum((y_true_sorted < threshold) & (y_true_sorted == pos_label))

        if sample_weight is not None:
            tp = np.sum((y_true_sorted >= threshold) & (y_true_sorted == pos_label) * sample_weight_sorted)
            fp = np.sum((y_true_sorted >= threshold) & (y_true_sorted != pos_label) * sample_weight_sorted)
            fn = np.sum((y_true_sorted < threshold) & (y_true_sorted == pos_label) * sample_weight_sorted)

        precision[i] = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) != 0 else 0

    return precision, recall, unique_scores

def _compute_average_precision(
    precision: np.ndarray,
    recall: np.ndarray
) -> float:
    """Compute average precision."""
    return -np.sum(np.diff(recall) * np.array(precision[:-1]))

def _normalize_precision_recall(
    precision: np.ndarray,
    recall: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize precision and recall values."""
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return precision, recall

def _check_warnings(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Dict[str, str]:
    """Check for potential warnings."""
    warnings = {}
    if np.all(y_true == y_true[0]):
        warnings['constant_labels'] = "All labels are the same"
    if np.all(y_scores == y_scores[0]):
        warnings['constant_scores'] = "All scores are the same"
    return warnings

################################################################################
# log_loss
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.ndim != 1 or y_pred.ndim != 2:
        raise ValueError("y_true must be 1D and y_pred must be 2D")
    if len(y_true) != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have compatible shapes")
    if np.any((y_pred < 0) | (y_pred > 1)):
        raise ValueError("Predicted probabilities must be in [0, 1]")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Inputs must not contain NaN values")

def _compute_log_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-15
) -> float:
    """Compute the log loss."""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    n_samples = len(y_true)
    log_loss = -np.sum(y_true * np.log(y_pred)) / n_samples
    return log_loss

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute additional metrics."""
    log_loss = _compute_log_loss(y_true, y_pred)
    return {"log_loss": log_loss}

def log_loss_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-15,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute the log loss for classification.

    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_pred : np.ndarray
        Predicted probabilities of the positive class.
    epsilon : float, optional
        Small value to avoid log(0).
    custom_metric : Callable, optional
        Custom metric function.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, and parameters used.
    """
    _validate_inputs(y_true, y_pred)

    result = {
        "result": None,
        "metrics": {},
        "params_used": {
            "epsilon": epsilon
        },
        "warnings": []
    }

    metrics = _compute_metrics(y_true, y_pred)
    result["metrics"].update(metrics)

    if custom_metric is not None:
        try:
            custom_value = custom_metric(y_true, y_pred)
            result["metrics"]["custom_metric"] = custom_value
        except Exception as e:
            result["warnings"].append(f"Custom metric failed: {str(e)}")

    return result

# Example usage:
"""
y_true = np.array([0, 1, 1])
y_pred = np.array([[0.1, 0.9], [0.2, 0.8], [0.7, 0.3]])
result = log_loss_fit(y_true, y_pred)
print(result)
"""

################################################################################
# brier_score
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> None:
    """Validate input arrays."""
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if np.any(y_true < 0) or np.any(y_true > 1):
        raise ValueError("y_true must be in [0, 1]")
    if np.any(y_prob < 0) or np.any(y_prob > 1):
        raise ValueError("y_prob must be in [0, 1]")
    if sample_weight is not None:
        if len(sample_weight) != len(y_true):
            raise ValueError("sample_weight must have the same length as y_true")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative values")

def _compute_brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> float:
    """Compute the Brier score."""
    squared_error = (y_true - y_prob) ** 2
    if sample_weight is not None:
        return np.average(squared_error, weights=sample_weight)
    return np.mean(squared_error)

def brier_score_fit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    metric_func: Optional[Callable] = None
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Compute the Brier score for probabilistic classification.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities of the positive class.
    sample_weight : Optional[np.ndarray], default=None
        Individual weights for each sample.
    metric_func : Optional[Callable], default=None
        Custom metric function. If None, uses the standard Brier score.

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], str]]
        Dictionary containing:
        - "result": computed Brier score
        - "metrics": additional metrics if provided
        - "params_used": parameters used in computation
        - "warnings": any warnings generated

    Example
    -------
    >>> y_true = np.array([0, 1, 1])
    >>> y_prob = np.array([0.1, 0.9, 0.8])
    >>> result = brier_score_fit(y_true, y_prob)
    """
    _validate_inputs(y_true, y_prob, sample_weight)

    warnings = []
    params_used = {
        "sample_weight": sample_weight is not None,
        "custom_metric": metric_func is not None
    }

    if metric_func is not None:
        metrics = {"custom_metric": metric_func(y_true, y_prob)}
    else:
        brier_score = _compute_brier_score(y_true, y_prob, sample_weight)
        metrics = {"brier_score": brier_score}

    return {
        "result": brier_score if metric_func is None else metrics["custom_metric"],
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

################################################################################
# cohen_kappa
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays for Cohen's Kappa calculation."""
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if not np.issubdtype(y_true.dtype, np.integer) or not np.issubdtype(y_pred.dtype, np.integer):
        raise ValueError("y_true and y_pred must contain integer values")
    if np.any((y_true < 0) | (y_pred < 0)):
        raise ValueError("Values in y_true and y_pred must be non-negative")

def _create_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Create confusion matrix from true and predicted labels."""
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(unique_labels)
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    for i, true_label in enumerate(unique_labels):
        for j, pred_label in enumerate(unique_labels):
            confusion_matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))

    return confusion_matrix

def _calculate_cohen_kappa(confusion_matrix: np.ndarray) -> float:
    """Calculate Cohen's Kappa coefficient from confusion matrix."""
    n_classes = confusion_matrix.shape[0]
    observed_agreement = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)

    row_sums = confusion_matrix.sum(axis=1)
    col_sums = confusion_matrix.sum(axis=0)

    expected_agreement = np.sum(row_sums * col_sums) / (np.sum(confusion_matrix) ** 2)

    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)

    return kappa

def cohen_kappa_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    validation_func: Optional[Callable[[np.ndarray, np.ndarray], None]] = _validate_inputs,
    confusion_matrix_func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = _create_confusion_matrix,
    kappa_func: Optional[Callable[[np.ndarray], float]] = _calculate_cohen_kappa,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Calculate Cohen's Kappa coefficient for classification evaluation.

    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth (correct) target values.
    y_pred : np.ndarray
        Estimated targets as returned by a classifier.
    validation_func : Callable, optional
        Function to validate input arrays (default: _validate_inputs).
    confusion_matrix_func : Callable, optional
        Function to create confusion matrix (default: _create_confusion_matrix).
    kappa_func : Callable, optional
        Function to calculate Cohen's Kappa (default: _calculate_cohen_kappa).
    **kwargs : Any
        Additional keyword arguments for future extensions.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - "result": float, Cohen's Kappa coefficient
        - "metrics": Dict[str, Any], additional metrics (currently empty)
        - "params_used": Dict[str, Any], parameters used in calculation
        - "warnings": List[str], any warnings generated

    Example:
    --------
    >>> y_true = np.array([0, 1, 2, 0, 1])
    >>> y_pred = np.array([0, 1, 2, 1, 1])
    >>> result = cohen_kappa_fit(y_true, y_pred)
    """
    warnings = []

    # Validate inputs
    if validation_func is not None:
        try:
            validation_func(y_true, y_pred)
        except ValueError as e:
            warnings.append(str(e))

    # Create confusion matrix
    if confusion_matrix_func is not None:
        try:
            cm = confusion_matrix_func(y_true, y_pred)
        except Exception as e:
            raise RuntimeError(f"Confusion matrix calculation failed: {str(e)}")
    else:
        cm = _create_confusion_matrix(y_true, y_pred)

    # Calculate Cohen's Kappa
    if kappa_func is not None:
        try:
            kappa = kappa_func(cm)
        except Exception as e:
            raise RuntimeError(f"Kappa calculation failed: {str(e)}")
    else:
        kappa = _calculate_cohen_kappa(cm)

    return {
        "result": kappa,
        "metrics": {},
        "params_used": {
            "validation_func": validation_func.__name__ if validation_func else None,
            "confusion_matrix_func": confusion_matrix_func.__name__ if confusion_matrix_func else None,
            "kappa_func": kappa_func.__name__ if kappa_func else None
        },
        "warnings": warnings
    }

################################################################################
# matthews_corrcoef
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays for Matthews correlation coefficient.

    Args:
        y_true: Array of true labels.
        y_pred: Array of predicted labels.

    Raises:
        ValueError: If inputs are invalid.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if not np.issubdtype(y_true.dtype, np.integer):
        raise ValueError("y_true must contain integer labels")
    if not np.issubdtype(y_pred.dtype, np.integer):
        raise ValueError("y_pred must contain integer labels")
    if len(np.unique(y_true)) != 2:
        raise ValueError("Matthews correlation coefficient requires binary classification")
    if len(np.unique(y_pred)) != 2:
        raise ValueError("Matthews correlation coefficient requires binary classification")

def _compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute confusion matrix for binary classification.

    Args:
        y_true: Array of true labels.
        y_pred: Array of predicted labels.

    Returns:
        Confusion matrix as a 2x2 numpy array.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return np.array([[tn, fp], [fn, tp]])

def matthews_corrcoef_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """Compute Matthews correlation coefficient for binary classification.

    Args:
        y_true: Array of true labels.
        y_pred: Array of predicted labels.
        custom_metric: Optional callable for custom metric calculation.

    Returns:
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Compute confusion matrix
    cm = _compute_confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate Matthews correlation coefficient
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    if denominator == 0:
        mcc = np.nan
    else:
        mcc = numerator / denominator

    # Prepare results dictionary
    result = {
        "result": mcc,
        "metrics": {},
        "params_used": {
            "custom_metric": custom_metric is not None
        },
        "warnings": []
    }

    # Add custom metric if provided
    if custom_metric is not None:
        try:
            result["metrics"]["custom"] = custom_metric(y_true, y_pred)
        except Exception as e:
            result["warnings"].append(f"Custom metric calculation failed: {str(e)}")

    return result

# Example usage:
"""
y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 0, 1, 0, 1])
result = matthews_corrcoef_fit(y_true, y_pred)
print(result['result'])
"""

################################################################################
# balanced_accuracy
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def balanced_accuracy_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray] = None,
    normalize: str = 'true',
    adjust: bool = False
) -> Dict[str, Any]:
    """
    Compute the balanced accuracy score.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    normalize : {'true', 'pred', 'all'}, default='true'
        Type of normalization performed on the results.
    adjust : bool, default=False
        When True, the result is adjusted for chance.

    Returns:
    --------
    dict
        Dictionary containing:
        - result: float, the balanced accuracy score.
        - metrics: dict, additional metrics if any.
        - params_used: dict, parameters used for the computation.
        - warnings: list, any warnings generated.

    Examples:
    ---------
    >>> y_true = np.array([0, 1, 0, 1])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> balanced_accuracy_fit(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred, sample_weight)

    # Compute balanced accuracy
    result = _compute_balanced_accuracy(
        y_true, y_pred,
        sample_weight=sample_weight,
        normalize=normalize,
        adjust=adjust
    )

    # Prepare output dictionary
    return {
        'result': result,
        'metrics': {},
        'params_used': {
            'normalize': normalize,
            'adjust': adjust
        },
        'warnings': []
    }

def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray]
) -> None:
    """Validate the inputs for balanced accuracy computation."""
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if sample_weight is not None and y_true.shape != sample_weight.shape:
        raise ValueError("y_true and sample_weight must have the same shape.")
    if not np.issubdtype(y_true.dtype, np.integer):
        raise ValueError("y_true must contain integer values.")
    if not np.issubdtype(y_pred.dtype, np.integer):
        raise ValueError("y_pred must contain integer values.")
    if sample_weight is not None and np.any(sample_weight < 0):
        raise ValueError("sample_weight cannot contain negative values.")

def _compute_balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray] = None,
    normalize: str = 'true',
    adjust: bool = False
) -> float:
    """Compute the balanced accuracy score."""
    # Calculate per-class accuracies
    class_accuracies = _compute_per_class_accuracies(
        y_true, y_pred,
        sample_weight=sample_weight
    )

    # Normalize the accuracies
    if normalize == 'true':
        result = np.mean(class_accuracies)
    elif normalize == 'pred':
        pred_counts = np.bincount(y_pred)
        result = np.sum(class_accuracies * pred_counts) / np.sum(pred_counts)
    elif normalize == 'all':
        result = np.mean(class_accuracies)
    else:
        raise ValueError("normalize must be 'true', 'pred', or 'all'.")

    # Adjust for chance if needed
    if adjust:
        result = (result - np.mean(class_accuracies)) / (1 - np.mean(class_accuracies))

    return result

def _compute_per_class_accuracies(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute the accuracy for each class."""
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(classes)

    class_accuracies = np.zeros(n_classes)
    for i, cls in enumerate(classes):
        mask = (y_true == cls)
        if sample_weight is not None:
            class_sample_weight = sample_weight[mask]
            correct_mask = (y_pred[mask] == cls)
            class_accuracies[i] = np.sum(class_sample_weight * correct_mask) / np.sum(class_sample_weight)
        else:
            class_accuracies[i] = np.mean((y_pred[mask] == cls))

    return class_accuracies

################################################################################
# average_precision
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

def _compute_average_precision(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    normalize: bool = False
) -> float:
    """Compute average precision."""
    # Sort the scores in descending order
    sorted_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]

    # Compute precision at each threshold
    precision_values = []
    true_positives = 0
    false_positives = 0

    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            true_positives += 1
        else:
            false_positives += 1

        precision = true_positives / (true_positives + false_positives)
        precision_values.append(precision)

    # Compute average precision
    ap = np.mean(precision_values) if normalize else sum(precision_values)

    return ap

def average_precision_fit(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    normalize: bool = False,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]:
    """
    Compute average precision for classification evaluation.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (binary) target values.
    y_scores : np.ndarray
        Estimated probabilities or decision function.
    normalize : bool, optional
        Whether to normalize the average precision (default is False).
    custom_metric : Callable, optional
        Custom metric function to compute additional metrics.

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]
        Dictionary containing:
        - "result": computed average precision
        - "metrics": additional metrics if custom_metric is provided
        - "params_used": parameters used in the computation
        - "warnings": any warnings generated during computation

    Examples
    --------
    >>> y_true = np.array([1, 0, 1, 1, 0])
    >>> y_scores = np.array([0.9, 0.2, 0.8, 0.7, 0.3])
    >>> result = average_precision_fit(y_true, y_scores)
    """
    # Validate inputs
    _validate_inputs(y_true, y_scores)

    # Compute average precision
    ap = _compute_average_precision(y_true, y_scores, normalize)

    # Compute custom metric if provided
    metrics = {}
    if custom_metric is not None:
        try:
            metrics["custom"] = custom_metric(y_true, y_scores)
        except Exception as e:
            metrics["custom_error"] = str(e)

    # Prepare output
    result_dict: Dict[str, Union[float, Dict[str, float], Dict[str, str], list]] = {
        "result": ap,
        "metrics": metrics,
        "params_used": {
            "normalize": normalize
        },
        "warnings": []
    }

    return result_dict

################################################################################
# roc_curve
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def _validate_inputs(y_true: np.ndarray, y_scores: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_scores, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_scores.shape:
        raise ValueError("y_true and y_scores must have the same shape")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_scores)):
        raise ValueError("Inputs must not contain NaN values")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_scores)):
        raise ValueError("Inputs must not contain infinite values")

def _compute_roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ROC curve points."""
    thresholds = np.unique(y_scores)[::-1]
    tpr = np.zeros_like(thresholds)
    fpr = np.zeros_like(thresholds)

    for i, threshold in enumerate(thresholds):
        tp = np.sum((y_scores >= threshold) & (y_true == 1))
        fp = np.sum((y_scores >= threshold) & (y_true == 0))
        tn = np.sum((y_scores < threshold) & (y_true == 0))
        fn = np.sum((y_scores < threshold) & (y_true == 1))

        tpr[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr[i] = fp / (fp + tn) if (fp + tn) > 0 else 0

    return fpr, tpr

def roc_curve_compute(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    normalize: Optional[str] = None,
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Tuple[np.ndarray, np.ndarray], Dict[str, float], Dict[str, str], list]]:
    """
    Compute ROC curve.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_scores : np.ndarray
        Predicted scores or probabilities.
    normalize : str, optional
        Normalization method (none, standard, minmax, robust).
    metric : callable, optional
        Custom metric function.

    Returns
    -------
    dict
        Dictionary containing:
        - "result": Tuple of (fpr, tpr) arrays
        - "metrics": Dictionary of computed metrics
        - "params_used": Dictionary of parameters used
        - "warnings": List of warnings

    Example
    -------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> result = roc_curve_compute(y_true, y_scores)
    """
    # Validate inputs
    _validate_inputs(y_true, y_scores)

    # Compute ROC curve
    fpr, tpr = _compute_roc_curve(y_true, y_scores)

    # Calculate metrics
    metrics = {}
    if metric is not None:
        try:
            metrics['custom_metric'] = metric(y_true, y_scores)
        except Exception as e:
            metrics['custom_metric'] = np.nan
            warnings.append(f"Custom metric calculation failed: {str(e)}")

    # Prepare output
    result = {
        "result": (fpr, tpr),
        "metrics": metrics,
        "params_used": {
            "normalize": normalize if normalize is not None else "none",
            "metric": metric.__name__ if metric is not None else "none"
        },
        "warnings": []
    }

    return result

################################################################################
# pr_auc
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(y_true: np.ndarray, y_scores: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_scores, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.ndim != 1 or y_scores.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional")
    if len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must have the same length")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or inf values")
    if np.any(np.isnan(y_scores)) or np.any(np.isinf(y_scores)):
        raise ValueError("y_scores contains NaN or inf values")

def _compute_precision_recall(y_true: np.ndarray, y_scores: np.ndarray) -> tuple:
    """Compute precision and recall values."""
    # Sort scores in descending order
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]

    # Compute precision and recall
    positives = np.sum(y_true_sorted)
    if positives == 0:
        raise ValueError("No positive samples in y_true")

    precision = np.cumsum(y_true_sorted) / (np.arange(len(y_scores)) + 1)
    recall = np.cumsum(y_true_sorted) / positives

    return precision, recall

def _compute_auc(precision: np.ndarray, recall: np.ndarray) -> float:
    """Compute AUC using the trapezoidal rule."""
    return np.trapz(precision, recall)

def pr_auc_compute(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    normalize: bool = False
) -> Dict[str, Any]:
    """
    Compute the Precision-Recall AUC.

    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_scores : np.ndarray
        Predicted scores or probabilities.
    normalize : bool, optional
        Whether to normalize the AUC score (default: False).

    Returns:
    --------
    dict
        Dictionary containing:
        - result: float, the PR AUC score
        - metrics: dict, additional metrics if any
        - params_used: dict, parameters used in the computation
        - warnings: list, any warnings generated

    Example:
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_scores = np.array([0.1, 0.9, 0.8, 0.2])
    >>> pr_auc_compute(y_true, y_scores)
    """
    _validate_inputs(y_true, y_scores)

    precision, recall = _compute_precision_recall(y_true, y_scores)
    auc_score = _compute_auc(precision, recall)

    if normalize:
        auc_score /= (recall[-1] * precision[0])

    return {
        "result": auc_score,
        "metrics": {"precision": precision, "recall": recall},
        "params_used": {
            "normalize": normalize
        },
        "warnings": []
    }
