"""
Quantix – Module metriques_performance
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# accuracy
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("Inputs must not contain NaN values")
    if np.isinf(y_true).any() or np.isinf(y_pred).any():
        raise ValueError("Inputs must not contain infinite values")

def _compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy metric."""
    return np.mean(y_true == y_pred)

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_funcs: Dict[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute all requested metrics."""
    return {name: func(y_true, y_pred) for name, func in metric_funcs.items()}

def accuracy_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_funcs: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
    **kwargs
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute accuracy and other performance metrics.

    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth (correct) target values.
    y_pred : np.ndarray
        Estimated targets as returned by a classifier.
    metric_funcs : Optional[Dict[str, Callable]]
        Dictionary of additional metrics to compute. Default is None.
    **kwargs
        Additional keyword arguments for future extensions.

    Returns:
    --------
    Dict containing:
    - result: float, the accuracy score
    - metrics: dict of additional computed metrics
    - params_used: dict of parameters used in computation
    - warnings: list of warning messages

    Example:
    --------
    >>> y_true = np.array([0, 1, 2, 2])
    >>> y_pred = np.array([0, 1, 2, 3])
    >>> accuracy_fit(y_true, y_pred)
    {
        'result': 0.75,
        'metrics': {},
        'params_used': {'metric_funcs': None},
        'warnings': []
    }
    """
    _validate_inputs(y_true, y_pred)

    # Default metrics
    default_metrics = {
        'accuracy': _compute_accuracy(y_true, y_pred)
    }

    # Compute additional metrics if provided
    additional_metrics = {}
    if metric_funcs is not None:
        additional_metrics = _compute_metrics(y_true, y_pred, metric_funcs)

    return {
        'result': default_metrics['accuracy'],
        'metrics': additional_metrics,
        'params_used': {
            'metric_funcs': metric_funcs
        },
        'warnings': []
    }

################################################################################
# precision
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def precision_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold: float = 0.5,
    sample_weight: Optional[np.ndarray] = None,
    normalize: bool = False
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Calculate precision metric for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (correct) target values.
    y_pred : np.ndarray
        Estimated targets as returned by a classifier.
    threshold : float, default=0.5
        The decision threshold for converting probabilities to binary predictions.
    sample_weight : np.ndarray, optional
        Sample weights.
    normalize : bool, default=False
        If True, return the normalized version of the precision.

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': float, the precision score
        - 'metrics': dict, additional metrics if any
        - 'params_used': dict, parameters used in the computation
        - 'warnings': str or None, warnings if any

    Examples
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0.1, 0.9, 0.8, 0.3])
    >>> precision_fit(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred, sample_weight)

    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Calculate precision
    tp, fp, _, _ = _calculate_confusion_matrix(y_true, y_pred_binary, sample_weight)
    precision_score = _compute_precision(tp, fp, normalize)

    return {
        'result': precision_score,
        'metrics': {},
        'params_used': {
            'threshold': threshold,
            'normalize': normalize
        },
        'warnings': None
    }

def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray]
) -> None:
    """Validate input arrays."""
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if not np.issubdtype(y_true.dtype, np.integer):
        raise ValueError("y_true must be integer type")
    if not (np.all(y_true == 0) or np.all(y_true == 1)):
        raise ValueError("y_true must contain only 0 and 1")
    if sample_weight is not None:
        if y_true.shape != sample_weight.shape:
            raise ValueError("y_true and sample_weight must have the same shape")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative values")

def _calculate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray]
) -> tuple:
    """Calculate confusion matrix components."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if sample_weight is not None:
        tp = np.sum((y_true == 1) & (y_pred == 1) * sample_weight)
        fp = np.sum((y_true == 0) & (y_pred == 1) * sample_weight)
        fn = np.sum((y_true == 1) & (y_pred == 0) * sample_weight)

    return tp, fp, fn

def _compute_precision(
    tp: float,
    fp: float,
    normalize: bool
) -> float:
    """Compute precision score."""
    if tp + fp == 0:
        return 0.0

    precision = tp / (tp + fp)

    if normalize:
        precision /= np.sum(y_true == 1) if sample_weight is None else np.sum(sample_weight[y_true == 1])

    return precision

################################################################################
# recall
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays for recall computation."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or inf values")
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError("y_pred contains NaN or inf values")

def _compute_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute recall metric."""
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0.0
    return recall

def recall_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute recall metric between true and predicted labels.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of true binary labels (0 or 1)
    y_pred : np.ndarray
        Array of predicted binary labels (0 or 1)
    metric_func : Optional[Callable]
        Custom metric function. If None, uses default recall computation.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - "result": computed recall value
        - "metrics": dictionary of metrics (only contains 'recall' here)
        - "params_used": parameters used in computation
        - "warnings": list of warnings (empty if no warnings)

    Example:
    --------
    >>> y_true = np.array([1, 0, 1, 1])
    >>> y_pred = np.array([1, 0, 0, 1])
    >>> recall_fit(y_true, y_pred)
    {
        'result': 0.666...,
        'metrics': {'recall': 0.666...},
        'params_used': {},
        'warnings': []
    }
    """
    _validate_inputs(y_true, y_pred)

    params_used = {}
    warnings_list = []

    if metric_func is None:
        recall_value = _compute_recall(y_true, y_pred)
    else:
        try:
            recall_value = metric_func(y_true, y_pred)
        except Exception as e:
            raise ValueError(f"Custom metric function failed: {str(e)}")

    return {
        "result": recall_value,
        "metrics": {"recall": recall_value},
        "params_used": params_used,
        "warnings": warnings_list
    }

################################################################################
# f1_score
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

def _compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Compute confusion matrix components."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

def _compute_f1_score(tp: int, fp: int, tn: int, fn: int) -> float:
    """Compute F1 score from confusion matrix components."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def f1_score_compute(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "binary",
    pos_label: int = 1,
    sample_weight: Optional[np.ndarray] = None
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Compute F1 score.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (correct) target values.
    y_pred : np.ndarray
        Estimated targets as returned by a classifier.
    average : str, optional (default="binary")
        This parameter is not used in binary classification.
    pos_label : int, optional (default=1)
        The class to be considered as positive.
    sample_weight : np.ndarray, optional
        Sample weights.

    Returns
    -------
    dict
        Dictionary containing:
        - "result": computed F1 score
        - "metrics": additional metrics (precision, recall)
        - "params_used": parameters used
        - "warnings": any warnings

    Example
    -------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> f1_score_compute(y_true, y_pred)
    {
        "result": 0.5,
        "metrics": {"precision": 1.0, "recall": 0.5},
        "params_used": {"average": "binary", "pos_label": 1, "sample_weight": None},
        "warnings": []
    }
    """
    _validate_inputs(y_true, y_pred)

    if sample_weight is not None:
        if len(sample_weight) != len(y_true):
            raise ValueError("sample_weight must have the same length as y_true")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight must be non-negative")

    cm = _compute_confusion_matrix(y_true, y_pred)
    f1 = _compute_f1_score(cm["tp"], cm["fp"], cm["tn"], cm["fn"])

    precision = cm["tp"] / (cm["tp"] + cm["fp"]) if (cm["tp"] + cm["fp"]) > 0 else 0
    recall = cm["tp"] / (cm["tp"] + cm["fn"]) if (cm["tp"] + cm["fn"]) > 0 else 0

    result = {
        "result": float(f1),
        "metrics": {"precision": precision, "recall": recall},
        "params_used": {
            "average": average,
            "pos_label": pos_label,
            "sample_weight": sample_weight
        },
        "warnings": []
    }

    return result

################################################################################
# roc_auc
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(y_true: np.ndarray, y_scores: np.ndarray) -> None:
    """Validate input arrays for ROC AUC calculation."""
    if y_true.ndim != 1 or y_scores.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must have the same length.")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or infinite values.")
    if np.any(np.isnan(y_scores)) or np.any(np.isinf(y_scores)):
        raise ValueError("y_scores contains NaN or infinite values.")
    if not np.array_equal(np.unique(y_true), [0, 1]):
        raise ValueError("y_true must contain only 0 and 1.")

def _compute_roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> tuple:
    """Compute ROC curve points."""
    # Sort scores and get corresponding true labels
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]

    # Compute cumulative sums
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)

    # Compute rates
    tpr = tp / tp[-1] if tp[-1] > 0 else np.zeros_like(tp)
    fpr = fp / (len(y_true) - tp[-1]) if len(y_true) - tp[-1] > 0 else np.zeros_like(fp)

    return fpr, tpr

def _compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Compute AUC using trapezoidal rule."""
    return np.trapz(tpr, fpr)

def roc_auc_compute(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute ROC AUC score.

    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_scores : np.ndarray
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions.
    sample_weight : np.ndarray, optional
        Sample weights.

    Returns:
    --------
    dict
        Dictionary containing:
        - "result": float, the ROC AUC score
        - "metrics": dict of additional metrics
        - "params_used": dict of parameters used
        - "warnings": dict of warnings

    Example:
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> roc_auc_compute(y_true, y_scores)
    """
    # Validate inputs
    _validate_inputs(y_true, y_scores)

    # Compute ROC curve and AUC
    fpr, tpr = _compute_roc_curve(y_true, y_scores)
    auc_score = _compute_auc(fpr, tpr)

    # Prepare output
    result = {
        "result": auc_score,
        "metrics": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        },
        "params_used": {
            "sample_weight": sample_weight is not None
        },
        "warnings": {}
    }

    return result

################################################################################
# pr_auc
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

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

def _compute_pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute PR AUC using trapezoidal rule."""
    # Sort scores and get corresponding true labels
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]

    # Compute precision and recall at each threshold
    unique_thresholds = np.unique(y_scores_sorted)
    precisions = []
    recalls = []

    for threshold in unique_thresholds:
        y_pred = (y_scores_sorted >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true_sorted == 1))
        fp = np.sum((y_pred == 1) & (y_true_sorted == 0))
        fn = np.sum((y_pred == 0) & (y_true_sorted == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    # Add point at (0,1) and (1,0)
    precisions = [1] + precisions
    recalls = [1] + recalls

    # Compute AUC using trapezoidal rule
    auc = np.trapz(recalls, precisions)
    return auc

def pr_auc_fit(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    *,
    normalization: Optional[str] = None,
    metric: str = "pr_auc",
    solver: str = "default",
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Union[float, Dict]]:
    """
    Compute Precision-Recall AUC.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_scores : np.ndarray
        Estimated probabilities or decision scores.
    normalization : str, optional
        Normalization method (not used for PR AUC).
    metric : str, optional
        Metric to compute ("pr_auc" by default).
    solver : str, optional
        Solver method (not used for PR AUC).
    custom_metric : callable, optional
        Custom metric function.
    **kwargs :
        Additional keyword arguments.

    Returns
    -------
    Dict[str, Union[float, Dict]]
        Dictionary containing:
        - "result": computed PR AUC
        - "metrics": additional metrics (empty for PR AUC)
        - "params_used": parameters used
        - "warnings": any warnings

    Example
    -------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_scores = np.array([0.1, 0.9, 0.8, 0.2])
    >>> result = pr_auc_fit(y_true, y_scores)
    """
    _validate_inputs(y_true, y_scores)

    params_used = {
        "normalization": normalization,
        "metric": metric,
        "solver": solver
    }

    if custom_metric is not None:
        result = custom_metric(y_true, y_scores)
    else:
        if metric != "pr_auc":
            raise ValueError("Only 'pr_auc' metric is supported in this function")
        result = _compute_pr_auc(y_true, y_scores)

    warnings = []

    return {
        "result": result,
        "metrics": {},
        "params_used": params_used,
        "warnings": warnings
    }

################################################################################
# mse
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("Inputs must not contain NaN values")
    if np.isinf(y_true).any() or np.isinf(y_pred).any():
        raise ValueError("Inputs must not contain infinite values")

def _compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def mse_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute Mean Squared Error (MSE) between true and predicted values.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    metric_func : Callable, optional
        Custom metric function. If None, uses default MSE.

    Returns:
    --------
    dict
        Dictionary containing:
        - "result": computed MSE value
        - "metrics": dictionary of metrics (only MSE in this case)
        - "params_used": parameters used
        - "warnings": any warnings encountered

    Example:
    --------
    >>> y_true = np.array([1, 2, 3])
    >>> y_pred = np.array([1.1, 1.9, 3.2])
    >>> mse_fit(y_true, y_pred)
    {
        'result': 0.023333333333333334,
        'metrics': {'mse': 0.023333333333333334},
        'params_used': {'metric_func': None},
        'warnings': []
    }
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Set default metric function if not provided
    if metric_func is None:
        metric_func = _compute_mse

    # Compute metrics
    mse_value = metric_func(y_true, y_pred)

    return {
        "result": mse_value,
        "metrics": {"mse": mse_value},
        "params_used": {"metric_func": metric_func.__name__ if metric_func is not None else None},
        "warnings": []
    }

################################################################################
# rmse
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("Inputs must not contain NaN values")
    if np.isinf(y_true).any() or np.isinf(y_pred).any():
        raise ValueError("Inputs must not contain infinite values")

def _compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE between true and predicted values."""
    squared_error = (y_true - y_pred) ** 2
    mse = np.mean(squared_error)
    return np.sqrt(mse)

def rmse_compute(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute RMSE (Root Mean Squared Error) between true and predicted values.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    metric_func : Optional[Callable]
        Custom metric function. If None, uses RMSE.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - "result": computed RMSE value
        - "metrics": dictionary of metrics (currently just RMSE)
        - "params_used": parameters used in computation
        - "warnings": any warnings generated

    Example:
    --------
    >>> y_true = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5, 0.0, 2, 8])
    >>> rmse_compute(y_true, y_pred)
    {
        'result': 0.6123724356957945,
        'metrics': {'rmse': 0.6123724356957945},
        'params_used': {},
        'warnings': []
    }
    """
    _validate_inputs(y_true, y_pred)

    params_used = {}
    warnings_list = []

    if metric_func is None:
        result = _compute_rmse(y_true, y_pred)
    else:
        try:
            result = metric_func(y_true, y_pred)
        except Exception as e:
            warnings_list.append(f"Custom metric function failed: {str(e)}")
            result = _compute_rmse(y_true, y_pred)

    metrics = {'rmse': result}

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings_list
    }

################################################################################
# mae
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays for MAE calculation."""
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Inputs must not contain NaN values.")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Inputs must not contain infinite values.")

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error (MAE)."""
    return np.mean(np.abs(y_true - y_pred))

def mae_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "mae",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> Dict[str, Any]:
    """
    Compute performance metrics including MAE.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    metric : str, optional
        Metric to compute. Default is "mae".
    custom_metric : Callable, optional
        Custom metric function. If provided, overrides the default metric.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    validate_inputs(y_true, y_pred)

    result = {}
    metrics = {}
    params_used = {
        "metric": metric,
        "custom_metric": custom_metric is not None
    }
    warnings = []

    # Compute requested metrics
    if metric == "mae" or custom_metric is None:
        mae_value = compute_mae(y_true, y_pred)
        metrics["mae"] = mae_value

    if custom_metric is not None:
        try:
            custom_value = custom_metric(y_true, y_pred)
            metrics["custom"] = custom_value
        except Exception as e:
            warnings.append(f"Custom metric computation failed: {str(e)}")

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

# Example usage:
# mae_fit(np.array([1, 2, 3]), np.array([1.1, 2.9, 3.0]))

################################################################################
# r2_score
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("Inputs must not contain NaN values")
    if np.isinf(y_true).any() or np.isinf(y_pred).any():
        raise ValueError("Inputs must not contain infinite values")

def _compute_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    r2 = 1 - (ss_res / ss_tot)
    return r2

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute additional metrics."""
    return {
        "r2_score": _compute_r2_score(y_true, y_pred),
        "mse": np.mean((y_true - y_pred) ** 2),
        "mae": np.mean(np.abs(y_true - y_pred))
    }

def r2_score_compute(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute R² score and additional metrics.

    Parameters:
    -----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    metric_func : Optional[Callable]
        Custom metric function. If None, uses default R² score.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, and warnings.

    Example:
    --------
    >>> y_true = np.array([1, 2, 3])
    >>> y_pred = np.array([1.1, 1.9, 3.2])
    >>> r2_score_compute(y_true, y_pred)
    """
    _validate_inputs(y_true, y_pred)

    if metric_func is None:
        r2 = _compute_r2_score(y_true, y_pred)
    else:
        r2 = metric_func(y_true, y_pred)

    metrics = _compute_metrics(y_true, y_pred)
    if metric_func is not None:
        metrics["custom_metric"] = r2

    return {
        "result": {"r2_score": r2},
        "metrics": metrics,
        "params_used": {
            "metric_func": metric_func.__name__ if metric_func else "default_r2"
        },
        "warnings": []
    }

################################################################################
# log_loss
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays for log loss calculation."""
    if y_true.ndim != 1 or y_pred.ndim != 2:
        raise ValueError("y_true must be 1D and y_pred must be 2D")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of samples")
    if np.any(y_pred < 0) or np.any(y_pred > 1):
        raise ValueError("Predicted probabilities must be between 0 and 1")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or Inf values")
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError("y_pred contains NaN or Inf values")

def _compute_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the log loss (cross-entropy loss)."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def log_loss_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = False,
    sample_weight: Optional[np.ndarray] = None,
    epsilon: float = 1e-15
) -> Dict[str, Any]:
    """
    Compute the log loss (cross-entropy loss).

    Parameters:
    -----------
    y_true : np.ndarray
        True labels (one-hot encoded for multiclass).
    y_pred : np.ndarray
        Predicted probabilities.
    normalize : bool, optional
        Whether to normalize the log loss by the number of samples.
    sample_weight : np.ndarray, optional
        Individual weights for each sample.
    epsilon : float, optional
        Small value to avoid log(0).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the result, metrics, parameters used, and warnings.

    Example:
    --------
    >>> y_true = np.array([[1, 0], [0, 1]])
    >>> y_pred = np.array([[0.9, 0.1], [0.2, 0.8]])
    >>> result = log_loss_fit(y_true, y_pred)
    """
    _validate_inputs(y_true, y_pred)

    if sample_weight is not None:
        if len(sample_weight) != y_true.shape[0]:
            raise ValueError("sample_weight must have the same length as y_true")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight must be non-negative")

    logloss = _compute_log_loss(y_true, y_pred)

    if normalize:
        logloss /= y_true.shape[0]

    if sample_weight is not None:
        logloss = np.average(logloss, weights=sample_weight)

    return {
        "result": logloss,
        "metrics": {"log_loss": logloss},
        "params_used": {
            "normalize": normalize,
            "sample_weight": sample_weight is not None,
            "epsilon": epsilon
        },
        "warnings": []
    }

################################################################################
# brier_score
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_prob, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or inf values")
    if np.any(np.isnan(y_prob)) or np.any(np.isinf(y_prob)):
        raise ValueError("y_prob contains NaN or inf values")
    if not np.allclose(y_true, y_true.astype(bool)):
        raise ValueError("y_true must contain only binary values (0 or 1)")

def _compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the Brier score."""
    return np.mean((y_true - y_prob) ** 2)

def brier_score_fit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    normalize: bool = False
) -> Dict[str, Any]:
    """
    Compute the Brier score for probabilistic binary classification.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of true binary labels (0 or 1)
    y_prob : np.ndarray
        Array of predicted probabilities for class 1
    metric_func : Callable, optional
        Custom metric function. If None, uses default Brier score.
    normalize : bool, optional
        Whether to normalize the metric (default: False)

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings

    Example:
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_prob = np.array([0.1, 0.9, 0.8, 0.2])
    >>> result = brier_score_fit(y_true, y_prob)
    """
    # Validate inputs
    _validate_inputs(y_true, y_prob)

    # Use custom metric if provided, otherwise use default Brier score
    compute_func = metric_func if metric_func is not None else _compute_brier_score

    # Compute the score
    score = compute_func(y_true, y_prob)

    # Normalize if requested
    if normalize:
        score = score / np.mean(y_true)

    return {
        "result": {"brier_score": score},
        "metrics": {"score": score},
        "params_used": {
            "metric_func": metric_func.__name__ if metric_func else "_compute_brier_score",
            "normalize": normalize
        },
        "warnings": []
    }

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
    normalize: Optional[str] = None
) -> np.ndarray:
    """Compute confusion matrix with optional normalization."""
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))

    if normalize is not None:
        if normalize == 'true':
            cm = cm / cm.sum(axis=1, keepdims=True)
        elif normalize == 'pred':
            cm = cm / cm.sum(axis=0, keepdims=True)
        elif normalize == 'all':
            cm = cm / cm.sum()
        else:
            raise ValueError("normalize must be 'true', 'pred', 'all' or None")

    return cm

def _compute_metrics(
    confusion_matrix: np.ndarray,
    metrics: Optional[Dict[str, Callable]] = None
) -> Dict[str, float]:
    """Compute performance metrics from confusion matrix."""
    if metrics is None:
        return {}

    results = {}
    for name, metric_func in metrics.items():
        try:
            results[name] = metric_func(confusion_matrix)
        except Exception as e:
            results[name] = np.nan
            print(f"Warning: Could not compute metric {name}: {str(e)}")

    return results

def confusion_matrix_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: Optional[str] = None,
    metrics: Optional[Dict[str, Callable]] = None
) -> Dict[str, Any]:
    """
    Compute confusion matrix and performance metrics.

    Parameters:
    - y_true: Ground truth labels
    - y_pred: Predicted labels
    - normalize: Normalization method ('true', 'pred', 'all' or None)
    - metrics: Dictionary of metric names and callable functions

    Returns:
    - Dictionary containing confusion matrix, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Compute confusion matrix
    cm = _compute_confusion_matrix(y_true, y_pred, normalize)

    # Compute metrics
    computed_metrics = _compute_metrics(cm, metrics)

    return {
        "result": cm,
        "metrics": computed_metrics,
        "params_used": {
            "normalize": normalize,
            "metrics": list(metrics.keys()) if metrics else None
        },
        "warnings": []
    }

# Example usage:
"""
y_true = np.array([0, 1, 2, 0, 1])
y_pred = np.array([0, 1, 1, 0, 2])

def accuracy(cm: np.ndarray) -> float:
    return np.trace(cm) / cm.sum()

metrics = {
    "accuracy": accuracy
}

result = confusion_matrix_fit(y_true, y_pred, normalize='true', metrics=metrics)
"""

################################################################################
# precision_recall_curve
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def precision_recall_curve_fit(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = 'binary',
    solver: str = 'default',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Compute the precision-recall curve for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_scores : np.ndarray
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the scores, by default None.
    metric : str, optional
        Metric to use for evaluation ('binary', 'micro', 'macro'), by default 'binary'.
    solver : str, optional
        Solver to use ('default', 'custom'), by default 'default'.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function, by default None.
    tol : float, optional
        Tolerance for stopping criteria, by default 1e-4.
    max_iter : int, optional
        Maximum number of iterations, by default 1000.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
            - 'precision': Precision values.
            - 'recall': Recall values.
            - 'thresholds': Thresholds used.
            - 'metrics': Additional metrics.
            - 'params_used': Parameters used in the computation.
            - 'warnings': Any warnings encountered.

    Examples
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> result = precision_recall_curve_fit(y_true, y_scores)
    """
    # Validate inputs
    _validate_inputs(y_true, y_scores)

    # Normalize scores if a normalizer is provided
    if normalizer is not None:
        y_scores = normalizer(y_scores)

    # Compute precision-recall curve
    precision, recall, thresholds = _compute_precision_recall_curve(y_true, y_scores)

    # Compute additional metrics
    metrics = _compute_metrics(y_true, y_scores, metric, custom_metric)

    # Prepare output
    result = {
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'solver': solver,
            'custom_metric': custom_metric.__name__ if custom_metric else None,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return result

def _validate_inputs(y_true: np.ndarray, y_scores: np.ndarray) -> None:
    """
    Validate the inputs for precision-recall curve computation.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_scores : np.ndarray
        Target scores.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if y_true.shape != y_scores.shape:
        raise ValueError("y_true and y_scores must have the same shape.")
    if np.any((y_true != 0) & (y_true != 1)):
        raise ValueError("y_true must contain only 0 and 1.")
    if np.any(np.isnan(y_scores)):
        raise ValueError("y_scores must not contain NaN values.")
    if np.any(np.isinf(y_scores)):
        raise ValueError("y_scores must not contain infinite values.")

def _compute_precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the precision-recall curve.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_scores : np.ndarray
        Target scores.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Precision values, recall values, and thresholds.
    """
    # Sort scores in descending order
    desc_score_indices = np.argsort(y_scores, kind="mergesort")[::-1]
    y_scores = y_scores[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # Compute thresholds
    thresholds = np.r_[y_scores.min(), y_scores[np.diff(y_true)]]

    # Compute precision and recall
    tp = np.cumsum(y_true)
    fp = np.cumsum(~y_true)
    precision = tp / (tp + fp)
    recall = tp / tp[-1]

    # Remove duplicate thresholds and corresponding precision/recall values
    unique_thresholds = np.unique(thresholds)
    precision = precision[np.searchsorted(y_scores, unique_thresholds)]
    recall = recall[np.searchsorted(y_scores, unique_thresholds)]

    return precision, recall, unique_thresholds

def _compute_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """
    Compute additional metrics based on the specified metric.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_scores : np.ndarray
        Target scores.
    metric : str
        Metric to use for evaluation ('binary', 'micro', 'macro').
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function.

    Returns
    -------
    Dict[str, float]
        Dictionary containing computed metrics.
    """
    metrics = {}

    if metric == 'binary':
        # Binary classification metrics
        precision, recall, _ = _compute_precision_recall_curve(y_true, y_scores)
        metrics['average_precision'] = np.trapz(recall, precision)

    elif metric == 'micro':
        # Micro-average metrics
        pass  # Implement micro-average logic if needed

    elif metric == 'macro':
        # Macro-average metrics
        pass  # Implement macro-average logic if needed

    if custom_metric is not None:
        metrics['custom_metric'] = custom_metric(y_true, y_scores)

    return metrics

################################################################################
# roc_curve
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def roc_curve_compute(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    normalization: str = 'none',
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = 'auc',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> Dict[str, Union[Dict[str, np.ndarray], float, Dict[str, str]]]:
    """
    Compute ROC curve and associated metrics.

    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_scores : np.ndarray
        Predicted scores or probabilities.
    normalization : str, optional (default='none')
        Normalization method for scores: 'none', 'standard', 'minmax', or 'robust'.
    custom_normalization : callable, optional
        Custom normalization function.
    metric : str, optional (default='auc')
        Metric to compute: 'auc', 'fpr', 'tpr', or 'accuracy'.
    custom_metric : callable, optional
        Custom metric function.

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': ROC curve data (fpr, tpr, thresholds)
        - 'metrics': Computed metrics
        - 'params_used': Parameters used in computation
        - 'warnings': Any warnings generated

    Example:
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_scores = np.array([0.1, 0.9, 0.8, 0.2])
    >>> result = roc_curve_compute(y_true, y_scores)
    """
    # Validate inputs
    _validate_inputs(y_true, y_scores)

    # Normalize scores if required
    normalized_scores = _normalize_scores(y_scores, normalization, custom_normalization)

    # Compute ROC curve
    fpr, tpr, thresholds = _compute_roc_curve(y_true, normalized_scores)

    # Compute metrics
    metrics = {}
    if metric == 'auc':
        metrics['auc'] = _compute_auc(fpr, tpr)
    elif metric == 'fpr':
        metrics['fpr'] = fpr
    elif metric == 'tpr':
        metrics['tpr'] = tpr
    elif metric == 'accuracy':
        metrics['accuracy'] = _compute_accuracy(y_true, normalized_scores)
    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, normalized_scores)

    return {
        'result': {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds},
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric
        },
        'warnings': []
    }

def _validate_inputs(y_true: np.ndarray, y_scores: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_scores, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_scores.shape:
        raise ValueError("y_true and y_scores must have the same shape")
    if np.any((y_true != 0) & (y_true != 1)):
        raise ValueError("y_true must contain only 0s and 1s")
    if np.any(~np.isfinite(y_scores)):
        raise ValueError("y_scores must contain only finite values")

def _normalize_scores(
    scores: np.ndarray,
    normalization: str,
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Normalize scores based on specified method."""
    if custom_normalization is not None:
        return custom_normalization(scores)

    if normalization == 'standard':
        mean = np.mean(scores)
        std = np.std(scores)
        if std == 0:
            return scores
        return (scores - mean) / std
    elif normalization == 'minmax':
        min_val = np.min(scores)
        max_val = np.max(scores)
        if max_val == min_val:
            return scores
        return (scores - min_val) / (max_val - min_val)
    elif normalization == 'robust':
        median = np.median(scores)
        iqr = np.percentile(scores, 75) - np.percentile(scores, 25)
        if iqr == 0:
            return scores
        return (scores - median) / iqr
    else:  # 'none'
        return scores

def _compute_roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve (fpr, tpr) and thresholds."""
    # Get unique thresholds in descending order
    thresholds = np.unique(y_scores)[::-1]

    # Initialize arrays
    fpr = np.zeros_like(thresholds)
    tpr = np.zeros_like(thresholds)

    # Compute FPR and TPR for each threshold
    for i, threshold in enumerate(thresholds):
        y_pred = (y_scores >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        fpr[i] = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr[i] = tp / (tp + fn) if (tp + fn) > 0 else 0

    return fpr, tpr, thresholds

def _compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Compute Area Under the ROC Curve."""
    return np.trapz(tpr, fpr)

def _compute_accuracy(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute accuracy at optimal threshold."""
    thresholds = np.unique(y_scores)
    best_threshold = None
    best_accuracy = 0

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        accuracy = np.mean(y_pred == y_true)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_accuracy if best_threshold is not None else 0
