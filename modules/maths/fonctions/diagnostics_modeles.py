"""
Quantix – Module diagnostics_modeles
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# matrice_confusion
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or inf values")
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError("y_pred contains NaN or inf values")

def _compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                             normalize: Optional[str] = None) -> np.ndarray:
    """Compute confusion matrix with optional normalization."""
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)

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
    return cm

def _compute_metrics(cm: np.ndarray) -> Dict[str, float]:
    """Compute common metrics from confusion matrix."""
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    metrics = {
        'accuracy': np.sum(tp) / np.sum(cm),
        'precision': tp / (tp + fp + 1e-9),
        'recall': tp / (tp + fn + 1e-9),
        'f1_score': 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'] + 1e-9),
        'specificity': tn / (tn + fp + 1e-9)
    }
    return metrics

def matrice_confusion_fit(y_true: np.ndarray, y_pred: np.ndarray,
                          normalize: Optional[str] = None,
                          metrics: bool = True,
                          custom_metric: Optional[Callable] = None) -> Dict:
    """
    Compute confusion matrix with optional normalization and metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (correct) target values.
    y_pred : np.ndarray
        Estimated targets as returned by a classifier.
    normalize : str, optional
        Normalization method ('true', 'pred', 'all', or None).
    metrics : bool, optional
        Whether to compute additional metrics.
    custom_metric : callable, optional
        Custom metric function that takes confusion matrix as input.

    Returns
    -------
    dict
        Dictionary containing:
        - 'confusion_matrix': computed confusion matrix
        - 'metrics': dictionary of metrics (if requested)
        - 'params_used': parameters used
        - 'warnings': any warnings generated

    Example
    -------
    >>> y_true = np.array([0, 1, 2, 2])
    >>> y_pred = np.array([0, 0, 2, 2])
    >>> result = matrice_confusion_fit(y_true, y_pred)
    """
    _validate_inputs(y_true, y_pred)

    cm = _compute_confusion_matrix(y_true, y_pred, normalize=normalize)
    metrics_dict = {}
    warnings_list = []

    if metrics:
        try:
            metrics_dict = _compute_metrics(cm)
        except Exception as e:
            warnings_list.append(f"Metrics computation failed: {str(e)}")

    if custom_metric is not None:
        try:
            metrics_dict['custom'] = custom_metric(cm)
        except Exception as e:
            warnings_list.append(f"Custom metric computation failed: {str(e)}")

    return {
        'confusion_matrix': cm,
        'metrics': metrics_dict if metrics else None,
        'params_used': {
            'normalize': normalize,
            'metrics': metrics,
            'custom_metric': custom_metric.__name__ if custom_metric else None
        },
        'warnings': warnings_list
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
    normalize: Optional[str] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "accuracy",
    sample_weight: Optional[np.ndarray] = None,
    custom_metric_kwargs: Optional[Dict] = None
) -> Dict[str, Union[float, Dict]]:
    """
    Calculate precision metrics for classification models.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (correct) target values.
    y_pred : np.ndarray
        Estimated targets as returned by a classifier.
    threshold : float, default=0.5
        Decision threshold for converting probabilities to binary predictions.
    normalize : str, optional
        Normalization method. Options: None, 'standard', 'minmax'.
    metric : str or callable, default="accuracy"
        Metric to compute. Options: "precision", "recall", "f1", or custom callable.
    sample_weight : np.ndarray, optional
        Individual weights for each sample.
    custom_metric_kwargs : dict, optional
        Additional keyword arguments for custom metric function.

    Returns
    -------
    Dict[str, Union[float, Dict]]
        Dictionary containing:
        - "result": computed precision value
        - "metrics": additional metrics if applicable
        - "params_used": parameters used in computation
        - "warnings": any warnings generated

    Examples
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0.1, 0.9, 0.8, 0.2])
    >>> precision_fit(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred, sample_weight)

    # Convert probabilities to binary predictions
    y_pred_binary = _apply_threshold(y_pred, threshold)

    # Calculate precision
    result = _calculate_precision(
        y_true,
        y_pred_binary,
        sample_weight=sample_weight
    )

    # Calculate additional metrics if requested
    metrics = _calculate_additional_metrics(
        y_true,
        y_pred_binary,
        metric=metric,
        sample_weight=sample_weight,
        custom_metric_kwargs=custom_metric_kwargs
    )

    # Normalize if requested
    if normalize is not None:
        result, metrics = _apply_normalization(result, metrics, method=normalize)

    return {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "threshold": threshold,
            "normalize": normalize,
            "metric": metric,
            "sample_weight": sample_weight is not None
        },
        "warnings": []
    }

def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray]
) -> None:
    """Validate input arrays."""
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if sample_weight is not None:
        if y_true.shape != sample_weight.shape:
            raise ValueError("y_true and sample_weight must have the same shape")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative values")

def _apply_threshold(
    y_pred: np.ndarray,
    threshold: float
) -> np.ndarray:
    """Convert probability predictions to binary using threshold."""
    return (y_pred >= threshold).astype(int)

def _calculate_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> float:
    """Calculate precision metric."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if sample_weight is not None:
        tp = np.sum((y_true == 1) & (y_pred == 1) * sample_weight)
        fp = np.sum((y_true == 0) & (y_pred == 1) * sample_weight)
        total_pos = np.sum((y_true == 1) * sample_weight)
    else:
        total_pos = np.sum(y_true == 1)

    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def _calculate_additional_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    sample_weight: Optional[np.ndarray] = None,
    custom_metric_kwargs: Optional[Dict] = None
) -> Dict[str, float]:
    """Calculate additional metrics based on user request."""
    metrics = {}

    if isinstance(metric, str):
        if metric == "precision":
            pass  # already calculated
        elif metric == "recall":
            metrics["recall"] = _calculate_recall(y_true, y_pred, sample_weight)
        elif metric == "f1":
            metrics["recall"] = _calculate_recall(y_true, y_pred, sample_weight)
            metrics["f1"] = 2 * (metrics["recall"] * _calculate_precision(y_true, y_pred, sample_weight)) / (metrics["recall"] + _calculate_precision(y_true, y_pred, sample_weight))
    else:
        if custom_metric_kwargs is None:
            metrics["custom"] = metric(y_true, y_pred)
        else:
            metrics["custom"] = metric(y_true, y_pred, **custom_metric_kwargs)

    return metrics

def _calculate_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> float:
    """Calculate recall metric."""
    tp = np.sum((y_true == 1) & (y_pred == 1))

    if sample_weight is not None:
        tp = np.sum((y_true == 1) & (y_pred == 1) * sample_weight)
        total_pos = np.sum((y_true == 1) * sample_weight)
    else:
        total_pos = np.sum(y_true == 1)

    if total_pos == 0:
        return 0.0
    return tp / total_pos

def _apply_normalization(
    result: float,
    metrics: Dict[str, float],
    method: str
) -> tuple[float, Dict[str, float]]:
    """Apply normalization to results."""
    if method == "standard":
        # Example: standardize the result
        pass  # implementation depends on specific normalization needs
    elif method == "minmax":
        # Example: min-max scaling
        pass  # implementation depends on specific normalization needs

    return result, metrics

################################################################################
# rappel
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def rappel_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: Union[str, Callable] = 'mse',
    normalization: str = 'none',
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Union[float, dict]]:
    """
    Calculate the recall metric between true and predicted values with various options.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    metric : str or callable, optional
        Metric to use for evaluation. Options: 'mse', 'mae', 'r2', or custom callable.
    normalization : str, optional
        Normalization method. Options: 'none', 'standard', 'minmax', 'robust'.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    **kwargs :
        Additional keyword arguments passed to the metric function.

    Returns
    -------
    Dict[str, Union[float, dict]]
        Dictionary containing:
        - "result": The computed recall value.
        - "metrics": Additional metrics if applicable.
        - "params_used": Parameters used in the computation.
        - "warnings": Any warnings generated during computation.

    Examples
    --------
    >>> y_true = np.array([1, 0, 1, 1])
    >>> y_pred = np.array([1, 0, 0, 1])
    >>> recall_fit(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Normalize data if required
    y_true_norm, y_pred_norm = _apply_normalization(y_true, y_pred, normalization)

    # Compute recall
    result = _compute_recall(y_true_norm, y_pred_norm, metric, custom_metric, **kwargs)

    # Prepare output
    output = {
        "result": result,
        "metrics": {},
        "params_used": {
            "metric": metric,
            "normalization": normalization
        },
        "warnings": []
    }

    return output

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Input arrays must not contain infinite values.")

def _apply_normalization(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalization: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply specified normalization to input arrays."""
    if normalization == 'standard':
        mean = np.mean(y_true)
        std = np.std(y_true)
        y_true_norm = (y_true - mean) / std
        y_pred_norm = (y_pred - mean) / std
    elif normalization == 'minmax':
        min_val = np.min(y_true)
        max_val = np.max(y_true)
        y_true_norm = (y_true - min_val) / (max_val - min_val)
        y_pred_norm = (y_pred - min_val) / (max_val - min_val)
    elif normalization == 'robust':
        median = np.median(y_true)
        iqr = np.percentile(y_true, 75) - np.percentile(y_true, 25)
        y_true_norm = (y_true - median) / iqr
        y_pred_norm = (y_pred - median) / iqr
    else:
        y_true_norm, y_pred_norm = y_true, y_pred
    return y_true_norm, y_pred_norm

def _compute_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> float:
    """Compute recall using specified metric."""
    if callable(metric):
        return metric(y_true, y_pred, **kwargs)
    elif custom_metric is not None:
        return custom_metric(y_true, y_pred, **kwargs)
    elif metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    else:
        raise ValueError(f"Unknown metric: {metric}")

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
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or infinite values")
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError("y_pred contains NaN or infinite values")

def _calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                               threshold: float = 0.5) -> Dict[str, int]:
    """Calculate confusion matrix components."""
    y_pred_bin = (y_pred >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred_bin == 1))
    fp = np.sum((y_true == 0) & (y_pred_bin == 1))
    tn = np.sum((y_true == 0) & (y_pred_bin == 0))
    fn = np.sum((y_true == 1) & (y_pred_bin == 0))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

def _compute_f1_score(confusion_matrix: Dict[str, int],
                      beta: float = 1.0) -> float:
    """Compute F1 score from confusion matrix."""
    precision = confusion_matrix["tp"] / (confusion_matrix["tp"] + confusion_matrix["fp"])
    recall = confusion_matrix["tp"] / (confusion_matrix["tp"] + confusion_matrix["fn"])
    if precision + recall == 0:
        return 0.0
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    return f1

def f1_score_compute(y_true: np.ndarray, y_pred: np.ndarray,
                     threshold: float = 0.5, beta: float = 1.0) -> Dict[str, Union[float, Dict]]:
    """
    Compute F1 score between true and predicted labels.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true binary labels (0 or 1)
    y_pred : np.ndarray
        Array of predicted probabilities or binary labels (0 or 1)
    threshold : float, optional
        Threshold for converting probabilities to binary predictions (default: 0.5)
    beta : float, optional
        Weighting factor for recall in F1 score (default: 1.0)

    Returns
    -------
    dict
        Dictionary containing:
        - "result": computed F1 score
        - "metrics": confusion matrix components
        - "params_used": parameters used in computation

    Example
    -------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0.1, 0.9, 0.8, 0.2])
    >>> f1_score_compute(y_true, y_pred)
    """
    _validate_inputs(y_true, y_pred)

    confusion_matrix = _calculate_confusion_matrix(y_true, y_pred, threshold)
    f1_score_value = _compute_f1_score(confusion_matrix, beta)

    return {
        "result": f1_score_value,
        "metrics": confusion_matrix,
        "params_used": {
            "threshold": threshold,
            "beta": beta
        },
        "warnings": []
    }

################################################################################
# auc_roc
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def auc_roc_compute(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    normalize: bool = False,
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Compute the Area Under the Receiver Operating Characteristic Curve (AUC-ROC).

    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_pred : np.ndarray
        Predicted probabilities or scores.
    normalize : bool, optional
        Whether to normalize the predicted probabilities (default: False).
    metric : Optional[Callable], optional
        Custom metric function to compute additional metrics (default: None).

    Returns:
    --------
    Dict[str, Union[float, Dict[str, float], str]]
        A dictionary containing:
        - "result": AUC-ROC score.
        - "metrics": Additional metrics if provided.
        - "params_used": Parameters used in the computation.
        - "warnings": Any warnings encountered.

    Example:
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0.1, 0.9, 0.8, 0.2])
    >>> auc_roc_compute(y_true, y_pred)
    {
        'result': 0.95,
        'metrics': {},
        'params_used': {'normalize': False, 'metric': None},
        'warnings': []
    }
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Normalize predictions if required
    if normalize:
        y_pred = _normalize_predictions(y_pred)

    # Compute AUC-ROC
    auc_score = _compute_auc_roc(y_true, y_pred)

    # Compute additional metrics if provided
    metrics = {}
    if metric is not None:
        metrics["custom_metric"] = metric(y_true, y_pred)

    # Prepare output
    result = {
        "result": auc_score,
        "metrics": metrics,
        "params_used": {"normalize": normalize, "metric": metric},
        "warnings": []
    }

    return result

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Validate the input arrays.

    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels.
    y_pred : np.ndarray
        Predicted probabilities or scores.

    Raises:
    -------
    ValueError
        If inputs are invalid.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.any((y_true != 0) & (y_true != 1)):
        raise ValueError("y_true must contain only 0s and 1s.")
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError("y_pred must not contain NaN or infinite values.")

def _normalize_predictions(y_pred: np.ndarray) -> np.ndarray:
    """
    Normalize the predicted probabilities.

    Parameters:
    -----------
    y_pred : np.ndarray
        Predicted probabilities or scores.

    Returns:
    --------
    np.ndarray
        Normalized predicted probabilities.
    """
    y_pred_min = np.min(y_pred)
    y_pred_max = np.max(y_pred)
    if y_pred_min == y_pred_max:
        return np.zeros_like(y_pred)
    return (y_pred - y_pred_min) / (y_pred_max - y_pred_min)

def _compute_auc_roc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the AUC-ROC score.

    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels.
    y_pred : np.ndarray
        Predicted probabilities or scores.

    Returns:
    --------
    float
        AUC-ROC score.
    """
    # Sort the predictions and corresponding true labels
    sorted_indices = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    # Compute the ROC curve
    tpr = np.cumsum(y_true_sorted) / np.sum(y_true)
    fpr = np.cumsum(1 - y_true_sorted) / np.sum(1 - y_true)

    # Compute the AUC using the trapezoidal rule
    auc = np.trapz(tpr, fpr)

    return auc

################################################################################
# bias_variance_decomposition
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def bias_variance_decomposition_fit(
    model: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metric: Union[str, Callable] = 'mse',
    n_bootstrap_samples: int = 100,
    normalize: Optional[str] = None,
    random_state: Optional[int] = None
) -> Dict:
    """
    Compute bias-variance decomposition for a given model.

    Parameters
    ----------
    model : Callable
        The model to evaluate. Should have fit and predict methods.
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training targets.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        True test targets.
    metric : Union[str, Callable], optional
        Metric to use for evaluation. Can be 'mse', 'mae', 'r2', or a custom callable.
        Default is 'mse'.
    n_bootstrap_samples : int, optional
        Number of bootstrap samples to use for variance estimation. Default is 100.
    normalize : Optional[str], optional
        Normalization method for features. Can be 'standard', 'minmax', or None.
        Default is None.
    random_state : Optional[int], optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Dictionary with bias, variance, and error components.
        - 'metrics': Dictionary of computed metrics.
        - 'params_used': Dictionary of parameters used.
        - 'warnings': List of warnings encountered.

    Example
    -------
    >>> from sklearn.linear_model import LinearRegression
    >>> X_train = np.random.rand(100, 5)
    >>> y_train = np.random.rand(100)
    >>> X_test = np.random.rand(20, 5)
    >>> y_test = np.random.rand(20)
    >>> result = bias_variance_decomposition_fit(
    ...     model=LinearRegression(),
    ...     X_train=X_train,
    ...     y_train=y_train,
    ...     X_test=X_test,
    ...     y_test=y_test
    ... )
    """
    # Validate inputs
    _validate_inputs(X_train, y_train, X_test, y_test)

    # Set random state
    rng = np.random.RandomState(random_state)

    # Normalize features if specified
    X_train_norm, X_test_norm = _normalize_features(X_train, X_test, normalize)

    # Fit the model
    model.fit(X_train_norm, y_train)
    y_pred = model.predict(X_test_norm)

    # Compute metrics
    metrics = _compute_metrics(y_test, y_pred, metric)

    # Bootstrap for variance estimation
    bias, variance = _bootstrap_bias_variance(
        model,
        X_train_norm,
        y_train,
        X_test_norm,
        n_bootstrap_samples,
        rng
    )

    # Calculate error components
    avg_y = np.mean(y_train)
    error_components = {
        'bias': bias,
        'variance': variance,
        'error': metrics['metric']
    }

    # Prepare output
    result = {
        'result': error_components,
        'metrics': metrics,
        'params_used': {
            'metric': metric,
            'n_bootstrap_samples': n_bootstrap_samples,
            'normalize': normalize,
            'random_state': random_state
        },
        'warnings': []
    }

    return result

def _validate_inputs(X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train must have the same number of samples.")
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError("X_test and y_test must have the same number of samples.")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("X_train and X_test must have the same number of features.")
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        raise ValueError("X_train contains NaN or infinite values.")
    if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
        raise ValueError("y_train contains NaN or infinite values.")
    if np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
        raise ValueError("X_test contains NaN or infinite values.")
    if np.any(np.isnan(y_test)) or np.any(np.isinf(y_test)):
        raise ValueError("y_test contains NaN or infinite values.")

def _normalize_features(X_train: np.ndarray, X_test: np.ndarray,
                       method: Optional[str] = None) -> tuple:
    """Normalize features using specified method."""
    if method is None:
        return X_train, X_test

    if method == 'standard':
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std
    elif method == 'minmax':
        min_val = np.min(X_train, axis=0)
        max_val = np.max(X_train, axis=0)
        X_train_norm = (X_train - min_val) / (max_val - min_val)
        X_test_norm = (X_test - min_val) / (max_val - min_val)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_train_norm, X_test_norm

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     metric: Union[str, Callable]) -> Dict:
    """Compute specified metrics between true and predicted values."""
    if callable(metric):
        return {'metric': metric(y_true, y_pred)}

    if metric == 'mse':
        return {'metric': np.mean((y_true - y_pred) ** 2)}
    elif metric == 'mae':
        return {'metric': np.mean(np.abs(y_true - y_pred))}
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return {'metric': 1 - (ss_res / ss_tot)}
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _bootstrap_bias_variance(
    model: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_samples: int,
    rng: np.random.RandomState
) -> tuple:
    """Estimate bias and variance using bootstrap."""
    n_samples_train = X_train.shape[0]
    y_preds = np.zeros((n_samples, X_test.shape[0]))

    for i in range(n_samples):
        # Bootstrap sample
        indices = rng.choice(n_samples_train, size=n_samples_train, replace=True)
        X_boot = X_train[indices]
        y_boot = y_train[indices]

        # Fit model and predict
        model.fit(X_boot, y_boot)
        y_preds[i] = model.predict(X_test)

    # Calculate bias and variance
    avg_y = np.mean(y_train)
    y_avg_pred = np.mean(y_preds, axis=0)

    bias_sq = np.mean((y_avg_pred - avg_y) ** 2)
    variance = np.mean(np.var(y_preds, axis=0))

    return bias_sq, variance

################################################################################
# learning_curve
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def learning_curve_fit(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    train_sizes: Optional[np.ndarray] = None,
    normalize: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None,
    custom_metric_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compute learning curves for model diagnostics.

    Parameters:
    -----------
    model : Callable
        The model to evaluate. Must have fit and predict methods.
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    cv : int, default=5
        Number of cross-validation folds.
    train_sizes : np.ndarray, optional
        Array of relative or absolute numbers of training examples.
    normalize : str, default='standard'
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : str or Callable, default='mse'
        Metric to evaluate: 'mse', 'mae', 'r2', 'logloss', or custom callable.
    solver : str, default='closed_form'
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'.
    regularization : str, optional
        Regularization type: 'none', 'l1', 'l2', or 'elasticnet'.
    tol : float, default=1e-4
        Tolerance for stopping criteria.
    max_iter : int, default=1000
        Maximum number of iterations.
    random_state : int, optional
        Random seed for reproducibility.
    custom_metric_params : Dict[str, Any], optional
        Parameters for custom metric function.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 5)

    # Normalize data
    X_normalized = _normalize_data(X, method=normalize)

    # Initialize results storage
    results = {
        'train_scores': [],
        'test_scores': [],
        'train_sizes': []
    }

    # Compute learning curve
    for size in train_sizes:
        n_train = int(size * len(X))
        X_train, y_train = X_normalized[:n_train], y[:n_train]
        X_test, y_test = X_normalized[n_train:], y[n_train:]

        # Fit model
        fitted_model = _fit_model(
            model, X_train, y_train,
            solver=solver,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state
        )

        # Compute metrics
        train_score = _compute_metric(
            metric, y_train, fitted_model.predict(X_train),
            params=custom_metric_params
        )
        test_score = _compute_metric(
            metric, y_test, fitted_model.predict(X_test),
            params=custom_metric_params
        )

        results['train_scores'].append(train_score)
        results['test_scores'].append(test_score)
        results['train_sizes'].append(n_train)

    return {
        'result': results,
        'metrics': {'train': metric, 'test': metric},
        'params_used': {
            'normalize': normalize,
            'metric': metric,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': _check_warnings(results)
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

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
        iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _fit_model(
    model: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Any:
    """Fit model with specified solver and regularization."""
    if random_state is not None:
        np.random.seed(random_state)

    # Here you would implement the actual model fitting logic
    # based on the solver and regularization parameters

    # This is a placeholder - actual implementation would depend
    # on the specific model and solver requirements

    return model.fit(X_train, y_train)

def _compute_metric(
    metric: Union[str, Callable],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> float:
    """Compute specified metric between true and predicted values."""
    if callable(metric):
        return metric(y_true, y_pred, **(params or {}))

    if metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    elif metric == 'logloss':
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _check_warnings(results: Dict[str, Any]) -> list:
    """Check for potential issues in the results."""
    warnings = []

    if len(results['train_scores']) < 2:
        warnings.append("Insufficient data points for reliable learning curve")

    if np.any(np.isnan(results['train_scores'] + results['test_scores'])):
        warnings.append("NaN values detected in scores")

    return warnings

# Example usage:
"""
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.random.rand(100, 5)
y = np.random.rand(100)

result = learning_curve_fit(
    model=LinearRegression(),
    X=X,
    y=y,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 5),
    normalize='standard',
    metric='mse'
)
"""

################################################################################
# residual_analysis
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def residual_analysis_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalization: str = "standard",
    metrics: Union[str, list] = ["mse", "mae"],
    distance_metric: str = "euclidean",
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Perform residual analysis on model predictions.

    Parameters:
    -----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values from the model.
    normalization : str, optional (default="standard")
        Normalization method for residuals: "none", "standard", "minmax", or "robust".
    metrics : str or list, optional (default=["mse", "mae"])
        Metrics to compute: "mse", "mae", "r2", "logloss", or custom callable.
    distance_metric : str, optional (default="euclidean")
        Distance metric for residuals: "euclidean", "manhattan", "cosine", or "minkowski".
    custom_metric : Callable, optional
        Custom metric function to compute.

    Returns:
    --------
    Dict containing:
        - "result": Computed residuals.
        - "metrics": Dictionary of computed metrics.
        - "params_used": Parameters used in the computation.
        - "warnings": List of warnings encountered.

    Example:
    --------
    >>> y_true = np.array([1, 2, 3])
    >>> y_pred = np.array([1.1, 1.9, 3.2])
    >>> result = residual_analysis_fit(y_true, y_pred)
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Normalize residuals
    residuals = y_true - y_pred
    if normalization != "none":
        residuals = _normalize_residuals(residuals, method=normalization)

    # Compute metrics
    metrics_dict = _compute_metrics(y_true, y_pred, residuals, metrics, custom_metric)

    # Compute distance metric
    if distance_metric != "none":
        distance = _compute_distance(residuals, metric=distance_metric)
    else:
        distance = None

    # Prepare output
    result_dict = {
        "result": residuals,
        "metrics": metrics_dict,
        "params_used": {
            "normalization": normalization,
            "metrics": metrics,
            "distance_metric": distance_metric
        },
        "warnings": []
    }

    return result_dict

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or inf values.")
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError("y_pred contains NaN or inf values.")

def _normalize_residuals(residuals: np.ndarray, method: str = "standard") -> np.ndarray:
    """Normalize residuals using specified method."""
    if method == "standard":
        return (residuals - np.mean(residuals)) / np.std(residuals)
    elif method == "minmax":
        return (residuals - np.min(residuals)) / (np.max(residuals) - np.min(residuals))
    elif method == "robust":
        median = np.median(residuals)
        iqr = np.percentile(residuals, 75) - np.percentile(residuals, 25)
        return (residuals - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    metrics: Union[str, list],
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Compute specified metrics."""
    metrics_dict = {}

    if isinstance(metrics, str):
        metrics = [metrics]

    for metric in metrics:
        if metric == "mse":
            metrics_dict["mse"] = np.mean(residuals**2)
        elif metric == "mae":
            metrics_dict["mae"] = np.mean(np.abs(residuals))
        elif metric == "r2":
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            metrics_dict["r2"] = 1 - (ss_res / ss_tot)
        elif metric == "logloss":
            metrics_dict["logloss"] = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        elif callable(custom_metric):
            metrics_dict["custom"] = custom_metric(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return metrics_dict

def _compute_distance(residuals: np.ndarray, metric: str = "euclidean") -> float:
    """Compute distance metric for residuals."""
    if metric == "euclidean":
        return np.linalg.norm(residuals)
    elif metric == "manhattan":
        return np.sum(np.abs(residuals))
    elif metric == "cosine":
        return 1 - np.dot(residuals, residuals) / (np.linalg.norm(residuals) * np.linalg.norm(residuals))
    elif metric == "minkowski":
        return np.sum(np.abs(residuals)**3)**(1/3)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

################################################################################
# overfitting_detection
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def overfitting_detection_fit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_fit_func: Callable,
    model_predict_func: Callable,
    metric_func: Union[str, Callable] = 'mse',
    normalization: Optional[str] = None,
    n_splits: int = 5,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Detect overfitting by comparing training and validation performance.

    Parameters:
    -----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training targets.
    X_val : np.ndarray
        Validation features.
    y_val : np.ndarray
        Validation targets.
    model_fit_func : Callable
        Function to fit the model (should take X, y and return fitted model).
    model_predict_func : Callable
        Function to predict using the model (should take model and X).
    metric_func : Union[str, Callable], optional
        Metric to evaluate performance ('mse', 'mae', 'r2', or custom callable).
    normalization : Optional[str], optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    n_splits : int, optional
        Number of splits for cross-validation.
    max_iter : int, optional
        Maximum iterations for model fitting.
    tol : float, optional
        Tolerance for convergence.
    random_state : Optional[int], optional
        Random seed for reproducibility.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X_train, y_train, X_val, y_val)

    # Normalize data if specified
    X_train_norm, X_val_norm = _apply_normalization(X_train, X_val, normalization)

    # Initialize results dictionary
    results = {
        'result': None,
        'metrics': {},
        'params_used': {
            'normalization': normalization,
            'n_splits': n_splits,
            'max_iter': max_iter,
            'tol': tol,
            'random_state': random_state
        },
        'warnings': []
    }

    # Perform overfitting detection
    train_metrics, val_metrics = _detect_overfitting(
        X_train_norm, y_train, X_val_norm, y_val,
        model_fit_func, model_predict_func, metric_func,
        n_splits, max_iter, tol, random_state
    )

    # Store results
    results['metrics']['train'] = train_metrics
    results['metrics']['val'] = val_metrics

    # Determine if overfitting is detected
    results['result'] = _check_overfitting(train_metrics, val_metrics)

    return results

def _validate_inputs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> None:
    """Validate input dimensions and types."""
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train must have the same number of samples.")
    if X_val.shape[0] != y_val.shape[0]:
        raise ValueError("X_val and y_val must have the same number of samples.")
    if X_train.ndim != 2 or X_val.ndim != 2:
        raise ValueError("X_train and X_val must be 2D arrays.")
    if y_train.ndim != 1 or y_val.ndim != 1:
        raise ValueError("y_train and y_val must be 1D arrays.")

def _apply_normalization(
    X_train: np.ndarray,
    X_val: np.ndarray,
    normalization: Optional[str]
) -> tuple:
    """Apply specified normalization to the data."""
    if normalization is None:
        return X_train, X_val

    if normalization == 'standard':
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std
    elif normalization == 'minmax':
        min_val = np.min(X_train, axis=0)
        max_val = np.max(X_train, axis=0)
        X_train_norm = (X_train - min_val) / (max_val - min_val)
        X_val_norm = (X_val - min_val) / (max_val - min_val)
    elif normalization == 'robust':
        median = np.median(X_train, axis=0)
        iqr = np.subtract(*np.percentile(X_train, [75, 25], axis=0))
        X_train_norm = (X_train - median) / iqr
        X_val_norm = (X_val - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    return X_train_norm, X_val_norm

def _detect_overfitting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_fit_func: Callable,
    model_predict_func: Callable,
    metric_func: Union[str, Callable],
    n_splits: int,
    max_iter: int,
    tol: float,
    random_state: Optional[int]
) -> tuple:
    """Detect overfitting by comparing training and validation metrics."""
    # Fit the model on training data
    model = model_fit_func(X_train, y_train, max_iter=max_iter, tol=tol)

    # Predict on training and validation data
    y_train_pred = model_predict_func(model, X_train)
    y_val_pred = model_predict_func(model, X_val)

    # Calculate metrics
    train_metric = _calculate_metric(y_train, y_train_pred, metric_func)
    val_metric = _calculate_metric(y_val, y_val_pred, metric_func)

    return train_metric, val_metric

def _calculate_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Union[str, Callable]
) -> float:
    """Calculate the specified metric."""
    if isinstance(metric_func, str):
        if metric_func == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif metric_func == 'mae':
            return np.mean(np.abs(y_true - y_pred))
        elif metric_func == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot)
        elif metric_func == 'logloss':
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            raise ValueError(f"Unknown metric: {metric_func}")
    else:
        return metric_func(y_true, y_pred)

def _check_overfitting(
    train_metric: float,
    val_metric: float
) -> bool:
    """Check if overfitting is detected."""
    # Overfitting is detected if validation metric is significantly worse than training metric
    return val_metric > train_metric * 1.2

# Example usage:
"""
from sklearn.linear_model import LinearRegression
import numpy as np

X_train = np.random.rand(100, 5)
y_train = np.random.rand(100)
X_val = np.random.rand(20, 5)
y_val = np.random.rand(20)

def model_fit(X, y, **kwargs):
    model = LinearRegression()
    model.fit(X, y)
    return model

def model_predict(model, X):
    return model.predict(X)

result = overfitting_detection_fit(
    X_train, y_train, X_val, y_val,
    model_fit_func=model_fit,
    model_predict_func=model_predict,
    metric_func='mse',
    normalization='standard'
)
"""

################################################################################
# underfitting_detection
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def underfitting_detection_fit(
    X: np.ndarray,
    y: np.ndarray,
    model_func: Callable[[np.ndarray], np.ndarray],
    metric_func: Union[str, Callable] = 'mse',
    normalization: str = 'none',
    solver: str = 'closed_form',
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric_func: Optional[Callable] = None,
    custom_distance_func: Optional[Callable] = None
) -> Dict:
    """
    Detect underfitting in a model by comparing its performance with a baseline.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    model_func : Callable[[np.ndarray], np.ndarray]
        Function that takes X and returns predicted y
    metric_func : Union[str, Callable], optional
        Metric to evaluate model performance. Can be 'mse', 'mae', 'r2', or custom callable
    normalization : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', or 'newton'
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
    custom_metric_func : Optional[Callable], optional
        Custom metric function if not using built-in metrics
    custom_distance_func : Optional[Callable], optional
        Custom distance function for model evaluation

    Returns:
    --------
    Dict containing:
        - result: dict with underfitting detection results
        - metrics: dict of computed metrics
        - params_used: dict of parameters used
        - warnings: list of warning messages

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> def simple_model(x): return x @ np.ones((x.shape[1], 1))
    >>> result = underfitting_detection_fit(X, y, simple_model)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if needed
    X_norm, y_norm = _apply_normalization(X, y, normalization)

    # Fit model
    params_used = {
        'normalization': normalization,
        'solver': solver,
        'metric': metric_func,
        'tol': tol,
        'max_iter': max_iter
    }

    # Get predictions
    y_pred = model_func(X_norm)

    # Calculate metrics
    metrics = _calculate_metrics(y_norm, y_pred, metric_func, custom_metric_func)

    # Detect underfitting
    result = _detect_underfitting(y_norm, y_pred, metrics)

    # Check for warnings
    warnings = _check_warnings(X_norm, y_norm, y_pred)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays contain infinite values")

def _apply_normalization(X: np.ndarray, y: np.ndarray, method: str) -> tuple:
    """Apply normalization to input data."""
    X_norm = X.copy()
    y_norm = y.copy()

    if method == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_norm = (X - X_mean) / (X_std + 1e-8)
        y_norm = (y - np.mean(y)) / (np.std(y) + 1e-8)
    elif method == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-8)
    elif method == 'robust':
        X_median = np.median(X, axis=0)
        X_q75, X_q25 = np.percentile(X, [75, 25], axis=0)
        X_iqr = X_q75 - X_q25
        X_norm = (X - X_median) / (X_iqr + 1e-8)
        y_median = np.median(y)
        y_q75, y_q25 = np.percentile(y, [75, 25])
        y_iqr = y_q75 - y_q25
        y_norm = (y - y_median) / (y_iqr + 1e-8)

    return X_norm, y_norm

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Union[str, Callable],
    custom_metric_func: Optional[Callable] = None
) -> Dict:
    """Calculate evaluation metrics."""
    metrics = {}

    if metric_func == 'mse' or (custom_metric_func is None and metric_func == 'mse'):
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    if metric_func == 'mae' or (custom_metric_func is None and metric_func == 'mae'):
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    if metric_func == 'r2' or (custom_metric_func is None and metric_func == 'r2'):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / (ss_tot + 1e-8))

    if custom_metric_func is not None:
        metrics['custom'] = custom_metric_func(y_true, y_pred)

    return metrics

def _detect_underfitting(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict
) -> Dict:
    """Detect underfitting based on model performance."""
    result = {
        'underfit': False,
        'severity': None,
        'recommendation': None
    }

    # Simple underfitting detection based on R2 score if available
    if 'r2' in metrics:
        r2 = metrics['r2']
        if r2 < 0.1:
            result['underfit'] = True
            result['severity'] = 'high'
            result['recommendation'] = 'Consider more complex model or feature engineering'
        elif r2 < 0.5:
            result['underfit'] = True
            result['severity'] = 'medium'
            result['recommendation'] = 'Model may be underfitting, consider adding more features'

    return result

def _check_warnings(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> List:
    """Check for potential issues and generate warnings."""
    warnings = []

    if np.any(np.isnan(y_pred)):
        warnings.append("Predictions contain NaN values")
    if np.any(np.isinf(y_pred)):
        warnings.append("Predictions contain infinite values")

    # Check for constant predictions
    if np.all(y_pred == y_pred[0]):
        warnings.append("Model is making constant predictions - potential underfitting")

    return warnings

################################################################################
# feature_importance
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def feature_importance_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute feature importance for a given dataset.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalizer : Callable[[np.ndarray], np.ndarray], optional
        Function to normalize features. Default is None.
    metric : str or Callable[[np.ndarray, np.ndarray], float]
        Metric to evaluate model performance. Default is "mse".
    distance : str or Callable[[np.ndarray, np.ndarray], float]
        Distance metric for feature importance. Default is "euclidean".
    solver : str
        Solver to use for optimization. Default is "closed_form".
    regularization : str, optional
        Type of regularization to apply. Default is None.
    tol : float
        Tolerance for convergence. Default is 1e-4.
    max_iter : int
        Maximum number of iterations. Default is 1000.
    custom_metric : Callable[[np.ndarray, np.ndarray], float], optional
        Custom metric function. Default is None.
    **kwargs : dict
        Additional solver-specific parameters.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = feature_importance_fit(X, y, normalizer="standard", metric="mse")
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Choose metric
    if isinstance(metric, str):
        metric_func = _get_metric_function(metric)
    else:
        metric_func = metric

    # Choose distance
    if isinstance(distance, str):
        distance_func = _get_distance_function(distance)
    else:
        distance_func = distance

    # Choose solver
    if solver == "closed_form":
        coefficients, metrics = _solve_closed_form(X_normalized, y, metric_func)
    elif solver == "gradient_descent":
        coefficients, metrics = _solve_gradient_descent(
            X_normalized, y, metric_func, tol=tol, max_iter=max_iter,
            regularization=regularization, **kwargs
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate feature importance
    feature_importance = _calculate_feature_importance(coefficients, distance_func)

    # Prepare results
    result = {
        "result": feature_importance,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to features."""
    if normalizer is None:
        return X
    return normalizer(X)

def _get_metric_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get metric function based on name."""
    metrics = {
        "mse": _mean_squared_error,
        "mae": _mean_absolute_error,
        "r2": _r_squared,
        "logloss": _log_loss
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _get_distance_function(distance: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get distance function based on name."""
    distances = {
        "euclidean": _euclidean_distance,
        "manhattan": _manhattan_distance,
        "cosine": _cosine_distance,
        "minkowski": _minkowski_distance
    }
    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")
    return distances[distance]

def _solve_closed_form(X: np.ndarray, y: np.ndarray, metric_func: Callable) -> tuple:
    """Solve using closed form solution."""
    # This is a placeholder - actual implementation would depend on the metric
    coefficients = np.linalg.pinv(X) @ y
    predictions = X @ coefficients
    metrics = {"metric_value": metric_func(y, predictions)}
    return coefficients, metrics

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000,
    regularization: Optional[str] = None,
    **kwargs
) -> tuple:
    """Solve using gradient descent."""
    # This is a placeholder - actual implementation would depend on the metric and regularization
    coefficients = np.zeros(X.shape[1])
    metrics_history = []

    for _ in range(max_iter):
        predictions = X @ coefficients
        gradient = _compute_gradient(X, y, predictions)
        if regularization == "l1":
            gradient += np.sign(coefficients) * kwargs.get("alpha", 0.1)
        elif regularization == "l2":
            gradient += 2 * kwargs.get("alpha", 0.1) * coefficients

        coefficients -= gradient * kwargs.get("learning_rate", 0.01)

        metrics_history.append(metric_func(y, predictions))

        if len(metrics_history) > 1 and abs(metrics_history[-2] - metrics_history[-1]) < tol:
            break

    return coefficients, {"metrics_history": metrics_history}

def _calculate_feature_importance(
    coefficients: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """Calculate feature importance based on coefficients and distance."""
    # This is a placeholder - actual implementation would depend on the distance metric
    return np.abs(coefficients)

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
    return 1 - (ss_res / ss_tot)

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute log loss."""
    # This is a placeholder - actual implementation would need proper handling of probabilities
    return np.mean(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance."""
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def _minkowski_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Minkowski distance."""
    return np.sum(np.abs(a - b) ** 3) ** (1/3)

def _compute_gradient(X: np.ndarray, y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Compute gradient for MSE."""
    return -2 * X.T @ (y - predictions) / len(y)

################################################################################
# shap_values
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def shap_values_fit(
    model: Any,
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalizer: str = "standard",
    metric: Union[str, Callable] = "mse",
    distance: str = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Compute SHAP values for a given model and dataset.

    Parameters:
    -----------
    model : Any
        The trained model for which SHAP values are computed.
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : Optional[np.ndarray]
        Target values array of shape (n_samples,). Required for supervised models.
    normalizer : str
        Normalization method: "none", "standard", "minmax", or "robust".
    metric : Union[str, Callable]
        Metric to use: "mse", "mae", "r2", "logloss", or custom callable.
    distance : str
        Distance metric: "euclidean", "manhattan", "cosine", or custom callable.
    solver : str
        Solver method: "closed_form", "gradient_descent", "newton", or "coordinate_descent".
    regularization : Optional[str]
        Regularization type: "none", "l1", "l2", or "elasticnet".
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable]
        Custom metric function if not using built-in metrics.
    custom_distance : Optional[Callable]
        Custom distance function if not using built-in distances.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing SHAP values, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _normalize_data(X, normalizer)

    # Prepare metrics and distances
    metric_func = _get_metric(metric, custom_metric)
    distance_func = _get_distance(distance, custom_distance)

    # Compute SHAP values based on solver
    if solver == "closed_form":
        shap_values = _shap_closed_form(model, X_normalized, y, distance_func)
    elif solver == "gradient_descent":
        shap_values = _shap_gradient_descent(model, X_normalized, y, metric_func, distance_func,
                                            regularization, tol, max_iter)
    elif solver == "newton":
        shap_values = _shap_newton(model, X_normalized, y, metric_func, distance_func,
                                   regularization, tol, max_iter)
    elif solver == "coordinate_descent":
        shap_values = _shap_coordinate_descent(model, X_normalized, y, metric_func, distance_func,
                                               regularization, tol, max_iter)
    else:
        raise ValueError("Unsupported solver method.")

    # Compute metrics
    metrics = _compute_metrics(model, X_normalized, y, metric_func)

    # Prepare output
    result = {
        "shap_values": shap_values,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer,
            "metric": metric if isinstance(metric, str) else "custom",
            "distance": distance,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    return result

def _validate_inputs(X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None.")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values.")
    if y is not None and np.any(np.isnan(y)):
        raise ValueError("y contains NaN values.")

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize data based on specified method."""
    if method == "none":
        return X
    elif method == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError("Unsupported normalization method.")

def _get_metric(metric: Union[str, Callable], custom_metric: Optional[Callable] = None) -> Callable:
    """Get metric function based on input."""
    if callable(metric):
        return metric
    elif custom_metric is not None:
        return custom_metric
    elif metric == "mse":
        return _mean_squared_error
    elif metric == "mae":
        return _mean_absolute_error
    elif metric == "r2":
        return _r_squared
    elif metric == "logloss":
        return _log_loss
    else:
        raise ValueError("Unsupported metric.")

def _get_distance(distance: str, custom_distance: Optional[Callable] = None) -> Callable:
    """Get distance function based on input."""
    if callable(distance):
        return distance
    elif custom_distance is not None:
        return custom_distance
    elif distance == "euclidean":
        return _euclidean_distance
    elif distance == "manhattan":
        return _manhattan_distance
    elif distance == "cosine":
        return _cosine_distance
    else:
        raise ValueError("Unsupported distance metric.")

def _shap_closed_form(model: Any, X: np.ndarray, y: Optional[np.ndarray], distance_func: Callable) -> np.ndarray:
    """Compute SHAP values using closed form solution."""
    # Placeholder for actual implementation
    return np.random.rand(X.shape[1])

def _shap_gradient_descent(
    model: Any,
    X: np.ndarray,
    y: Optional[np.ndarray],
    metric_func: Callable,
    distance_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Compute SHAP values using gradient descent."""
    # Placeholder for actual implementation
    return np.random.rand(X.shape[1])

def _shap_newton(
    model: Any,
    X: np.ndarray,
    y: Optional[np.ndarray],
    metric_func: Callable,
    distance_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Compute SHAP values using Newton's method."""
    # Placeholder for actual implementation
    return np.random.rand(X.shape[1])

def _shap_coordinate_descent(
    model: Any,
    X: np.ndarray,
    y: Optional[np.ndarray],
    metric_func: Callable,
    distance_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Compute SHAP values using coordinate descent."""
    # Placeholder for actual implementation
    return np.random.rand(X.shape[1])

def _compute_metrics(model: Any, X: np.ndarray, y: Optional[np.ndarray], metric_func: Callable) -> Dict[str, float]:
    """Compute various metrics for the model."""
    predictions = _predict(model, X)
    return {
        "metric_value": metric_func(y, predictions),
        "sample_size": X.shape[0],
        "feature_count": X.shape[1]
    }

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

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute log loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance."""
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def _predict(model: Any, X: np.ndarray) -> np.ndarray:
    """Make predictions using the model."""
    # Placeholder for actual implementation
    return np.random.rand(X.shape[0])

################################################################################
# partial_dependence_plot
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def partial_dependence_plot_fit(
    model: Callable,
    X: np.ndarray,
    feature_indices: Union[int, list],
    grid_points: int = 100,
    normalize: str = 'none',
    metric: Callable = None,
    distance_metric: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: str = 'none',
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_weights: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute partial dependence plots for a given model.

    Parameters:
    -----------
    model : Callable
        The trained model to analyze. Must have a predict method.
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    feature_indices : int or list
        Indices of features to analyze.
    grid_points : int, optional
        Number of points in the grid for partial dependence (default: 100).
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'none').
    metric : Callable, optional
        Custom metric function. If None, uses model's default prediction.
    distance_metric : str, optional
        Distance metric for feature space ('euclidean', 'manhattan', 'cosine', 'minkowski') (default: 'euclidean').
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent') (default: 'closed_form').
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet') (default: 'none').
    tol : float, optional
        Tolerance for convergence (default: 1e-4).
    max_iter : int, optional
        Maximum number of iterations (default: 1000).
    custom_weights : np.ndarray, optional
        Custom weights for samples (default: None).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, feature_indices, grid_points)

    # Normalize features if required
    X_normalized = _normalize_features(X, normalize)

    # Compute partial dependence
    pdp_result = _compute_partial_dependence(
        model, X_normalized, feature_indices, grid_points,
        metric=metric, distance_metric=distance_metric,
        solver=solver, regularization=regularization,
        tol=tol, max_iter=max_iter, custom_weights=custom_weights
    )

    # Calculate metrics
    metrics = _calculate_metrics(pdp_result, X_normalized)

    return {
        'result': pdp_result,
        'metrics': metrics,
        'params_used': {
            'normalize': normalize,
            'metric': metric.__name__ if metric else None,
            'distance_metric': distance_metric,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, feature_indices: Union[int, list], grid_points: int) -> None:
    """Validate input parameters."""
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values.")

    if isinstance(feature_indices, int):
        feature_indices = [feature_indices]
    elif not isinstance(feature_indices, list):
        raise ValueError("feature_indices must be an int or a list of indices.")

    if grid_points <= 0:
        raise ValueError("grid_points must be a positive integer.")

def _normalize_features(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize features based on the specified method."""
    if method == 'none':
        return X
    elif method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_partial_dependence(
    model: Callable,
    X: np.ndarray,
    feature_indices: list,
    grid_points: int,
    metric: Optional[Callable] = None,
    distance_metric: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: str = 'none',
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_weights: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Compute partial dependence for specified features."""
    pdp_values = []
    feature_ranges = []

    for idx in feature_indices:
        # Get the range of the feature
        feature_range = np.linspace(np.min(X[:, idx]), np.max(X[:, idx]), grid_points)
        feature_ranges.append(feature_range)

        # Compute partial dependence for this feature
        pdp_values.append(_compute_single_feature_dependence(
            model, X, idx, feature_range,
            metric=metric, distance_metric=distance_metric,
            solver=solver, regularization=regularization,
            tol=tol, max_iter=max_iter, custom_weights=custom_weights
        ))

    return {
        'feature_indices': feature_indices,
        'pdp_values': pdp_values,
        'feature_ranges': feature_ranges
    }

def _compute_single_feature_dependence(
    model: Callable,
    X: np.ndarray,
    feature_idx: int,
    grid_points: np.ndarray,
    metric: Optional[Callable] = None,
    distance_metric: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: str = 'none',
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute partial dependence for a single feature."""
    pdp_values = []

    for value in grid_points:
        # Create modified dataset with this feature value
        X_modified = X.copy()
        X_modified[:, feature_idx] = value

        # Predict using the model
        if metric is None:
            predictions = model.predict(X_modified)
        else:
            predictions = _apply_metric(model, X_modified, metric)

        # Average predictions
        if custom_weights is None:
            pdp_values.append(np.mean(predictions))
        else:
            pdp_values.append(np.average(predictions, weights=custom_weights))

    return np.array(pdp_values)

def _apply_metric(model: Callable, X: np.ndarray, metric: Callable) -> np.ndarray:
    """Apply a custom metric to model predictions."""
    predictions = model.predict(X)
    return metric(predictions, X)

def _calculate_metrics(
    pdp_result: Dict[str, Any],
    X: np.ndarray
) -> Dict[str, float]:
    """Calculate metrics for partial dependence results."""
    # Example metric: variance of PDP values
    variances = []
    for pdp_values in pdp_result['pdp_values']:
        variances.append(np.var(pdp_values))

    return {
        'variance': np.mean(variances),
        'feature_range': [np.min(X), np.max(X)]
    }

# Example usage:
"""
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Generate sample data
X = np.random.rand(100, 5)
y = np.sum(X, axis=1) + np.random.normal(0, 0.1, 100)

# Train a model
model = RandomForestRegressor()
model.fit(X, y)

# Compute partial dependence plot
result = partial_dependence_plot_fit(
    model=model,
    X=X,
    feature_indices=[0, 1],
    grid_points=50,
    normalize='standard',
    metric=None
)

print(result)
"""

################################################################################
# permutation_importance
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def permutation_importance_fit(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    n_repeats: int = 10,
    random_state: Optional[int] = None,
    n_jobs: int = 1
) -> Dict[str, Any]:
    """
    Compute permutation importance for a given model.

    Parameters:
    -----------
    model : Any
        A fitted model with predict method.
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    metric : str or callable, optional
        Metric to evaluate model performance. Default is 'mse'.
    n_repeats : int, optional
        Number of permutations to perform for each feature. Default is 10.
    random_state : int, optional
        Random seed for reproducibility. Default is None.
    n_jobs : int, optional
        Number of parallel jobs to run. Default is 1.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - 'importance': array of importance scores
        - 'metrics': dictionary of metrics used
        - 'params_used': parameters used in the computation
        - 'warnings': list of warnings encountered

    Example:
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> lr = LinearRegression().fit(X, y)
    >>> result = permutation_importance_fit(lr, X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set random state
    rng = np.random.RandomState(random_state)

    # Get baseline score
    baseline_score = _compute_baseline_score(model, X, y, metric)

    # Compute permutation importance
    importance = _compute_permutation_importance(
        model, X, y, metric, n_repeats, rng, n_jobs
    )

    # Prepare output
    result = {
        'importance': importance,
        'metrics': {'baseline_score': baseline_score},
        'params_used': {
            'n_repeats': n_repeats,
            'random_state': random_state,
            'n_jobs': n_jobs
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _compute_baseline_score(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> float:
    """Compute baseline score for the model."""
    y_pred = model.predict(X)
    return _evaluate_metric(y, y_pred, metric)

def _compute_permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    n_repeats: int,
    rng: np.random.RandomState,
    n_jobs: int
) -> np.ndarray:
    """Compute permutation importance for each feature."""
    n_features = X.shape[1]
    importance = np.zeros(n_features)

    for i in range(n_features):
        # Permute feature column
        X_permuted = X.copy()
        for _ in range(n_repeats):
            rng.shuffle(X_permuted[:, i])
            y_pred = model.predict(X_permuted)
            score = _evaluate_metric(y, y_pred, metric)
            importance[i] += (baseline_score - score) / n_repeats

    return importance

def _evaluate_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> float:
    """Evaluate metric for given true and predicted values."""
    if callable(metric):
        return metric(y_true, y_pred)
    elif metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    else:
        raise ValueError(f"Unknown metric: {metric}")

################################################################################
# calibration_curve
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def calibration_curve_fit(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    strategy: str = 'uniform',
    normalize: bool = True,
    metric_func: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Compute the calibration curve for a probabilistic classifier.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_proba : np.ndarray
        Estimated probabilities of the positive class.
    n_bins : int, optional
        Number of bins to discretize [0,1] interval.
    strategy : str, optional
        Strategy used to define the widths of the bins:
            - 'uniform': all bins have identical widths
            - 'quantile': all bins have the same number of samples
    normalize : bool, optional
        Whether to normalize the bin counts.
    metric_func : Callable, optional
        Custom metric function. If None, uses Brier score by default.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict]]
        Dictionary containing:
            - 'result': Tuple of (prob_true, prob_pred) arrays
            - 'metrics': Dictionary of computed metrics
            - 'params_used': Parameters used for computation
            - 'warnings': Any warnings encountered

    Examples
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_proba = np.array([0.1, 0.4, 0.35, 0.8])
    >>> result = calibration_curve_fit(y_true, y_proba)
    """
    # Validate inputs
    _validate_inputs(y_true, y_proba)

    # Compute calibration curve
    prob_true, prob_pred = _compute_calibration_curve(
        y_true, y_proba, n_bins, strategy
    )

    # Compute metrics
    metrics = _compute_metrics(y_true, y_proba, prob_pred, metric_func)

    # Prepare output
    result = {
        'result': (prob_true, prob_pred),
        'metrics': metrics,
        'params_used': {
            'n_bins': n_bins,
            'strategy': strategy,
            'normalize': normalize
        },
        'warnings': _check_warnings(y_true, y_proba)
    }

    return result

def _validate_inputs(
    y_true: np.ndarray,
    y_proba: np.ndarray
) -> None:
    """Validate input arrays."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_proba, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")

    if y_true.shape != y_proba.shape:
        raise ValueError("y_true and y_proba must have the same shape")

    if np.any((y_true != 0) & (y_true != 1)):
        raise ValueError("y_true must contain only 0s and 1s")

    if np.any((y_proba < 0) | (y_proba > 1)):
        raise ValueError("y_proba must be in [0, 1] range")

    if np.isnan(y_true).any() or np.isinf(y_true).any():
        raise ValueError("y_true contains NaN or Inf values")

    if np.isnan(y_proba).any() or np.isinf(y_proba).any():
        raise ValueError("y_proba contains NaN or Inf values")

def _compute_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int,
    strategy: str
) -> tuple:
    """Compute the calibration curve."""
    if strategy == 'uniform':
        bin_edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == 'quantile':
        bin_edges = np.percentile(y_proba, np.linspace(0, 100, n_bins + 1))
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")

    bin_indices = np.digitize(y_proba, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    prob_true = np.zeros(n_bins)
    prob_pred = np.zeros(n_bins)
    n_samples_per_bin = np.zeros(n_bins)

    for i in range(n_bins):
        mask = bin_indices == i
        if np.any(mask):
            prob_true[i] = np.mean(y_true[mask])
            prob_pred[i] = np.mean(y_proba[mask])
            n_samples_per_bin[i] = np.sum(mask)

    return prob_true, prob_pred

def _compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    prob_pred: np.ndarray,
    metric_func: Optional[Callable]
) -> Dict[str, float]:
    """Compute metrics for the calibration curve."""
    metrics = {}

    if metric_func is None:
        # Default to Brier score
        brier_score = np.mean((y_true - y_proba) ** 2)
        metrics['brier_score'] = brier_score
    else:
        try:
            custom_metric = metric_func(y_true, y_proba)
            metrics['custom_metric'] = custom_metric
        except Exception as e:
            raise ValueError(f"Custom metric function failed: {str(e)}")

    # Add calibration metrics
    cal_error = np.mean(np.abs(prob_pred - prob_pred))
    metrics['calibration_error'] = cal_error

    return metrics

def _check_warnings(
    y_true: np.ndarray,
    y_proba: np.ndarray
) -> Dict[str, str]:
    """Check for potential warnings."""
    warnings = {}

    if len(y_true) < 20:
        warnings['small_sample'] = "Small sample size may affect calibration curve reliability"

    if np.all(y_true == 0) or np.all(y_true == 1):
        warnings['monotonic_labels'] = "All labels are identical, calibration curve may be uninformative"

    return warnings

################################################################################
# cross_validation_scores
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def cross_validation_scores_fit(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    cv_splits: int = 5,
    normalize: str = "standard",
    metric: Union[str, Callable] = "mse",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None,
    custom_metric_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compute cross-validation scores for a given model.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    model : Callable
        Model to evaluate. Must have fit and predict methods.
    cv_splits : int, optional
        Number of cross-validation splits (default: 5)
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard')
    metric : Union[str, Callable], optional
        Metric to evaluate ('mse', 'mae', 'r2', 'logloss') or custom callable (default: 'mse')
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent') (default: 'closed_form')
    regularization : Optional[str], optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet') (default: None)
    tol : float, optional
        Tolerance for convergence (default: 1e-4)
    max_iter : int, optional
        Maximum number of iterations (default: 1000)
    random_state : Optional[int], optional
        Random seed for reproducibility (default: None)
    custom_metric_kwargs : Optional[Dict[str, Any]], optional
        Additional arguments for custom metric (default: None)

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if needed
    X_normalized = _apply_normalization(X, normalize)

    # Initialize results dictionary
    results = {
        "result": {},
        "metrics": {},
        "params_used": {
            "cv_splits": cv_splits,
            "normalize": normalize,
            "metric": metric,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    # Perform cross-validation
    fold_scores = _cross_validate(
        X_normalized, y, model, cv_splits,
        metric, solver, regularization,
        tol, max_iter, random_state
    )

    # Store results
    results["result"]["fold_scores"] = fold_scores
    results["metrics"]["mean_score"] = np.mean(fold_scores)
    results["metrics"]["std_score"] = np.std(fold_scores)

    return results

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to feature matrix."""
    if method == "none":
        return X
    elif method == "standard":
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == "minmax":
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == "robust":
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    cv_splits: int,
    metric: Union[str, Callable],
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    random_state: Optional[int]
) -> np.ndarray:
    """Perform cross-validation and return scores for each fold."""
    n_samples = X.shape[0]
    fold_size = n_samples // cv_splits
    scores = np.zeros(cv_splits)

    for i in range(cv_splits):
        # Split data
        test_indices = range(i * fold_size, (i + 1) * fold_size)
        train_indices = [idx for idx in range(n_samples) if idx not in test_indices]

        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        # Fit model
        fitted_model = _fit_model(
            X_train, y_train,
            model, solver, regularization,
            tol, max_iter, random_state
        )

        # Predict and compute metric
        y_pred = fitted_model.predict(X_test)
        scores[i] = _compute_metric(y_test, y_pred, metric)

    return scores

def _fit_model(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    random_state: Optional[int]
) -> Callable:
    """Fit model with specified solver and regularization."""
    # Set model parameters
    model_params = {
        "solver": solver,
        "regularization": regularization,
        "tol": tol,
        "max_iter": max_iter
    }

    if random_state is not None:
        model_params["random_state"] = random_state

    # Fit model
    fitted_model = model(**model_params)
    fitted_model.fit(X, y)

    return fitted_model

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> float:
    """Compute specified metric between true and predicted values."""
    if callable(metric):
        return metric(y_true, y_pred)
    elif metric == "mse":
        return np.mean((y_true - y_pred) ** 2)
    elif metric == "mae":
        return np.mean(np.abs(y_true - y_pred))
    elif metric == "r2":
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    elif metric == "logloss":
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

################################################################################
# model_complexity_analysis
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def model_complexity_analysis_fit(
    X: np.ndarray,
    y: np.ndarray,
    model_class: Callable,
    param_grid: Dict[str, List],
    scoring_metric: Union[str, Callable] = 'mse',
    normalizer: Optional[Callable] = None,
    cv_folds: int = 5,
    random_state: Optional[int] = None
) -> Dict:
    """
    Perform model complexity analysis by fitting multiple models with different hyperparameters.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    model_class : Callable
        Model class to fit (e.g., sklearn.linear_model.LinearRegression)
    param_grid : Dict[str, List]
        Dictionary of hyperparameters to test
    scoring_metric : Union[str, Callable], optional
        Scoring metric ('mse', 'mae', 'r2', etc.) or custom callable
    normalizer : Optional[Callable], optional
        Normalization function (e.g., sklearn.preprocessing.StandardScaler)
    cv_folds : int, optional
        Number of cross-validation folds (default: 5)
    random_state : Optional[int], optional
        Random seed for reproducibility

    Returns
    -------
    Dict
        Dictionary containing:
        - 'results': List of tuples (params, score)
        - 'best_params': Best hyperparameters found
        - 'best_score': Best score achieved
        - 'warnings': List of warnings encountered

    Example
    -------
    >>> from sklearn.linear_model import LinearRegression
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> param_grid = {'fit_intercept': [True, False]}
    >>> results = model_complexity_analysis_fit(X, y, LinearRegression, param_grid)
    """
    # Input validation
    _validate_inputs(X, y)

    # Initialize results and warnings
    results = []
    warnings_list = []

    # Normalize data if specified
    X_normalized, y_normalized = _apply_normalization(X, y, normalizer)

    # Generate all parameter combinations
    param_combinations = _generate_param_combinations(param_grid)

    # Cross-validation and scoring
    for params in param_combinations:
        try:
            scores = _cross_validate_model(
                X_normalized, y_normalized,
                model_class, params,
                scoring_metric, cv_folds,
                random_state
            )
            avg_score = np.mean(scores)
            results.append((params, avg_score))
        except Exception as e:
            warnings_list.append(f"Error with params {params}: {str(e)}")

    if not results:
        raise ValueError("No successful model fits. Check warnings for details.")

    # Find best parameters
    best_params, best_score = max(results, key=lambda x: x[1])

    return {
        'results': results,
        'best_params': best_params,
        'best_score': best_score,
        'warnings': warnings_list
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalization(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable]
) -> tuple:
    """Apply normalization to features and target."""
    if normalizer is None:
        return X, y

    try:
        X_normalized = normalizer.fit_transform(X)
        if hasattr(normalizer, 'transform_y'):
            y_normalized = normalizer.transform_y(y)
        else:
            y_normalized = y
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

    return X_normalized, y_normalized

def _generate_param_combinations(param_grid: Dict[str, List]) -> List[Dict]:
    """Generate all combinations of hyperparameters."""
    from itertools import product

    keys = param_grid.keys()
    values = product(*param_grid.values())

    return [dict(zip(keys, v)) for v in values]

def _cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
    model_class: Callable,
    params: Dict,
    scoring_metric: Union[str, Callable],
    cv_folds: int,
    random_state: Optional[int]
) -> List[float]:
    """Perform cross-validation and return scores."""
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = model_class(**params)
        model.fit(X_train, y_train)

        if isinstance(scoring_metric, str):
            score = _compute_metric(model, X_test, y_test, scoring_metric)
        else:
            score = scoring_metric(model.predict(X_test), y_test)

        scores.append(score)

    return scores

def _compute_metric(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    metric_name: str
) -> float:
    """Compute specified metric."""
    y_pred = model.predict(X)

    if metric_name == 'mse':
        return np.mean((y_pred - y) ** 2)
    elif metric_name == 'mae':
        return np.mean(np.abs(y_pred - y))
    elif metric_name == 'r2':
        ss_res = np.sum((y_pred - y) ** 2)
        ss_tot = np.sum((np.mean(y) - y) ** 2)
        return 1 - (ss_res / ss_tot)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")
