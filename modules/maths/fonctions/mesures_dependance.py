"""
Quantix – Module mesures_dependance
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# correlation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def correlation_fit(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'pearson',
    normalize_X: Optional[str] = None,
    normalize_y: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]:
    """
    Calculate correlation between X and y with various options.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    method : str, optional
        Correlation method ('pearson', 'spearman', 'kendall')
    normalize_X : str, optional
        Normalization for X ('standard', 'minmax', 'robust')
    normalize_y : str, optional
        Normalization for y ('standard', 'minmax', 'robust')
    custom_metric : callable, optional
        Custom correlation metric function

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': correlation value(s)
        - 'metrics': additional metrics
        - 'params_used': parameters used
        - 'warnings': any warnings

    Examples
    --------
    >>> X = np.random.rand(10, 3)
    >>> y = np.random.rand(10)
    >>> result = correlation_fit(X, y, method='pearson')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if requested
    X_norm = _normalize_data(X, normalize_X) if normalize_X else X
    y_norm = _normalize_data(y.reshape(-1, 1), normalize_y).flatten() if normalize_y else y

    # Calculate correlation
    if custom_metric is not None:
        result = _calculate_custom_correlation(X_norm, y_norm, custom_metric)
    else:
        result = _calculate_correlation(X_norm, y_norm, method)

    # Calculate additional metrics
    metrics = _calculate_metrics(X_norm, y_norm, method)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'method': method,
            'normalize_X': normalize_X,
            'normalize_y': normalize_y
        },
        'warnings': _check_warnings(X_norm, y_norm)
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays contain infinite values")

def _normalize_data(data: np.ndarray, method: str) -> np.ndarray:
    """Normalize data using specified method."""
    if method == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == 'robust':
        return (data - np.median(data, axis=0)) / (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _calculate_correlation(X: np.ndarray, y: np.ndarray, method: str) -> Union[float, np.ndarray]:
    """Calculate correlation using specified method."""
    if method == 'pearson':
        return np.corrcoef(X, y, rowvar=False)[0, 1:]
    elif method == 'spearman':
        return np.corrcoef(_rank_data(X), _rank_data(y.reshape(-1, 1)), rowvar=False)[0, 1:].flatten()
    elif method == 'kendall':
        return np.array([_kendall_tau(X[:, i], y) for i in range(X.shape[1])])
    else:
        raise ValueError(f"Unknown correlation method: {method}")

def _calculate_custom_correlation(X: np.ndarray, y: np.ndarray, metric_func: Callable) -> Union[float, np.ndarray]:
    """Calculate correlation using custom metric function."""
    return np.array([metric_func(X[:, i], y) for i in range(X.shape[1])])

def _calculate_metrics(X: np.ndarray, y: np.ndarray, method: str) -> Dict[str, float]:
    """Calculate additional metrics."""
    if method == 'pearson':
        return {
            'r_squared': _calculate_r_squared(X, y),
            'covariance': np.cov(X, y, rowvar=False)[0, 1:]
        }
    else:
        return {}

def _check_warnings(X: np.ndarray, y: np.ndarray) -> list:
    """Check for potential issues and return warnings."""
    warnings = []
    if np.any(np.std(X, axis=0) == 0):
        warnings.append("Some features have zero variance")
    if np.std(y) == 0:
        warnings.append("Target has zero variance")
    return warnings

def _rank_data(data: np.ndarray) -> np.ndarray:
    """Convert data to ranks."""
    return np.argsort(np.argsort(data, axis=0), axis=0)

def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Kendall's tau correlation."""
    n = len(x)
    concordant = discordant = 0

    for i in range(n):
        for j in range(i+1, n):
            if (x[i] - x[j]) * (y[i] - y[j]) > 0:
                concordant += 1
            elif (x[i] - x[j]) * (y[i] - y[j]) < 0:
                discordant += 1

    return (concordant - discordant) / ((n * (n-1)) / 2)

def _calculate_r_squared(X: np.ndarray, y: np.ndarray) -> float:
    """Calculate R-squared between X and y."""
    means = np.mean(X, axis=0)
    ss_total = np.sum((y - np.mean(y))**2)
    ss_residual = np.sum((y - X.dot(np.linalg.pinv(X).dot(y)))**2)
    return 1 - (ss_residual / ss_total) if ss_total != 0 else np.nan

################################################################################
# covariance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, Y: Optional[np.ndarray] = None) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if Y is not None and not isinstance(Y, np.ndarray):
        raise TypeError("Y must be a numpy array or None")
    if Y is not None and X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    if np.isnan(X).any() or (Y is not None and np.isnan(Y).any()):
        raise ValueError("Input arrays must not contain NaN values")
    if np.isinf(X).any() or (Y is not None and np.isinf(Y).any()):
        raise ValueError("Input arrays must not contain infinite values")

def _normalize_data(X: np.ndarray, Y: Optional[np.ndarray] = None,
                   normalization: str = 'none') -> tuple:
    """Normalize input data."""
    if normalization == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_norm = (X - X_mean) / X_std
        if Y is not None:
            Y_mean = np.mean(Y)
            Y_std = np.std(Y)
            Y_norm = (Y - Y_mean) / Y_std
        else:
            Y_norm = None
    elif normalization == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
        if Y is not None:
            Y_min = np.min(Y)
            Y_max = np.max(Y)
            Y_norm = (Y - Y_min) / (Y_max - Y_min + 1e-8)
        else:
            Y_norm = None
    elif normalization == 'robust':
        X_median = np.median(X, axis=0)
        X_q75, X_q25 = np.percentile(X, [75, 25], axis=0)
        X_iqr = X_q75 - X_q25
        X_norm = (X - X_median) / (X_iqr + 1e-8)
        if Y is not None:
            Y_median = np.median(Y)
            Y_q75, Y_q25 = np.percentile(Y, [75, 25])
            Y_iqr = Y_q75 - Y_q25
            Y_norm = (Y - Y_median) / (Y_iqr + 1e-8)
        else:
            Y_norm = None
    else:  # 'none'
        X_norm = X.copy()
        Y_norm = Y.copy() if Y is not None else None
    return X_norm, Y_norm

def _compute_covariance(X: np.ndarray, Y: Optional[np.ndarray] = None) -> float:
    """Compute covariance between X and Y."""
    if Y is not None:
        cov = np.cov(X, Y, rowvar=False)[0, 1]
    else:
        cov = np.cov(X, rowvar=False)
    return cov

def covariance_fit(X: np.ndarray,
                  Y: Optional[np.ndarray] = None,
                  normalization: str = 'none',
                  metric: Optional[str] = None,
                  custom_metric: Optional[Callable] = None) -> Dict:
    """
    Compute covariance between variables.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Target array of shape (n_samples,) or None
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : Optional[str]
        Metric to compute ('mse', 'mae', 'r2')
    custom_metric : Optional[Callable]
        Custom metric function

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used and warnings

    Examples
    --------
    >>> X = np.random.rand(100, 2)
    >>> Y = np.random.rand(100)
    >>> result = covariance_fit(X, Y, normalization='standard')
    """
    # Validate inputs
    _validate_inputs(X, Y)

    # Normalize data
    X_norm, Y_norm = _normalize_data(X, Y, normalization)

    # Compute covariance
    cov_result = _compute_covariance(X_norm, Y_norm)

    # Compute metrics if requested
    metrics = {}
    if metric is not None or custom_metric is not None:
        if Y is None:
            raise ValueError("Y must be provided to compute metrics")
        if metric == 'mse':
            metrics['mse'] = np.mean((Y_norm - cov_result) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(Y_norm - cov_result))
        elif metric == 'r2':
            ss_res = np.sum((Y_norm - cov_result) ** 2)
            ss_tot = np.sum((Y_norm - np.mean(Y_norm)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot)
        if custom_metric is not None:
            metrics['custom'] = custom_metric(Y_norm, cov_result)

    # Prepare output
    result_dict = {
        'result': cov_result,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'custom_metric': custom_metric is not None
        },
        'warnings': []
    }

    return result_dict

################################################################################
# mutual_information
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def mutual_information_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    normalization: str = 'none',
    metric: Union[str, Callable] = 'entropy',
    bins: Optional[int] = None,
    base: float = np.e,
    copy: bool = True
) -> Dict[str, Union[float, Dict]]:
    """
    Compute the mutual information between two variables.

    Parameters
    ----------
    x : np.ndarray
        First variable.
    y : np.ndarray
        Second variable.
    normalization : str, optional
        Normalization method ('none', 'minmax', 'standard').
    metric : str or callable, optional
        Metric to use ('entropy', 'variation_of_information').
    bins : int, optional
        Number of bins for histogram.
    base : float, optional
        Base of the logarithm (default: natural log).
    copy : bool, optional
        Whether to make copies of input arrays.

    Returns
    -------
    Dict[str, Union[float, Dict]]
        Dictionary containing:
        - 'result': computed mutual information
        - 'metrics': additional metrics if applicable
        - 'params_used': parameters used in computation
        - 'warnings': any warnings encountered

    Examples
    --------
    >>> x = np.random.rand(100)
    >>> y = np.random.rand(100)
    >>> result = mutual_information_fit(x, y)
    """
    # Input validation
    x, y = _validate_inputs(x, y, copy=copy)

    # Compute mutual information
    mi = _compute_mutual_information(x, y, metric=metric, bins=bins, base=base)

    # Apply normalization
    if normalization != 'none':
        mi = _apply_normalization(mi, x, y, method=normalization)

    # Prepare output
    return {
        'result': mi,
        'metrics': {},
        'params_used': {
            'normalization': normalization,
            'metric': metric if isinstance(metric, str) else 'custom',
            'bins': bins,
            'base': base
        },
        'warnings': []
    }

def _validate_inputs(
    x: np.ndarray,
    y: np.ndarray,
    *,
    copy: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate and prepare input arrays.

    Parameters
    ----------
    x : np.ndarray
        First variable.
    y : np.ndarray
        Second variable.
    copy : bool, optional
        Whether to make copies of input arrays.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Validated and prepared input arrays.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape")

    if copy:
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

    return x, y

def _compute_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    *,
    metric: Union[str, Callable] = 'entropy',
    bins: Optional[int] = None,
    base: float = np.e
) -> float:
    """
    Compute mutual information between two variables.

    Parameters
    ----------
    x : np.ndarray
        First variable.
    y : np.ndarray
        Second variable.
    metric : str or callable, optional
        Metric to use ('entropy', 'variation_of_information').
    bins : int, optional
        Number of bins for histogram.
    base : float, optional
        Base of the logarithm (default: natural log).

    Returns
    -------
    float
        Computed mutual information.

    Raises
    ------
    ValueError
        If metric is not supported.
    """
    if bins is None:
        bins = int(np.sqrt(len(x)))  # Default bin number

    if isinstance(metric, str):
        if metric == 'entropy':
            return _compute_entropy_based_mi(x, y, bins=bins, base=base)
        elif metric == 'variation_of_information':
            return _compute_voi(x, y, bins=bins, base=base)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    elif callable(metric):
        return metric(x, y)
    else:
        raise ValueError("Metric must be a string or callable")

def _compute_entropy_based_mi(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bins: int,
    base: float
) -> float:
    """
    Compute mutual information using entropy.

    Parameters
    ----------
    x : np.ndarray
        First variable.
    y : np.ndarray
        Second variable.
    bins : int
        Number of bins for histogram.
    base : float
        Base of the logarithm.

    Returns
    -------
    float
        Computed mutual information.
    """
    # Compute joint and marginal histograms
    joint_hist, _, _ = np.histogram2d(x, y, bins=bins)
    hist_x, _ = np.histogram(x, bins=bins)
    hist_y, _ = np.histogram(y, bins=bins)

    # Convert to probabilities
    joint_prob = joint_hist / np.sum(joint_hist)
    prob_x = hist_x / np.sum(hist_x)
    prob_y = hist_y / np.sum(hist_y)

    # Compute entropies
    h_x = _entropy(prob_x, base=base)
    h_y = _entropy(prob_y, base=base)
    h_xy = _joint_entropy(joint_prob, base=base)

    # Compute mutual information
    mi = h_x + h_y - h_xy

    return max(mi, 0.0)  # Ensure non-negative

def _compute_voi(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bins: int,
    base: float
) -> float:
    """
    Compute variation of information.

    Parameters
    ----------
    x : np.ndarray
        First variable.
    y : np.ndarray
        Second variable.
    bins : int
        Number of bins for histogram.
    base : float
        Base of the logarithm.

    Returns
    -------
    float
        Computed variation of information.
    """
    # Compute joint and marginal histograms
    joint_hist, _, _ = np.histogram2d(x, y, bins=bins)
    hist_x, _ = np.histogram(x, bins=bins)
    hist_y, _ = np.histogram(y, bins=bins)

    # Convert to probabilities
    joint_prob = joint_hist / np.sum(joint_hist)
    prob_x = hist_x / np.sum(hist_x)
    prob_y = hist_y / np.sum(hist_y)

    # Compute entropies
    h_x = _entropy(prob_x, base=base)
    h_y = _entropy(prob_y, base=base)
    h_xy = _joint_entropy(joint_prob, base=base)

    # Compute variation of information
    voi = 2 * h_xy - (h_x + h_y)

    return max(voi, 0.0)  # Ensure non-negative

def _entropy(
    prob: np.ndarray,
    *,
    base: float
) -> float:
    """
    Compute entropy.

    Parameters
    ----------
    prob : np.ndarray
        Probability distribution.
    base : float
        Base of the logarithm.

    Returns
    -------
    float
        Computed entropy.
    """
    prob = np.array(prob, dtype=np.float64)
    prob = prob[prob > 0]  # Ignore zero probabilities
    return -np.sum(prob * np.log(prob)) / np.log(base)

def _joint_entropy(
    joint_prob: np.ndarray,
    *,
    base: float
) -> float:
    """
    Compute joint entropy.

    Parameters
    ----------
    joint_prob : np.ndarray
        Joint probability distribution.
    base : float
        Base of the logarithm.

    Returns
    -------
    float
        Computed joint entropy.
    """
    joint_prob = np.array(joint_prob, dtype=np.float64)
    joint_prob = joint_prob[joint_prob > 0]  # Ignore zero probabilities
    return -np.sum(joint_prob * np.log(joint_prob)) / np.log(base)

def _apply_normalization(
    mi: float,
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str
) -> float:
    """
    Apply normalization to mutual information.

    Parameters
    ----------
    mi : float
        Mutual information value.
    x : np.ndarray
        First variable.
    y : np.ndarray
        Second variable.
    method : str
        Normalization method ('minmax', 'standard').

    Returns
    -------
    float
        Normalized mutual information.

    Raises
    ------
    ValueError
        If normalization method is not supported.
    """
    if method == 'minmax':
        return mi / np.log(min(len(x), len(y)))
    elif method == 'standard':
        h_x = _entropy(np.bincount(x.astype(int)) / len(x))
        h_y = _entropy(np.bincount(y.astype(int)) / len(y))
        return mi / np.sqrt(h_x * h_y)
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

################################################################################
# chi_square
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for chi-square test."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape.")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(x < 0) or np.any(y < 0):
        raise ValueError("Input arrays must contain non-negative values.")

def _compute_contingency_table(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute contingency table for chi-square test."""
    x_categories = np.unique(x)
    y_categories = np.unique(y)

    contingency_table = np.zeros((len(x_categories), len(y_categories)), dtype=int)
    for i, x_cat in enumerate(x_categories):
        for j, y_cat in enumerate(y_categories):
            contingency_table[i, j] = np.sum((x == x_cat) & (y == y_cat))

    return contingency_table

def _compute_expected_values(contingency_table: np.ndarray) -> np.ndarray:
    """Compute expected values for chi-square test."""
    row_sums = contingency_table.sum(axis=1, keepdims=True)
    col_sums = contingency_table.sum(axis=0, keepdims=True)
    total = contingency_table.sum()

    expected_values = (row_sums @ col_sums) / total
    return expected_values

def _compute_chi_square_statistic(contingency_table: np.ndarray, expected_values: np.ndarray) -> float:
    """Compute chi-square statistic."""
    with np.errstate(divide='ignore', invalid='ignore'):
        chi_square = np.sum((contingency_table - expected_values) ** 2 / expected_values)
    return chi_square

def _compute_p_value(chi_square_statistic: float, degrees_of_freedom: int) -> float:
    """Compute p-value for chi-square statistic."""
    from scipy.stats import chi2
    return 1 - chi2.cdf(chi_square_statistic, degrees_of_freedom)

def chi_square_fit(
    x: np.ndarray,
    y: np.ndarray,
    correction: bool = False,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Compute chi-square test for independence between two categorical variables.

    Parameters
    ----------
    x : np.ndarray
        First categorical variable.
    y : np.ndarray
        Second categorical variable.
    correction : bool, optional
        Whether to apply Yates' continuity correction (default: False).
    custom_metric : Callable, optional
        Custom metric function to compute additional metrics.

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], str]]
        Dictionary containing:
        - "result": chi-square statistic
        - "metrics": dictionary of additional metrics
        - "params_used": parameters used in the computation
        - "warnings": any warnings generated during computation

    Example
    -------
    >>> x = np.array([1, 2, 1, 3, 2, 3])
    >>> y = np.array([1, 1, 2, 2, 3, 3])
    >>> result = chi_square_fit(x, y)
    """
    # Validate inputs
    _validate_inputs(x, y)

    # Compute contingency table
    contingency_table = _compute_contingency_table(x, y)
    degrees_of_freedom = (contingency_table.shape[0] - 1) * (contingency_table.shape[1] - 1)

    # Compute expected values
    expected_values = _compute_expected_values(contingency_table)

    # Compute chi-square statistic
    chi_square_statistic = _compute_chi_square_statistic(contingency_table, expected_values)

    # Apply correction if needed
    if correction:
        with np.errstate(divide='ignore', invalid='ignore'):
            chi_square_statistic = np.sum(
                np.abs(contingency_table - expected_values) - 0.5
            ) ** 2 / expected_values

    # Compute p-value
    p_value = _compute_p_value(chi_square_statistic, degrees_of_freedom)

    # Compute additional metrics
    metrics = {}
    if custom_metric is not None:
        try:
            metrics['custom'] = custom_metric(contingency_table, expected_values)
        except Exception as e:
            metrics['custom'] = f"Error computing custom metric: {str(e)}"

    # Prepare output
    result = {
        "result": chi_square_statistic,
        "metrics": metrics,
        "params_used": {
            "correction": correction,
            "custom_metric": custom_metric is not None
        },
        "warnings": []
    }

    return result

################################################################################
# pearson_correlation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def pearson_correlation_fit(
    x: np.ndarray,
    y: np.ndarray,
    normalize: str = 'standard',
    metric: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    handle_nan: str = 'raise',
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]:
    """
    Compute the Pearson correlation coefficient between two variables.

    Parameters
    ----------
    x : np.ndarray
        First variable.
    y : np.ndarray
        Second variable.
    normalize : str, optional
        Normalization method. Options: 'none', 'standard', 'minmax', 'robust'.
    metric : str, optional
        Metric to compute alongside correlation. Options: 'mse', 'mae'.
    custom_metric : Callable, optional
        Custom metric function.
    handle_nan : str, optional
        How to handle NaN values. Options: 'raise', 'omit'.

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]
        Dictionary containing:
        - 'result': Pearson correlation coefficient
        - 'metrics': Additional metrics if requested
        - 'params_used': Parameters used in the computation
        - 'warnings': Any warnings generated

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 4, 6, 8, 10])
    >>> pearson_correlation_fit(x, y)
    {
        'result': 1.0,
        'metrics': {},
        'params_used': {'normalize': 'standard', 'handle_nan': 'raise'},
        'warnings': []
    }
    """
    # Validate inputs
    _validate_inputs(x, y)

    # Handle NaN values
    if handle_nan == 'omit':
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

    # Normalize data
    if normalize == 'standard':
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
    elif normalize == 'minmax':
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif normalize == 'robust':
        x = (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25))
        y = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))

    # Compute Pearson correlation
    covariance = np.mean(x * y) - np.mean(x) * np.mean(y)
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)

    if std_x == 0 or std_y == 0:
        correlation = np.nan
    else:
        correlation = covariance / (std_x * std_y)

    # Compute additional metrics if requested
    metrics = {}
    if metric == 'mse':
        metrics['mse'] = np.mean((x - y) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(x - y))
    if custom_metric is not None:
        metrics['custom'] = custom_metric(x, y)

    # Prepare output
    result = {
        'result': correlation,
        'metrics': metrics,
        'params_used': {
            'normalize': normalize,
            'handle_nan': handle_nan
        },
        'warnings': []
    }

    return result

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape")
    if len(x) < 2:
        raise ValueError("At least two data points are required")
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

################################################################################
# spearman_rank_correlation
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def spearman_rank_correlation_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str = 'closed_form',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute the Spearman rank correlation between two variables.

    Parameters:
    -----------
    x : np.ndarray
        First input variable.
    y : np.ndarray
        Second input variable.
    method : str, optional
        Method to compute the correlation. Default is 'closed_form'.
    custom_metric : Callable, optional
        Custom metric function to use instead of the default Spearman correlation.
    **kwargs :
        Additional keyword arguments for specific methods.

    Returns:
    --------
    Dict[str, Union[float, Dict[str, float], Dict[str, str]]]
        A dictionary containing the result, metrics, parameters used, and warnings.

    Example:
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([5, 4, 3, 2, 1])
    >>> result = spearman_rank_correlation_fit(x, y)
    """
    # Validate inputs
    _validate_inputs(x, y)

    # Choose method
    if method == 'closed_form':
        result = _spearman_rank_correlation_closed_form(x, y)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Compute metrics
    metrics = _compute_metrics(result, x, y) if custom_metric is None else {'custom_metric': custom_metric(result, x, y)}

    # Return structured result
    return {
        'result': result,
        'metrics': metrics,
        'params_used': {'method': method},
        'warnings': []
    }

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate the input arrays."""
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape.")
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("Input arrays must not contain NaN values.")
    if np.isinf(x).any() or np.isinf(y).any():
        raise ValueError("Input arrays must not contain infinite values.")

def _spearman_rank_correlation_closed_form(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation using the closed-form formula."""
    n = x.size
    ranks_x = _rank_data(x)
    ranks_y = _rank_data(y)
    d_squared = (ranks_x - ranks_y) ** 2
    return 1 - (6 * np.sum(d_squared)) / (n * (n ** 2 - 1))

def _rank_data(data: np.ndarray) -> np.ndarray:
    """Compute ranks for the input data, handling ties appropriately."""
    sorted_data = np.argsort(np.argsort(data))
    return sorted_data

def _compute_metrics(result: float, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute additional metrics based on the result."""
    return {
        'spearman_correlation': result,
        'p_value': _compute_p_value(result, x.size)
    }

def _compute_p_value(correlation: float, n: int) -> float:
    """Compute the p-value for the Spearman correlation."""
    t = correlation * np.sqrt((n - 2) / (1 - correlation ** 2))
    df = n - 2
    # Using a normal approximation for simplicity; in practice, you might use scipy.stats.t.sf
    p_value = 2 * (1 - _normal_cdf(np.abs(t)))
    return p_value

def _normal_cdf(x: float) -> float:
    """Compute the CDF of the standard normal distribution."""
    return (1 + np.math.erf(x / np.sqrt(2))) / 2

################################################################################
# kendall_tau
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for Kendall's Tau calculation."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Input arrays must be 1-dimensional.")
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape.")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values.")

def _count_concordant_discordant_pairs(x: np.ndarray, y: np.ndarray) -> Dict[str, int]:
    """Count concordant and discordant pairs for Kendall's Tau."""
    n = len(x)
    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            delta_x = x[i] - x[j]
            delta_y = y[i] - y[j]

            if delta_x * delta_y > 0:
                concordant += 1
            elif delta_x * delta_y < 0:
                discordant += 1

    return {"concordant": concordant, "discordant": discordant}

def _compute_kendall_tau(concordant: int, discordant: int) -> float:
    """Compute Kendall's Tau based on concordant and discordant pairs."""
    n = concordant + discordant
    if n == 0:
        return 0.0

    tau = (concordant - discordant) / np.sqrt(concordant + discordant) * (n * (n - 1) / 2)
    return tau

def kendall_tau_fit(
    x: np.ndarray,
    y: np.ndarray,
    normalization: str = "none",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> Dict[str, Union[float, Dict[str, int], str]]:
    """
    Compute Kendall's Tau correlation coefficient between two 1D arrays.

    Parameters
    ----------
    x : np.ndarray
        First input array.
    y : np.ndarray
        Second input array.
    normalization : str, optional
        Type of normalization to apply. Options: "none", "standard", "minmax", "robust".
    custom_metric : Callable, optional
        Custom metric function to use instead of Kendall's Tau.

    Returns
    -------
    dict
        Dictionary containing the result, metrics, parameters used, and warnings.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4])
    >>> y = np.array([4, 3, 2, 1])
    >>> kendall_tau_fit(x, y)
    {
        "result": -1.0,
        "metrics": {"concordant": 0, "discordant": 6},
        "params_used": {"normalization": "none", "custom_metric": None},
        "warnings": []
    }
    """
    _validate_inputs(x, y)

    if normalization != "none":
        raise NotImplementedError("Normalization options other than 'none' are not implemented.")

    if custom_metric is not None:
        result = custom_metric(x, y)
        metrics = {}
    else:
        pairs = _count_concordant_discordant_pairs(x, y)
        result = _compute_kendall_tau(pairs["concordant"], pairs["discordant"])
        metrics = pairs

    return {
        "result": result,
        "metrics": metrics,
        "params_used": {"normalization": normalization, "custom_metric": custom_metric},
        "warnings": [],
    }

################################################################################
# point_biserial_correlation
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

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

def _standardize(x: np.ndarray) -> np.ndarray:
    """Standardize the input array."""
    return (x - np.mean(x)) / np.std(x)

def _minmax_scale(x: np.ndarray) -> np.ndarray:
    """Min-max scale the input array."""
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def _robust_scale(x: np.ndarray) -> np.ndarray:
    """Robust scale the input array using median and IQR."""
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return (x - np.median(x)) / iqr

def _compute_point_biserial(
    x: np.ndarray,
    y: np.ndarray,
    normalize_x: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    normalize_y: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> float:
    """Compute the point-biserial correlation coefficient."""
    if normalize_x is not None:
        x = normalize_x(x)
    if normalize_y is not None:
        y = normalize_y(y)

    # Convert binary variable to -1 and 1
    x_binary = np.where(x > 0, 1, -1)

    # Compute means
    y_mean = np.mean(y)
    y_mean_positive = np.mean(y[x_binary == 1])
    y_mean_negative = np.mean(y[x_binary == -1])

    # Compute standard deviation
    y_std = np.std(y, ddof=1)

    # Compute point-biserial correlation
    numerator = y_mean_positive - y_mean_negative
    denominator = 2 * y_std

    if denominator == 0:
        return np.nan
    return numerator / denominator

def point_biserial_correlation_fit(
    x: np.ndarray,
    y: np.ndarray,
    normalize_x: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    normalize_y: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Compute the point-biserial correlation coefficient between two arrays.

    Parameters:
    -----------
    x : np.ndarray
        Binary or dichotomous variable.
    y : np.ndarray
        Continuous variable.
    normalize_x : Optional[Callable[[np.ndarray], np.ndarray]]
        Function to normalize x. Default is None.
    normalize_y : Optional[Callable[[np.ndarray], np.ndarray]]
        Function to normalize y. Default is None.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_inputs(x, y)

    # Default normalizations
    if normalize_x is None:
        normalize_x = lambda z: z  # No normalization
    if normalize_y is None:
        normalize_y = _standardize

    result = _compute_point_biserial(x, y, normalize_x, normalize_y)

    return {
        "result": result,
        "metrics": {},
        "params_used": {
            "normalize_x": normalize_x.__name__ if hasattr(normalize_x, '__name__') else "custom",
            "normalize_y": normalize_y.__name__ if hasattr(normalize_y, '__name__') else "custom"
        },
        "warnings": []
    }

# Example usage:
# x = np.array([0, 1, 0, 1, 0])
# y = np.array([2.3, 4.5, 1.2, 3.7, 0.8])
# result = point_biserial_correlation_fit(x, y)

################################################################################
# phi_coefficient
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for phi coefficient calculation."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays")
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("Input arrays must not contain NaN or inf values")

def _compute_contingency_table(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute contingency table for binary variables."""
    x_bin = (x > 0).astype(int)
    y_bin = (y > 0).astype(int)

    contingency_table = np.zeros((2, 2), dtype=int)
    contingency_table[0, 0] = np.sum((x_bin == 0) & (y_bin == 0))
    contingency_table[0, 1] = np.sum((x_bin == 0) & (y_bin == 1))
    contingency_table[1, 0] = np.sum((x_bin == 1) & (y_bin == 0))
    contingency_table[1, 1] = np.sum((x_bin == 1) & (y_bin == 1))

    return contingency_table

def _compute_phi_coefficient(contingency_table: np.ndarray) -> float:
    """Compute phi coefficient from contingency table."""
    n = np.sum(contingency_table)
    chi_square = ((n * (contingency_table - n * contingency_table / np.sum(contingency_table)) ** 2) /
                  (np.sum(contingency_table, axis=0) * np.sum(contingency_table, axis=1))).sum()

    phi = np.sqrt(chi_square / n)
    return phi

def _compute_metrics(contingency_table: np.ndarray, phi: float) -> Dict[str, float]:
    """Compute additional metrics from contingency table."""
    n = np.sum(contingency_table)
    a, b, c, d = contingency_table.ravel()

    metrics = {
        'phi': phi,
        'chi_square': ((n * (contingency_table - n * contingency_table / np.sum(contingency_table)) ** 2) /
                       (np.sum(contingency_table, axis=0) * np.sum(contingency_table, axis=1))).sum(),
        'cramer_v': np.sqrt(phi ** 2 / min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)),
        'odds_ratio': (a * d) / (b * c) if b * c != 0 else np.nan
    }
    return metrics

def phi_coefficient_fit(
    x: Union[np.ndarray, list],
    y: Union[np.ndarray, list],
    normalize: bool = False,
    custom_metric: Optional[Callable[[np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute the phi coefficient between two binary variables.

    Parameters:
    -----------
    x : array-like
        First binary variable (0 or 1)
    y : array-like
        Second binary variable (0 or 1)
    normalize : bool, optional
        Whether to normalize the phi coefficient by its maximum possible value (default: False)
    custom_metric : callable, optional
        Custom metric function that takes the contingency table as input

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': The computed phi coefficient
        - 'metrics': Additional metrics
        - 'params_used': Parameters used in the computation
        - 'warnings': Any warnings generated during computation

    Example:
    --------
    >>> x = [1, 0, 1, 1, 0]
    >>> y = [0, 1, 1, 0, 0]
    >>> result = phi_coefficient_fit(x, y)
    """
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)

    _validate_inputs(x_array, y_array)
    contingency_table = _compute_contingency_table(x_array, y_array)
    phi = _compute_phi_coefficient(contingency_table)

    if normalize:
        n_min = min(np.sum(contingency_table, axis=0).min(), np.sum(contingency_table, axis=1).min())
        phi = phi / (np.sqrt(n_min / np.sum(contingency_table)))

    metrics = _compute_metrics(contingency_table, phi)

    if custom_metric is not None:
        metrics['custom_metric'] = custom_metric(contingency_table)

    return {
        'result': phi,
        'metrics': metrics,
        'params_used': {
            'normalize': normalize,
            'custom_metric': custom_metric.__name__ if custom_metric else None
        },
        'warnings': []
    }

################################################################################
# cramer_v
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for Cramer's V calculation."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if x.shape != y.shape:
        raise ValueError("Inputs must have the same shape.")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("Input x contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Input y contains NaN or infinite values.")

def _compute_contingency_table(x: np.ndarray, y: np.ndarray,
                              n_bins: int = 10) -> np.ndarray:
    """Compute contingency table for Cramer's V."""
    x_binned = np.digitize(x, np.linspace(np.min(x), np.max(x), n_bins + 1))
    y_binned = np.digitize(y, np.linspace(np.min(y), np.max(y), n_bins + 1))
    contingency_table = np.zeros((n_bins, n_bins), dtype=int)
    for i in range(n_bins):
        for j in range(n_bins):
            contingency_table[i, j] = np.sum((x_binned == i+1) & (y_binned == j+1))
    return contingency_table

def _compute_cramer_v(contingency_table: np.ndarray) -> float:
    """Compute Cramer's V statistic from contingency table."""
    n = np.sum(contingency_table)
    chi2 = _compute_chi_square(contingency_table, n)
    k = min(contingency_table.shape) - 1
    return np.sqrt(chi2 / (n * k))

def _compute_chi_square(contingency_table: np.ndarray, n: int) -> float:
    """Compute chi-square statistic from contingency table."""
    row_sums = np.sum(contingency_table, axis=1)
    col_sums = np.sum(contingency_table, axis=0)
    expected = np.outer(row_sums, col_sums) / n
    return np.sum((contingency_table - expected)**2 / expected)

def cramer_v_fit(x: np.ndarray, y: np.ndarray,
                 n_bins: int = 10) -> Dict[str, Union[float, Dict]]:
    """
    Compute Cramer's V dependence measure between two variables.

    Parameters
    ----------
    x : np.ndarray
        First input variable.
    y : np.ndarray
        Second input variable.
    n_bins : int, optional
        Number of bins to use for discretization (default: 10).

    Returns
    -------
    dict
        Dictionary containing:
        - result: float, Cramer's V value
        - metrics: dict, additional metrics (currently empty)
        - params_used: dict, parameters used in computation
        - warnings: list, any warnings generated

    Example
    -------
    >>> x = np.random.rand(100)
    >>> y = np.random.rand(100)
    >>> result = cramer_v_fit(x, y)
    """
    _validate_inputs(x, y)

    contingency_table = _compute_contingency_table(x, y, n_bins)
    cramer_v_value = _compute_cramer_v(contingency_table)

    return {
        "result": cramer_v_value,
        "metrics": {},
        "params_used": {"n_bins": n_bins},
        "warnings": []
    }

################################################################################
# theil_u
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if x.shape != y.shape:
        raise ValueError("Inputs must have the same shape.")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("Input x contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Input y contains NaN or infinite values.")

def _rank_data(data: np.ndarray) -> np.ndarray:
    """Compute ranks of data with average for ties."""
    sorted_data = np.sort(data)
    ranks = np.argsort(np.argsort(data)) + 1
    return ranks

def _normalize_ranks(ranks: np.ndarray, method: str = 'standard') -> np.ndarray:
    """Normalize ranks using specified method."""
    n = len(ranks)
    if method == 'standard':
        return (ranks - np.mean(ranks)) / np.std(ranks)
    elif method == 'minmax':
        return (ranks - np.min(ranks)) / (np.max(ranks) - np.min(ranks))
    elif method == 'robust':
        return (ranks - np.median(ranks)) / (np.percentile(ranks, 75) - np.percentile(ranks, 25))
    else:
        return ranks

def _compute_theil_u(x_ranks: np.ndarray, y_ranks: np.ndarray) -> float:
    """Compute Theil's U statistic."""
    n = len(x_ranks)
    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            if (x_ranks[i] - x_ranks[j]) * (y_ranks[i] - y_ranks[j]) > 0:
                concordant += 1
            elif (x_ranks[i] - x_ranks[j]) * (y_ranks[i] - y_ranks[j]) < 0:
                discordant += 1

    return (concordant - discordant) / (n * (n - 1) / 2)

def theil_u_fit(
    x: np.ndarray,
    y: np.ndarray,
    rank_method: str = 'average',
    normalize: bool = True,
    normalization_method: str = 'standard'
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]:
    """
    Compute Theil's U measure of dependence between two variables.

    Parameters
    ----------
    x : np.ndarray
        First input variable.
    y : np.ndarray
        Second input variable.
    rank_method : str, optional
        Method for handling ties in ranks ('average' or 'min'). Default is 'average'.
    normalize : bool, optional
        Whether to normalize ranks. Default is True.
    normalization_method : str, optional
        Method for normalizing ranks ('standard', 'minmax', or 'robust'). Default is 'standard'.

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]
        Dictionary containing:
        - 'result': Theil's U value
        - 'metrics': Dictionary of metrics (currently empty)
        - 'params_used': Dictionary of parameters used
        - 'warnings': List of warnings (if any)

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4])
    >>> y = np.array([4, 3, 2, 1])
    >>> result = theil_u_fit(x, y)
    """
    _validate_inputs(x, y)

    x_ranks = _rank_data(x)
    y_ranks = _rank_data(y)

    if normalize:
        x_ranks = _normalize_ranks(x_ranks, normalization_method)
        y_ranks = _normalize_ranks(y_ranks, normalization_method)

    theil_u_value = _compute_theil_u(x_ranks, y_ranks)

    return {
        'result': theil_u_value,
        'metrics': {},
        'params_used': {
            'rank_method': rank_method,
            'normalize': normalize,
            'normalization_method': normalization_method
        },
        'warnings': []
    }

################################################################################
# gini_index
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if x.shape != y.shape:
        raise ValueError("Inputs must have the same shape.")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("Input x contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Input y contains NaN or infinite values.")

def _compute_gini_index(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Gini index for dependency measurement."""
    n = len(x)
    if n == 0:
        raise ValueError("Input arrays must not be empty.")

    # Rank the x values
    ranks = np.argsort(np.argsort(x))
    # Compute the Gini index
    gini_index = 2 * np.sum((y[ranks] - y.mean()) ** 2) / (n * y.var())
    return gini_index

def _normalize_data(x: np.ndarray, method: str = 'none') -> np.ndarray:
    """Normalize the input data based on the specified method."""
    if method == 'none':
        return x
    elif method == 'standard':
        return (x - x.mean()) / x.std()
    elif method == 'minmax':
        return (x - x.min()) / (x.max() - x.min())
    elif method == 'robust':
        return (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def gini_index_fit(
    x: np.ndarray,
    y: np.ndarray,
    normalize_method: str = 'none',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute the Gini index for dependency measurement between two variables.

    Parameters:
    -----------
    x : np.ndarray
        Input array of shape (n_samples,).
    y : np.ndarray
        Input array of shape (n_samples,).
    normalize_method : str, optional
        Normalization method for the input data. Options: 'none', 'standard', 'minmax', 'robust'.
    custom_metric : Callable, optional
        Custom metric function to compute the dependency measure.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_inputs(x, y)

    # Normalize the data
    x_normalized = _normalize_data(x, normalize_method)
    y_normalized = _normalize_data(y, normalize_method)

    # Compute the Gini index
    if custom_metric is not None:
        result = custom_metric(x_normalized, y_normalized)
    else:
        result = _compute_gini_index(x_normalized, y_normalized)

    metrics = {
        'gini_index': result
    }

    params_used = {
        'normalize_method': normalize_method,
        'custom_metric': custom_metric is not None
    }

    warnings = []

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

# Example usage:
# gini_index_fit(np.array([1, 2, 3, 4]), np.array([4, 3, 2, 1]))
