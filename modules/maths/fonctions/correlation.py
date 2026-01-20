"""
Quantix – Module correlation
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# pearson_correlation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def pearson_correlation_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    normalize_x: bool = False,
    normalize_y: bool = False,
    metric: str = 'pearson',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[float, Dict[str, float], Dict[str, bool]]]:
    """
    Compute Pearson correlation between two variables with configurable options.

    Parameters:
    -----------
    x : np.ndarray
        First input variable.
    y : np.ndarray
        Second input variable.
    normalize_x : bool, optional
        Whether to normalize x before computation (default: False).
    normalize_y : bool, optional
        Whether to normalize y before computation (default: False).
    metric : str, optional
        Metric to use for correlation ('pearson', 'spearman') (default: 'pearson').
    custom_metric : Callable, optional
        Custom metric function if not using built-in metrics.

    Returns:
    --------
    Dict containing:
        - result: float, the computed correlation value
        - metrics: dict of additional metrics if applicable
        - params_used: dict of parameters used in computation
        - warnings: dict of any warnings encountered

    Example:
    --------
    >>> x = np.array([1, 2, 3, 4])
    >>> y = np.array([4, 3, 2, 1])
    >>> pearson_correlation_fit(x, y)
    """
    # Validate inputs
    _validate_inputs(x, y)

    # Normalize data if requested
    x_normalized = _normalize_data(x) if normalize_x else x.copy()
    y_normalized = _normalize_data(y) if normalize_y else y.copy()

    # Compute correlation
    result, metrics = _compute_correlation(
        x_normalized,
        y_normalized,
        metric=metric,
        custom_metric=custom_metric
    )

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalize_x': normalize_x,
            'normalize_y': normalize_y,
            'metric': metric
        },
        'warnings': {}
    }

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape")
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("Input arrays must not contain NaN values")
    if np.isinf(x).any() or np.isinf(y).any():
        raise ValueError("Input arrays must not contain infinite values")

def _normalize_data(data: np.ndarray) -> np.ndarray:
    """Normalize data to zero mean and unit variance."""
    return (data - np.mean(data)) / np.std(data)

def _compute_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    metric: str = 'pearson',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> tuple:
    """Compute correlation using specified metric."""
    if custom_metric is not None:
        result = custom_metric(x, y)
        return result, {}

    if metric == 'pearson':
        cov = np.cov(x, y)[0, 1]
        std_x = np.std(x)
        std_y = np.std(y)
        result = cov / (std_x * std_y)
    elif metric == 'spearman':
        x_rank = _rank_data(x)
        y_rank = _rank_data(y)
        result, _ = _compute_correlation(x_rank, y_rank, metric='pearson')
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return result, {}

def _rank_data(data: np.ndarray) -> np.ndarray:
    """Convert data to ranks."""
    sorted_indices = np.argsort(data)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(len(data))
    return ranks

################################################################################
# spearman_rank_correlation
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for Spearman rank correlation."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape.")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("Input arrays must not contain NaN or inf values.")

def _rank_data(data: np.ndarray) -> np.ndarray:
    """Compute ranks of data with average ranks for ties."""
    sorted_data = np.sort(data)
    ranks = np.zeros_like(data, dtype=float)
    i = 0
    while i < len(sorted_data):
        j = i
        while j < len(sorted_data) and sorted_data[j] == sorted_data[i]:
            j += 1
        ranks[data == sorted_data[i]] = (i + 1 + j - 1) / 2
        i = j
    return ranks

def _spearman_correlation(ranks_x: np.ndarray, ranks_y: np.ndarray) -> float:
    """Compute Spearman rank correlation coefficient."""
    n = len(ranks_x)
    cov = np.cov(ranks_x, ranks_y)[0, 1]
    var_x = np.var(ranks_x)
    var_y = np.var(ranks_y)
    return cov / np.sqrt(var_x * var_y)

def spearman_rank_correlation_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    normalization: Optional[str] = None,
    metric: str = "spearman",
    handle_ties: bool = True
) -> Dict[str, Any]:
    """
    Compute Spearman rank correlation between two arrays.

    Parameters:
    -----------
    x : np.ndarray
        First input array.
    y : np.ndarray
        Second input array.
    normalization : str, optional
        Normalization method (not used for Spearman but kept for consistency).
    metric : str, optional
        Metric to compute (must be "spearman" for this function).
    handle_ties : bool, optional
        Whether to handle ties in the data (default: True).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.

    Example:
    --------
    >>> x = np.array([1, 2, 3, 4])
    >>> y = np.array([4, 3, 2, 1])
    >>> spearman_rank_correlation_fit(x, y)
    {
        'result': -1.0,
        'metrics': {'spearman': -1.0},
        'params_used': {
            'normalization': None,
            'metric': 'spearman',
            'handle_ties': True
        },
        'warnings': []
    }
    """
    _validate_inputs(x, y)

    ranks_x = _rank_data(x)
    ranks_y = _rank_data(y)

    correlation = _spearman_correlation(ranks_x, ranks_y)

    return {
        "result": correlation,
        "metrics": {"spearman": correlation},
        "params_used": {
            "normalization": normalization,
            "metric": metric,
            "handle_ties": handle_ties
        },
        "warnings": []
    }

################################################################################
# kendall_tau_correlation
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for Kendall's tau correlation."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Input arrays must be 1-dimensional.")
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape.")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("Input arrays must contain only finite values.")

def _count_concordant_discordant_pairs(x: np.ndarray, y: np.ndarray) -> Dict[str, int]:
    """Count concordant and discordant pairs for Kendall's tau."""
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
    """Compute Kendall's tau correlation coefficient."""
    n = concordant + discordant
    if n == 0:
        return np.nan

    tau = (concordant - discordant) / n
    return tau

def kendall_tau_correlation_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    normalization: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute Kendall's tau correlation coefficient between two arrays.

    Parameters:
    -----------
    x : np.ndarray
        First input array.
    y : np.ndarray
        Second input array.
    normalization : str, optional
        Normalization method (not used in Kendall's tau but kept for consistency).
    custom_metric : callable, optional
        Custom metric function (not used in Kendall's tau but kept for consistency).

    Returns:
    --------
    dict
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_inputs(x, y)

    pairs = _count_concordant_discordant_pairs(x, y)
    tau = _compute_kendall_tau(pairs["concordant"], pairs["discordant"])

    result = {
        "result": tau,
        "metrics": {"concordant_pairs": pairs["concordant"], "discordant_pairs": pairs["discordant"]},
        "params_used": {
            "normalization": normalization,
            "custom_metric": custom_metric.__name__ if custom_metric else None
        },
        "warnings": []
    }

    return result

# Example usage:
# x = np.array([1, 2, 3, 4])
# y = np.array([4, 3, 2, 1])
# result = kendall_tau_correlation_fit(x, y)

################################################################################
# point_biserial_correlation
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if len(x) != len(y):
        raise ValueError("Inputs must have the same length.")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("x contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")
    if not np.issubdtype(y.dtype, np.number):
        raise ValueError("y must be a binary array (0 or 1).")

def _standardize(x: np.ndarray) -> np.ndarray:
    """Standardize the input array."""
    return (x - np.mean(x)) / np.std(x)

def _minmax_scale(x: np.ndarray) -> np.ndarray:
    """Min-max scale the input array."""
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def _robust_scale(x: np.ndarray) -> np.ndarray:
    """Robust scale the input array using median and IQR."""
    median = np.median(x)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return (x - median) / iqr

def _compute_point_biserial(
    x: np.ndarray,
    y: np.ndarray,
    normalize_x: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> float:
    """Compute the point-biserial correlation coefficient."""
    if normalize_x is not None:
        x = normalize_x(x)
    mean_x_0 = np.mean(x[y == 0])
    mean_x_1 = np.mean(x[y == 1])
    std_x = np.std(x, ddof=1)
    n_0 = np.sum(y == 0)
    n_1 = np.sum(y == 1)
    numerator = mean_x_1 - mean_x_0
    denominator = std_x * np.sqrt(n_0 * n_1 / len(x))
    return numerator / denominator

def point_biserial_correlation_fit(
    x: np.ndarray,
    y: np.ndarray,
    normalize_x: Optional[str] = None,
    custom_normalize: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Compute the point-biserial correlation coefficient between x and y.

    Parameters:
    -----------
    x : np.ndarray
        Continuous variable.
    y : np.ndarray
        Binary variable (0 or 1).
    normalize_x : str, optional
        Normalization method for x. Options: 'standard', 'minmax', 'robust'.
    custom_normalize : callable, optional
        Custom normalization function.

    Returns:
    --------
    dict
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_inputs(x, y)

    normalize_func = None
    if normalize_x == 'standard':
        normalize_func = _standardize
    elif normalize_x == 'minmax':
        normalize_func = _minmax_scale
    elif normalize_x == 'robust':
        normalize_func = _robust_scale
    elif custom_normalize is not None:
        normalize_func = custom_normalize

    result = _compute_point_biserial(x, y, normalize_func)

    return {
        "result": result,
        "metrics": {},
        "params_used": {
            "normalize_x": normalize_x if normalize_func is not None else "none",
            "custom_normalize": custom_normalize is not None
        },
        "warnings": []
    }

################################################################################
# phi_coefficient
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

def _compute_phi_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the phi coefficient between two binary arrays."""
    contingency_table = np.array([
        [np.sum((x == 0) & (y == 0)),
         np.sum((x == 0) & (y == 1))],
        [np.sum((x == 1) & (y == 0)),
         np.sum((x == 1) & (y == 1))]
    ])
    n = contingency_table.sum()
    chi_squared = ((contingency_table * (n - contingency_table)) / n).sum()
    phi = np.sqrt(chi_squared / n)
    return phi

def _normalize_data(x: np.ndarray, y: np.ndarray,
                    normalization: str = 'none') -> tuple:
    """Normalize input data based on the specified method."""
    if normalization == 'none':
        return x, y
    elif normalization == 'standard':
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
    elif normalization == 'minmax':
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif normalization == 'robust':
        x = (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25))
        y = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    else:
        raise ValueError("Invalid normalization method.")
    return x, y

def phi_coefficient_fit(x: np.ndarray,
                        y: np.ndarray,
                        normalization: str = 'none',
                        metric: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Compute the phi coefficient between two binary arrays.

    Parameters:
    -----------
    x : np.ndarray
        First binary array.
    y : np.ndarray
        Second binary array.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : Callable, optional
        Custom metric function.

    Returns:
    --------
    dict
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_inputs(x, y)
    x_norm, y_norm = _normalize_data(x, y, normalization)

    result = _compute_phi_coefficient(x_norm, y_norm)
    metrics = {}
    if metric is not None:
        metrics['custom_metric'] = metric(x_norm, y_norm)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric.__name__ if metric else None
        },
        'warnings': []
    }

# Example usage:
# x = np.array([0, 1, 0, 1, 1])
# y = np.array([0, 0, 1, 1, 1])
# result = phi_coefficient_fit(x, y)

################################################################################
# cramers_v
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for Cramér's V calculation."""
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional")
    if len(x) != len(y):
        raise ValueError("Inputs must have the same length")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("Inputs must not contain NaN or infinite values")

def _compute_contingency_table(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute contingency table from two categorical arrays."""
    x_categories = np.unique(x)
    y_categories = np.unique(y)

    table = np.zeros((len(x_categories), len(y_categories)), dtype=int)
    for i, x_cat in enumerate(x_categories):
        for j, y_cat in enumerate(y_categories):
            table[i, j] = np.sum((x == x_cat) & (y == y_cat))

    return table

def _compute_chi_square(table: np.ndarray) -> float:
    """Compute chi-square statistic from contingency table."""
    n = np.sum(table)
    row_sums = np.sum(table, axis=1)
    col_sums = np.sum(table, axis=0)

    expected = (row_sums[:, np.newaxis] * col_sums) / n
    chi2 = np.sum((table - expected)**2 / expected)
    return chi2

def cramers_v_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    correction: bool = True
) -> Dict[str, Union[float, Dict]]:
    """
    Calculate Cramér's V association measure for categorical variables.

    Parameters
    ----------
    x : np.ndarray
        First categorical variable array.
    y : np.ndarray
        Second categorical variable array.
    correction : bool, optional
        Whether to apply continuity correction (default: True).

    Returns
    -------
    Dict[str, Union[float, Dict]]
        Dictionary containing:
        - "result": Cramér's V value
        - "metrics": Dictionary with chi-square statistic and degrees of freedom
        - "params_used": Parameters used in calculation
        - "warnings": Any warnings generated

    Examples
    --------
    >>> x = np.array([1, 2, 1, 3, 2])
    >>> y = np.array([0, 1, 0, 1, 0])
    >>> result = cramers_v_fit(x, y)
    """
    _validate_inputs(x, y)

    contingency_table = _compute_contingency_table(x, y)
    chi2 = _compute_chi_square(contingency_table)

    n = np.sum(contingency_table)
    r, k = contingency_table.shape
    phi2 = chi2 / n

    if correction:
        phi2_corrected = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        v = np.sqrt(phi2_corrected / min((k-1), (r-1)))
    else:
        v = np.sqrt(phi2 / min((k-1), (r-1)))

    return {
        "result": float(v),
        "metrics": {
            "chi_square": float(chi2),
            "degrees_of_freedom": (r - 1) * (k - 1)
        },
        "params_used": {
            "correction": correction
        },
        "warnings": []
    }

################################################################################
# partial_correlation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def partial_correlation_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    control_vars: Optional[np.ndarray] = None,
    method: str = 'pearson',
    normalize: bool = True,
    solver: str = 'closed_form',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Compute partial correlation between X and y controlling for other variables.

    Parameters
    ----------
    X : np.ndarray
        Input variable (1D array)
    y : np.ndarray
        Target variable (1D array)
    control_vars : Optional[np.ndarray]
        Control variables (2D array), each column is a variable
    method : str
        Correlation method ('pearson', 'spearman')
    normalize : bool
        Whether to normalize the data
    solver : str
        Solver method ('closed_form', 'gradient_descent')
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    custom_metric : Optional[Callable]
        Custom metric function

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], str]]
        Dictionary containing:
        - 'result': partial correlation coefficient
        - 'metrics': dictionary of metrics
        - 'params_used': parameters used
        - 'warnings': any warnings

    Examples
    --------
    >>> X = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 4, 6, 8, 10])
    >>> partial_correlation_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y, control_vars)

    # Normalize data if required
    X_norm, y_norm = _normalize_data(X, y, control_vars, normalize)

    # Prepare data for partial correlation
    X_aug = _prepare_data(X_norm, y_norm, control_vars)

    # Compute partial correlation
    if solver == 'closed_form':
        result = _partial_correlation_closed_form(X_aug, method)
    elif solver == 'gradient_descent':
        result = _partial_correlation_gradient_descent(X_aug, method, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_aug, result, custom_metric)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'method': method,
            'normalize': normalize,
            'solver': solver
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray, control_vars: Optional[np.ndarray]) -> None:
    """Validate input arrays."""
    if X.ndim != 1 or y.ndim != 1:
        raise ValueError("X and y must be 1D arrays")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if control_vars is not None:
        if control_vars.ndim != 2 or len(X) != len(control_vars):
            raise ValueError("Control variables must be 2D array with same number of rows as X and y")
        if np.any(np.isnan(control_vars)) or np.any(np.isinf(control_vars)):
            raise ValueError("Control variables contain NaN or Inf values")

def _normalize_data(
    X: np.ndarray,
    y: np.ndarray,
    control_vars: Optional[np.ndarray],
    normalize: bool
) -> tuple:
    """Normalize input data."""
    if not normalize:
        return X, y

    X_norm = (X - np.mean(X)) / np.std(X)
    y_norm = (y - np.mean(y)) / np.std(y)

    if control_vars is not None:
        control_vars_norm = (control_vars - np.mean(control_vars, axis=0)) / np.std(control_vars, axis=0)
        return X_norm, y_norm, control_vars_norm
    return X_norm, y_norm

def _prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    control_vars: Optional[np.ndarray]
) -> tuple:
    """Prepare data for partial correlation computation."""
    if control_vars is None:
        return X, y

    # Combine X and control variables
    X_aug = np.column_stack([X, control_vars])
    return X_aug, y

def _partial_correlation_closed_form(
    X_aug: np.ndarray,
    method: str
) -> float:
    """Compute partial correlation using closed form solution."""
    X, y = X_aug[:, 0], X_aug[:, -1]
    control_vars = X_aug[:, 1:-1] if X_aug.shape[1] > 2 else None

    # Compute correlation matrix
    corr_matrix = np.corrcoef(np.column_stack([X, y, control_vars]), rowvar=False)

    # Get relevant correlations
    r_XY = corr_matrix[0, 1]
    r_XC = corr_matrix[0, 2:]
    r_YC = corr_matrix[1, 2:]

    # Compute partial correlation
    if control_vars is None:
        return r_XY

    numerator = r_XY - np.dot(r_XC, r_YC)
    denominator = np.sqrt(1 - np.sum(r_XC**2)) * np.sqrt(1 - np.sum(r_YC**2))

    return numerator / denominator

def _partial_correlation_gradient_descent(
    X_aug: np.ndarray,
    method: str,
    tol: float,
    max_iter: int
) -> float:
    """Compute partial correlation using gradient descent."""
    # This is a placeholder implementation
    # In practice, you would implement the actual gradient descent algorithm
    raise NotImplementedError("Gradient descent solver not implemented")

def _compute_metrics(
    X_aug: np.ndarray,
    result: float,
    custom_metric: Optional[Callable]
) -> Dict[str, float]:
    """Compute metrics for partial correlation."""
    metrics = {
        'partial_correlation': result,
        'p_value': _compute_p_value(result, len(X_aug))
    }

    if custom_metric is not None:
        X, y = X_aug[:, 0], X_aug[:, -1]
        metrics['custom_metric'] = custom_metric(X, y)

    return metrics

def _compute_p_value(
    r: float,
    n: int
) -> float:
    """Compute p-value for partial correlation."""
    df = n - 2
    t = r * np.sqrt(df / (1 - r**2))
    from scipy.stats import t as t_dist
    return 2 * (1 - t_dist.cdf(np.abs(t), df))

################################################################################
# multiple_correlation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def multiple_correlation_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalization: str = "standard",
    metric: Union[str, Callable] = "r2",
    distance: str = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Compute multiple correlation between features and target.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    normalization : str, optional
        Normalization method: "none", "standard", "minmax", or "robust".
    metric : str or callable, optional
        Metric to evaluate correlation: "mse", "mae", "r2", or custom callable.
    distance : str, optional
        Distance metric: "euclidean", "manhattan", "cosine", or "minkowski".
    solver : str, optional
        Solver method: "closed_form", "gradient_descent", or "newton".
    regularization : str, optional
        Regularization type: "none", "l1", "l2", or "elasticnet".
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function.
    custom_distance : callable, optional
        Custom distance function.

    Returns:
    --------
    Dict containing:
        - "result": Computed correlation result.
        - "metrics": Dictionary of computed metrics.
        - "params_used": Parameters used in the computation.
        - "warnings": List of warnings encountered.

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = multiple_correlation_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _normalize_data(X, method=normalization)

    # Choose solver
    if solver == "closed_form":
        coefficients = _solve_closed_form(X_normalized, y)
    elif solver == "gradient_descent":
        coefficients = _solve_gradient_descent(
            X_normalized, y,
            tol=tol,
            max_iter=max_iter,
            regularization=regularization
        )
    elif solver == "newton":
        coefficients = _solve_newton(
            X_normalized, y,
            tol=tol,
            max_iter=max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(
        X_normalized, y,
        coefficients,
        metric=metric,
        custom_metric=custom_metric
    )

    # Prepare output
    result = {
        "result": coefficients,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularization": regularization
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

def _normalize_data(X: np.ndarray, method: str = "standard") -> np.ndarray:
    """Normalize data using specified method."""
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
        raise ValueError(f"Unknown normalization method: {method}")

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve using closed-form solution."""
    XTX = np.dot(X.T, X)
    XTy = np.dot(X.T, y)
    return np.linalg.solve(XTX + 1e-8 * np.eye(X.shape[1]), XTy)

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-6,
    max_iter: int = 1000,
    regularization: Optional[str] = None
) -> np.ndarray:
    """Solve using gradient descent."""
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradient = np.dot(X.T, (np.dot(X, coefficients) - y)) / X.shape[0]

        if regularization == "l1":
            gradient += np.sign(coefficients)
        elif regularization == "l2":
            gradient += 2 * coefficients
        elif regularization == "elasticnet":
            gradient += np.sign(coefficients) + 2 * coefficients

        new_coefficients = coefficients - learning_rate * gradient
        if np.linalg.norm(new_coefficients - coefficients) < tol:
            break
        coefficients = new_coefficients

    return coefficients

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> np.ndarray:
    """Solve using Newton's method."""
    n_features = X.shape[1]
    coefficients = np.zeros(n_features)

    for _ in range(max_iter):
        residuals = np.dot(X, coefficients) - y
        gradient = np.dot(X.T, residuals)
        hessian = np.dot(X.T, X)

        delta = np.linalg.solve(hessian + 1e-8 * np.eye(n_features), -gradient)
        new_coefficients = coefficients + delta

        if np.linalg.norm(delta) < tol:
            break
        coefficients = new_coefficients

    return coefficients

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    *,
    metric: Union[str, Callable] = "r2",
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Compute metrics for the correlation."""
    predictions = np.dot(X, coefficients)
    metrics_dict = {}

    if metric == "mse" or custom_metric is None:
        mse = np.mean((predictions - y) ** 2)
        metrics_dict["mse"] = mse

    if metric == "mae" or custom_metric is None:
        mae = np.mean(np.abs(predictions - y))
        metrics_dict["mae"] = mae

    if metric == "r2" or custom_metric is None:
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        metrics_dict["r2"] = r2

    if custom_metric is not None:
        custom_value = custom_metric(y, predictions)
        metrics_dict["custom"] = custom_value

    return metrics_dict

################################################################################
# intraclass_correlation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    data: np.ndarray,
    groups: np.ndarray,
    normalize: str = "none",
    metric: Union[str, Callable] = "mse",
    solver: str = "closed_form"
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray) or not isinstance(groups, np.ndarray):
        raise TypeError("Data and groups must be numpy arrays")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")
    if groups.ndim != 1:
        raise ValueError("Groups must be a 1D array")
    if len(data) != len(groups):
        raise ValueError("Data and groups must have the same length")
    if normalize not in ["none", "standard", "minmax", "robust"]:
        raise ValueError("Invalid normalization method")
    if isinstance(metric, str) and metric not in ["mse", "mae", "r2", "logloss"]:
        raise ValueError("Invalid metric")
    if solver not in ["closed_form", "gradient_descent", "newton", "coordinate_descent"]:
        raise ValueError("Invalid solver")

def _normalize_data(
    data: np.ndarray,
    method: str = "none"
) -> np.ndarray:
    """Normalize data based on the specified method."""
    if method == "none":
        return data
    elif method == "standard":
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == "minmax":
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == "robust":
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / iqr
    else:
        raise ValueError("Invalid normalization method")

def _compute_intraclass_correlation(
    data: np.ndarray,
    groups: np.ndarray,
    normalize: str = "none",
    metric: Union[str, Callable] = "mse"
) -> float:
    """Compute the intraclass correlation coefficient."""
    if isinstance(metric, str):
        metric_func = _get_metric_function(metric)
    else:
        metric_func = metric

    normalized_data = _normalize_data(data, normalize)

    # Calculate between-group variance
    unique_groups = np.unique(groups)
    group_means = np.array([normalized_data[groups == g].mean(axis=0) for g in unique_groups])
    overall_mean = normalized_data.mean(axis=0)
    between_var = np.sum([len(normalized_data[groups == g]) * metric_func(group_means[i], overall_mean)
                          for i, g in enumerate(unique_groups)]) / (len(unique_groups) - 1)

    # Calculate within-group variance
    within_var = np.sum([metric_func(normalized_data[groups == g], group_means[i].mean(axis=0))
                         for i, g in enumerate(unique_groups)]) / (len(data) - len(unique_groups))

    # ICC formula
    icc = between_var / (between_var + within_var)
    return icc

def _get_metric_function(metric: str) -> Callable:
    """Return the appropriate metric function based on the input string."""
    metrics = {
        "mse": lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
        "mae": lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)),
        "r2": lambda y_true, y_pred: 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2),
        "logloss": lambda y_true, y_pred: -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    }
    return metrics[metric]

def intraclass_correlation_fit(
    data: np.ndarray,
    groups: np.ndarray,
    normalize: str = "none",
    metric: Union[str, Callable] = "mse",
    solver: str = "closed_form"
) -> Dict[str, Union[float, Dict, Dict]]:
    """
    Compute the intraclass correlation coefficient.

    Parameters
    ----------
    data : np.ndarray
        2D array of observations.
    groups : np.ndarray
        1D array indicating group membership for each observation.
    normalize : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    metric : str or callable, optional
        Metric to use ("mse", "mae", "r2", "logloss") or custom callable.
    solver : str, optional
        Solver method ("closed_form", "gradient_descent", "newton", "coordinate_descent").

    Returns
    -------
    dict
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_inputs(data, groups, normalize, metric, solver)

    result = _compute_intraclass_correlation(data, groups, normalize, metric)

    metrics = {
        "icc": result
    }

    params_used = {
        "normalize": normalize,
        "metric": metric if isinstance(metric, str) else "custom",
        "solver": solver
    }

    warnings = []

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

# Example usage:
# data = np.random.rand(100, 5)
# groups = np.random.randint(0, 10, size=100)
# result = intraclass_correlation_fit(data, groups)

################################################################################
# canonical_correlation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def canonical_correlation_fit(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    normalizer_X: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    normalizer_Y: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    solver: str = 'svd',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, str]]:
    """
    Compute canonical correlation between two datasets X and Y.

    Parameters
    ----------
    X : np.ndarray
        First dataset of shape (n_samples, n_features_X)
    Y : np.ndarray
        Second dataset of shape (n_samples, n_features_Y)
    normalizer_X : Optional[Callable]
        Function to normalize X. If None, no normalization.
    normalizer_Y : Optional[Callable]
        Function to normalize Y. If None, no normalization.
    solver : str
        Solver method: 'svd' or 'eig'
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable]
        Custom metric function. If None, uses default correlation.

    Returns
    -------
    Dict[str, Union[Dict, float, str]]
        Dictionary containing results, metrics, parameters used and warnings.

    Examples
    --------
    >>> X = np.random.randn(100, 5)
    >>> Y = np.random.randn(100, 3)
    >>> result = canonical_correlation_fit(X, Y)
    """
    # Validate inputs
    _validate_inputs(X, Y)

    # Normalize data if needed
    X_norm = normalizer_X(X) if normalizer_X else X
    Y_norm = normalizer_Y(Y) if normalizer_Y else Y

    # Compute canonical correlations
    if solver == 'svd':
        corr, U, V = _canonical_correlation_svd(X_norm, Y_norm)
    elif solver == 'eig':
        corr, U, V = _canonical_correlation_eig(X_norm, Y_norm)
    else:
        raise ValueError("Solver must be either 'svd' or 'eig'")

    # Compute metrics
    metrics = {}
    if custom_metric:
        metrics['custom'] = custom_metric(X_norm, Y_norm)
    else:
        metrics['correlation'] = np.mean(corr)

    # Prepare output
    result = {
        'result': {
            'correlations': corr,
            'X_weights': U,
            'Y_weights': V
        },
        'metrics': metrics,
        'params_used': {
            'normalizer_X': normalizer_X.__name__ if normalizer_X else None,
            'normalizer_Y': normalizer_Y.__name__ if normalizer_Y else None,
            'solver': solver,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, Y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
        raise ValueError("Y contains NaN or infinite values")

def _canonical_correlation_svd(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute canonical correlation using SVD."""
    n = X.shape[0]
    Cxy = np.dot(X.T, Y) / (n - 1)
    Cx = np.dot(X.T, X) / (n - 1)
    Cy = np.dot(Y.T, Y) / (n - 1)

    # Compute SVD of Cx^{-1/2} Cxy Cy^{-1/2}
    inv_sqrt_Cx = np.linalg.inv(np.linalg.cholesky(Cx))
    inv_sqrt_Cy = np.linalg.inv(np.linalg.cholesky(Cy))
    M = inv_sqrt_Cx.T @ Cxy @ inv_sqrt_Cy

    U, S, Vt = np.linalg.svd(M)
    corr = np.diag(S)

    return corr, U, Vt.T

def _canonical_correlation_eig(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute canonical correlation using eigenvalue decomposition."""
    n = X.shape[0]
    Cxy = np.dot(X.T, Y) / (n - 1)
    Cx = np.dot(X.T, X) / (n - 1)
    Cy = np.dot(Y.T, Y) / (n - 1)

    # Compute eigenvalue decomposition of Cx^{-1} Cxy Cy^{-1} Cxy^T
    inv_Cx = np.linalg.inv(Cx)
    inv_Cy = np.linalg.inv(Cy)
    M = inv_Cx @ Cxy @ inv_Cy @ Cxy.T

    eigenvalues, U = np.linalg.eigh(M)
    corr = np.sqrt(eigenvalues)

    # Compute V
    V = inv_Cy @ Cxy.T @ U

    return corr, U, V
