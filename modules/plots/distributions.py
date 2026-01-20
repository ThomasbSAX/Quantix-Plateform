from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class FitResult:
    name: str
    params: Tuple[float, ...]


def _try_import_scipy():
    try:
        import scipy.stats as st  # type: ignore

        return st
    except Exception:
        return None


def fit_distribution(x: np.ndarray, name: str) -> Optional[FitResult]:
    """Fit a known distribution. Returns None if SciPy unavailable.

    Supported names: normal, lognormal, exponential, uniform.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return None

    st = _try_import_scipy()
    if st is None:
        # Fallback: only normal via mean/std.
        if name == "normal":
            mu = float(np.mean(x))
            sigma = float(np.std(x, ddof=1))
            if sigma <= 0:
                return None
            return FitResult("normal", (mu, sigma))
        return None

    try:
        if name == "normal":
            mu, sigma = st.norm.fit(x)
            return FitResult("normal", (float(mu), float(sigma)))
        if name == "lognormal":
            # Constrain loc=0 for stability on positive data.
            shape, loc, scale = st.lognorm.fit(x, floc=0)
            return FitResult("lognormal", (float(shape), float(loc), float(scale)))
        if name == "exponential":
            loc, scale = st.expon.fit(x)
            return FitResult("exponential", (float(loc), float(scale)))
        if name == "uniform":
            loc, scale = st.uniform.fit(x)
            return FitResult("uniform", (float(loc), float(scale)))
    except Exception:
        return None

    return None


def pdf(name: str, params: Sequence[float], grid: np.ndarray) -> Optional[np.ndarray]:
    st = _try_import_scipy()
    grid = np.asarray(grid, dtype=float)

    if st is None:
        if name == "normal" and len(params) == 2:
            mu, sigma = float(params[0]), float(params[1])
            if sigma <= 0:
                return None
            z = (grid - mu) / sigma
            return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * z * z)
        return None

    try:
        if name == "normal":
            mu, sigma = params
            return st.norm.pdf(grid, loc=mu, scale=sigma)
        if name == "lognormal":
            shape, loc, scale = params
            return st.lognorm.pdf(grid, s=shape, loc=loc, scale=scale)
        if name == "exponential":
            loc, scale = params
            return st.expon.pdf(grid, loc=loc, scale=scale)
        if name == "uniform":
            loc, scale = params
            return st.uniform.pdf(grid, loc=loc, scale=scale)
    except Exception:
        return None

    return None
