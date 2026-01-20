from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    return pd.to_numeric(series, errors="coerce")


def coerce_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce", utc=False)


def finite_1d(values: Iterable[Any]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr


def top_categories(series: pd.Series, top_n: int) -> pd.Series:
    vc = series.astype("string").fillna("(NA)").value_counts(dropna=False)
    if top_n and top_n > 0:
        vc = vc.head(top_n)
    return vc


def param_bool(params: Mapping[str, Any], key: str, default: bool = False) -> bool:
    v = params.get(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(v)


def param_int(params: Mapping[str, Any], key: str, default: Optional[int] = None) -> Optional[int]:
    v = params.get(key, default)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


def param_float(params: Mapping[str, Any], key: str, default: Optional[float] = None) -> Optional[float]:
    v = params.get(key, default)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def param_list_str(params: Mapping[str, Any], key: str) -> List[str]:
    v = params.get(key)
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    if isinstance(v, str):
        return [s.strip() for s in v.split(",") if s.strip()]
    return [str(v)]


def param_list_float(params: Mapping[str, Any], key: str) -> List[float]:
    v = params.get(key)
    if v is None or v == "":
        return []
    items: List[Any]
    if isinstance(v, list):
        items = v
    elif isinstance(v, str):
        items = [s.strip() for s in v.split(",") if s.strip()]
    else:
        items = [v]

    out: List[float] = []
    for it in items:
        try:
            out.append(float(it))
        except Exception:
            continue
    return out


def validate_positive_for_log(values: Iterable[Any], *, axis_label: str) -> None:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError(f"Aucune valeur finie pour l'axe {axis_label}")
    if (arr <= 0).any():
        raise ValueError(
            f"Axe {axis_label} en log impossible: valeurs <= 0 détectées. Filtre/shift requis."
        )
