from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from .utils import coerce_datetime, coerce_numeric


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        if isinstance(v, (float, int, np.floating, np.integer)):
            fv = float(v)
            return fv if np.isfinite(fv) else None
        fv = float(v)
        return fv if np.isfinite(fv) else None
    except Exception:
        return None


def profile_dataframe(
    df: pd.DataFrame,
    *,
    quantiles: List[float] | None = None,
    max_examples: int = 5,
) -> Dict[str, Any]:
    """Return a JSON-friendly profile of a DataFrame.

    Designed for a Flask+HTML frontend to:
    - know which columns are numeric/text/datetime/categorical
    - show classic stats (mean/median/quantiles, missing, nunique)
    - build forms (column pickers) safely
    """
    if quantiles is None:
        quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

    out: Dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": [],
        "quantiles": quantiles,
    }

    for col in df.columns:
        s = df[col]
        col_info: Dict[str, Any] = {
            "name": str(col),
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": float(s.isna().mean()) if len(s) else 0.0,
            "nunique": int(s.nunique(dropna=True)),
        }

        # Examples (as strings, JSON-safe)
        examples = (
            s.dropna()
            .astype(str)
            .head(max_examples)
            .tolist()
        )
        col_info["examples"] = examples

        # Type hints for UI
        is_numeric = pd.api.types.is_numeric_dtype(s)
        is_datetime = pd.api.types.is_datetime64_any_dtype(s)
        is_text_like = pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)

        if not is_datetime and is_text_like:
            # Try datetime parse on a sample, to help UI if dates come as strings.
            sample = s.dropna().head(50)
            if len(sample):
                parsed = coerce_datetime(sample)
                # If most values parsed, consider datetime-like.
                ratio = float(parsed.notna().mean())
                if ratio >= 0.8:
                    is_datetime = True

        col_info["kind"] = (
            "numeric" if is_numeric else "datetime" if is_datetime else "text" if is_text_like else "other"
        )

        # Numeric stats
        if is_numeric:
            sn = coerce_numeric(s).dropna()
            if len(sn):
                values = sn.to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size:
                    col_info["stats"] = {
                        "count": int(values.size),
                        "mean": _safe_float(np.mean(values)),
                        "median": _safe_float(np.median(values)),
                        "std": _safe_float(np.std(values, ddof=1)) if values.size >= 2 else None,
                        "min": _safe_float(np.min(values)),
                        "max": _safe_float(np.max(values)),
                        "quantiles": {str(q): _safe_float(np.quantile(values, q)) for q in quantiles},
                    }

        # Text/categorical extras
        if (not is_numeric) and (is_text_like or s.dtype.name == "category"):
            vc = (
                s.astype("string")
                .fillna("(NA)")
                .value_counts(dropna=False)
                .head(10)
            )
            col_info["top_values"] = [
                {"value": str(idx), "count": int(cnt)} for idx, cnt in vc.items()
            ]

        out["columns"].append(col_info)

    # Convenience lists for UI
    out["columns_by_kind"] = {
        "numeric": [c["name"] for c in out["columns"] if c.get("kind") == "numeric"],
        "datetime": [c["name"] for c in out["columns"] if c.get("kind") == "datetime"],
        "text": [c["name"] for c in out["columns"] if c.get("kind") == "text"],
        "other": [c["name"] for c in out["columns"] if c.get("kind") == "other"],
    }

    return out
