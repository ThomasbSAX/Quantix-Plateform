from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import pandas as pd


EncodingStrategy = Literal["label", "frequency"]


@dataclass(frozen=True)
class EncodingResult:
    values: pd.Series
    mapping: Dict[str, int]
    unknown_value: int
    missing_value: int


def encode_unique_as_int(
    series: pd.Series,
    *,
    strategy: EncodingStrategy = "label",
    start: int = 0,
    unknown_value: int = -1,
    missing_value: int = -1,
    lowercase: bool = True,
) -> EncodingResult:
    """Encode unique text/category values as integers.

    - strategy="label": alphabetical order
    - strategy="frequency": most frequent gets smallest code

    Returns:
      - encoded series (int)
      - mapping {token -> int}

    Notes:
      - missing (NaN/None/empty after strip) -> missing_value
      - unseen values at transform time (not used here) should map to unknown_value
    """
    s = series.copy()

    def norm(v: Any) -> Optional[str]:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        t = str(v).strip()
        if not t:
            return None
        return t.lower() if lowercase else t

    normalized = s.map(norm)

    non_missing = normalized.dropna()

    if strategy == "frequency":
        order = non_missing.value_counts(dropna=False).index.tolist()
    else:
        order = sorted(set(non_missing.tolist()))

    mapping: Dict[str, int] = {token: (start + i) for i, token in enumerate(order)}

    encoded = normalized.map(lambda t: missing_value if t is None else mapping.get(t, unknown_value)).astype(int)

    return EncodingResult(
        values=encoded,
        mapping=mapping,
        unknown_value=int(unknown_value),
        missing_value=int(missing_value),
    )


def encode_dataframe_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    strategy: EncodingStrategy = "label",
    start: int = 0,
    unknown_value: int = -1,
    missing_value: int = -1,
    lowercase: bool = True,
    suffix: str = "_encoded",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """Encode multiple columns and return a new dataframe + mappings per column."""
    out = df.copy()
    mappings: Dict[str, Dict[str, int]] = {}

    for col in columns:
        if col not in out.columns:
            raise ValueError(f"Missing column: {col}")
        res = encode_unique_as_int(
            out[col],
            strategy=strategy,
            start=start,
            unknown_value=unknown_value,
            missing_value=missing_value,
            lowercase=lowercase,
        )
        out[f"{col}{suffix}"] = res.values
        mappings[col] = res.mapping

    return out, mappings
