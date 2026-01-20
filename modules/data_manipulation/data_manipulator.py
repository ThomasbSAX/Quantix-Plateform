from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


try:
    # On réutilise les briques déjà présentes (unités + patterns) si dispo
    from modules.cleaner.Cleaner import Cleaner
except Exception:  # pragma: no cover
    Cleaner = None

try:
    from modules.cleaner.convertUnit import extract_quantity as extract_unit_quantity
except Exception:  # pragma: no cover
    extract_unit_quantity = None


PathLike = Union[str, Path]


@dataclass(frozen=True)
class MergeSpec:
    how: Literal["left", "right", "inner", "outer"] = "left"
    on: Optional[Union[str, List[str]]] = None
    left_on: Optional[Union[str, List[str]]] = None
    right_on: Optional[Union[str, List[str]]] = None
    suffixes: Tuple[str, str] = ("_x", "_y")
    validate: Optional[str] = None  # ex: '1:1', '1:m', 'm:1', 'm:m'
    indicator: bool = False


@dataclass(frozen=True)
class FeatureSpec:
    """Décrit une feature dérivée à créer à partir d'une colonne."""

    name: str
    kind: Literal[
        "regex_flag",
        "regex_extract",
        "unit_extract",
        "unit_convert",
        "is_missing",
        "to_numeric",
        "to_datetime",
        "normalize_whitespace",
    ]
    source: str
    # paramètres optionnels
    pattern: Optional[str] = None
    group: Union[int, str, None] = 0
    flags: int = re.IGNORECASE
    target_unit: Optional[str] = None
    numeric_errors: Literal["coerce", "raise", "ignore"] = "coerce"
    datetime_errors: Literal["coerce", "raise", "ignore"] = "coerce"
    dayfirst: bool = True


class DataManipulator:
    """Manipulation "semi-manuelle" mais assistée des datasets.

    Objectifs:
    - charger/sauvegarder facilement
    - fusionner (merge) de façon sûre
    - créer des colonnes dérivées (email/tel/uuid/ip, unités, etc.)
    - garder un petit rapport JSON des opérations
    """

    def __init__(self) -> None:
        self._report: Dict[str, Any] = {
            "operations": [],
        }

        # Patterns par défaut: on réutilise ceux du Cleaner si possible
        if Cleaner is not None:
            try:
                self.default_patterns = dict(Cleaner.PATTERNS)
            except Exception:
                self.default_patterns = {}
        else:
            self.default_patterns = {}

        # Un petit set "pratique" (mêmes noms que Cleaner quand possible)
        # NB: volontairement pas de phone_generic (trop de faux positifs)
        self.feature_patterns = {
            "email": self.default_patterns.get("email"),
            "url": self.default_patterns.get("url"),
            "ip_address": self.default_patterns.get("ip_address"),
            "uuid": self.default_patterns.get("uuid"),
            "iban": self.default_patterns.get("iban"),
            "credit_card": self.default_patterns.get("credit_card"),
            "phone_fr": r"(?:\+33\s?|0)[1-9](?:[\s\.-]?\d{2}){4}",
            "phone_us": self.default_patterns.get("phone_us"),
            # unités: heuristiques (la vraie extraction passe par convertUnit)
            "unit_distance": self.default_patterns.get("unit_distance"),
            "unit_speed": self.default_patterns.get("unit_speed"),
            "unit_weight": self.default_patterns.get("unit_weight"),
            "unit_pressure": self.default_patterns.get("unit_pressure"),
        }
        self.feature_patterns = {k: v for k, v in self.feature_patterns.items() if v}

    # -----------------
    # Report
    # -----------------

    def get_report(self) -> Dict[str, Any]:
        return dict(self._report)

    def write_report(self, path: PathLike) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self._report, ensure_ascii=False, indent=2), encoding="utf-8")

    def _op(self, name: str, details: Optional[Dict[str, Any]] = None) -> None:
        entry = {"name": name}
        if details:
            entry.update(details)
        self._report.setdefault("operations", []).append(entry)

    # -----------------
    # IO
    # -----------------

    def load(self, path: PathLike, **kwargs) -> pd.DataFrame:
        """Charge un fichier en s'appuyant sur le loader du Cleaner si disponible.

        Ça évite de dupliquer la logique: sniff CSV, encodages, excel, etc.
        """
        p = Path(path)
        if Cleaner is None:
            # fallback minimal
            if p.suffix.lower() in {".xlsx", ".xls"}:
                df = pd.read_excel(p, **kwargs)
            else:
                df = pd.read_csv(p, **kwargs)
            self._op("load", {"path": str(p), "rows": int(len(df)), "cols": int(len(df.columns))})
            return df

        cleaner = Cleaner()
        df = cleaner.load_file(p, **kwargs)
        self._op("load", {"path": str(p), "rows": int(len(df)), "cols": int(len(df.columns))})
        return df

    def save(self, df: pd.DataFrame, path: PathLike, **kwargs) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        suf = p.suffix.lower()
        if suf in {".xlsx", ".xls"}:
            df.to_excel(p, index=False, **kwargs)
        elif suf == ".parquet":
            df.to_parquet(p, index=False, **kwargs)
        else:
            df.to_csv(p, index=False, **kwargs)
        self._op("save", {"path": str(p), "rows": int(len(df)), "cols": int(len(df.columns))})
        return p

    # -----------------
    # Simple transforms
    # -----------------

    def rename_columns(self, df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        before = list(df.columns)
        df = df.rename(columns=mapping)
        after = list(df.columns)
        self._op("rename_columns", {"mapping": dict(mapping), "before": before, "after": after})
        return df

    def drop_columns(self, df: pd.DataFrame, columns: Sequence[str], errors: Literal["raise", "ignore"] = "ignore") -> pd.DataFrame:
        cols = list(columns)
        df = df.drop(columns=cols, errors=errors)
        self._op("drop_columns", {"columns": cols, "errors": errors})
        return df

    def replace_in_column(
        self,
        df: pd.DataFrame,
        *,
        column: str,
        pattern: str,
        repl: str,
        regex: bool = True,
        flags: int = re.IGNORECASE,
    ) -> pd.DataFrame:
        if column not in df.columns:
            raise KeyError(f"Column not found: {column}")
        s = df[column]
        before_non_null = int(s.notna().sum())
        s_str = s.where(s.notna(), None).astype(str)
        if regex:
            df[column] = s_str.str.replace(pattern, repl, regex=True, flags=flags)
        else:
            df[column] = s_str.str.replace(pattern, repl, regex=False)
        self._op(
            "replace_in_column",
            {
                "column": column,
                "pattern": pattern,
                "repl": repl,
                "regex": bool(regex),
                "non_null": before_non_null,
            },
        )
        return df

    def clean(self, df: pd.DataFrame, **cleaner_kwargs) -> pd.DataFrame:
        """Passe un Cleaner sur un DataFrame (utile dans une pipeline config-driven)."""
        if Cleaner is None:
            raise RuntimeError("Cleaner unavailable")
        auto_detect_types = bool(cleaner_kwargs.pop("auto_detect_types", True))
        validate_structure = bool(cleaner_kwargs.pop("validate_structure", True))
        fix_encoding = bool(cleaner_kwargs.pop("fix_encoding", True))
        cleaner = Cleaner(**cleaner_kwargs)
        out = cleaner.clean(
            df=df,
            auto_detect_types=auto_detect_types,
            validate_structure=validate_structure,
            fix_encoding=fix_encoding,
        )
        # On garde une trace compacte (les détails complets existent déjà côté Cleaner)
        try:
            stats = cleaner.get_stats(detailed=False)
            self._op(
                "clean",
                {
                    "rows_before": int(stats.get("rows_before", 0) or 0),
                    "rows_after": int(stats.get("rows_after", 0) or 0),
                    "cols_before": int(stats.get("cols_before", 0) or 0),
                    "cols_after": int(stats.get("cols_after", 0) or 0),
                    "operations": list(stats.get("operations", []) or []),
                },
            )
        except Exception:
            self._op("clean", {"note": "completed"})
        return out

    # -----------------
    # Merge
    # -----------------

    def merge(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        spec: MergeSpec,
    ) -> pd.DataFrame:
        """Fusionne deux DataFrames avec un spec explicite et report."""
        before = (int(len(left)), int(len(left.columns)))
        out = left.merge(
            right,
            how=spec.how,
            on=spec.on,
            left_on=spec.left_on,
            right_on=spec.right_on,
            suffixes=spec.suffixes,
            validate=spec.validate,
            indicator=spec.indicator,
        )
        after = (int(len(out)), int(len(out.columns)))
        self._op(
            "merge",
            {
                "how": spec.how,
                "on": spec.on,
                "left_on": spec.left_on,
                "right_on": spec.right_on,
                "validate": spec.validate,
                "indicator": spec.indicator,
                "shape_before": {"rows": before[0], "cols": before[1]},
                "shape_after": {"rows": after[0], "cols": after[1]},
            },
        )
        return out

    # -----------------
    # Features helpers
    # -----------------

    def add_regex_flags(
        self,
        df: pd.DataFrame,
        *,
        source: str,
        patterns: Dict[str, str],
        prefix: str = "has_",
        to_int: bool = True,
    ) -> pd.DataFrame:
        """Ajoute des colonnes bool/int pour présence de motifs regex dans une colonne."""
        if source not in df.columns:
            raise KeyError(f"Column not found: {source}")

        s = df[source].astype(str)
        created = []
        for name, rx in patterns.items():
            col = f"{prefix}{name}"
            mask = s.str.contains(rx, regex=True, flags=re.IGNORECASE, na=False)
            df[col] = mask.astype(int) if to_int else mask
            created.append(col)

        self._op("add_regex_flags", {"source": source, "created": created})
        return df

    def add_unit_columns(
        self,
        df: pd.DataFrame,
        *,
        source: str,
        value_col: Optional[str] = None,
        unit_col: Optional[str] = None,
        converted_col: Optional[str] = None,
        target_unit: Optional[str] = None,
        exclude_currency: bool = True,
    ) -> pd.DataFrame:
        """Extrait valeur + unité d'une colonne texte et optionnellement convertit.

        - nécessite `modules.cleaner.convertUnit` pour une extraction robuste.
        - `exclude_currency=True` ignore les cellules contenant des symboles monétaires.
        """
        if source not in df.columns:
            raise KeyError(f"Column not found: {source}")
        if extract_unit_quantity is None:
            raise RuntimeError("convertUnit.extract_quantity unavailable")

        value_col = value_col or f"{source}__value"
        unit_col = unit_col or f"{source}__unit"
        converted_col = converted_col or (f"{source}__converted_{target_unit.replace('/', '_')}" if target_unit else None)

        currency_symbols = ("€", "$", "£", "¥", "₹", "₽")

        def _safe_extract(x):
            if x is None:
                return None
            s = str(x).strip()
            if not s:
                return None
            if exclude_currency and any(sym in s for sym in currency_symbols):
                return None
            try:
                return extract_unit_quantity(s)
            except Exception:
                return None

        parsed = df[source].where(df[source].notna(), None).apply(_safe_extract)
        df[value_col] = parsed.apply(lambda t: float(t[0]) if isinstance(t, tuple) else np.nan)
        df[unit_col] = parsed.apply(lambda t: t[1] if isinstance(t, tuple) else None)

        converted_count = 0
        if target_unit and converted_col:
            try:
                from modules.cleaner.convertUnit import convert as convert_unit

                def _conv(row):
                    v = row[0]
                    u = row[1]
                    if v is None or (isinstance(v, float) and np.isnan(v)) or not u:
                        return np.nan
                    try:
                        return float(convert_unit(float(v), str(u), str(target_unit)))
                    except Exception:
                        return np.nan

                df[converted_col] = pd.concat([df[value_col], df[unit_col]], axis=1).apply(_conv, axis=1)
                converted_count = int(df[converted_col].notna().sum())
            except Exception:
                converted_col = None

        self._op(
            "add_unit_columns",
            {
                "source": source,
                "value_col": value_col,
                "unit_col": unit_col,
                "converted_col": converted_col,
                "target_unit": target_unit,
                "converted_count": converted_count,
            },
        )
        return df

    def apply_features(self, df: pd.DataFrame, features: Sequence[FeatureSpec]) -> pd.DataFrame:
        """Applique une liste de FeatureSpec de façon déterministe."""
        for f in features:
            if f.source not in df.columns:
                raise KeyError(f"Column not found: {f.source}")

            if f.kind == "normalize_whitespace":
                df[f.name] = (
                    df[f.source]
                    .astype(str)
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip()
                )
                self._op("feature.normalize_whitespace", {"source": f.source, "name": f.name})

            elif f.kind == "is_missing":
                df[f.name] = df[f.source].isna()
                self._op("feature.is_missing", {"source": f.source, "name": f.name})

            elif f.kind == "regex_flag":
                if not f.pattern:
                    raise ValueError(f"FeatureSpec.pattern required for {f.kind}")
                s = df[f.source].astype(str)
                df[f.name] = s.str.contains(f.pattern, regex=True, flags=f.flags, na=False).astype(int)
                self._op("feature.regex_flag", {"source": f.source, "name": f.name, "pattern": f.pattern})

            elif f.kind == "regex_extract":
                if not f.pattern:
                    raise ValueError(f"FeatureSpec.pattern required for {f.kind}")
                s = df[f.source].astype(str)
                extracted = s.str.extract(f.pattern, flags=f.flags)
                if isinstance(f.group, int):
                    # pandas retourne DataFrame; le groupe 0 est la première capture
                    if extracted.shape[1] == 1:
                        df[f.name] = extracted.iloc[:, 0]
                    else:
                        df[f.name] = extracted.iloc[:, f.group]
                else:
                    df[f.name] = extracted[f.group]
                self._op("feature.regex_extract", {"source": f.source, "name": f.name, "pattern": f.pattern, "group": f.group})

            elif f.kind == "to_numeric":
                df[f.name] = pd.to_numeric(df[f.source], errors=f.numeric_errors)
                self._op("feature.to_numeric", {"source": f.source, "name": f.name, "errors": f.numeric_errors})

            elif f.kind == "to_datetime":
                df[f.name] = pd.to_datetime(df[f.source], errors=f.datetime_errors, dayfirst=f.dayfirst)
                self._op(
                    "feature.to_datetime",
                    {"source": f.source, "name": f.name, "errors": f.datetime_errors, "dayfirst": bool(f.dayfirst)},
                )

            elif f.kind == "unit_extract":
                self.add_unit_columns(df, source=f.source, value_col=f"{f.name}__value", unit_col=f"{f.name}__unit")

            elif f.kind == "unit_convert":
                if not f.target_unit:
                    raise ValueError("FeatureSpec.target_unit required for unit_convert")
                self.add_unit_columns(
                    df,
                    source=f.source,
                    value_col=f"{f.name}__value",
                    unit_col=f"{f.name}__unit",
                    converted_col=f.name,
                    target_unit=f.target_unit,
                )

            else:
                raise ValueError(f"Unknown feature kind: {f.kind}")

        return df

    # -----------------
    # Convenience presets
    # -----------------

    def add_contact_and_id_features(
        self,
        df: pd.DataFrame,
        *,
        source_columns: Optional[Iterable[str]] = None,
        prefix: str = "has_",
    ) -> pd.DataFrame:
        """Ajoute rapidement des flags email/tel/ip/uuid/iban/cc sur plusieurs colonnes."""
        cols = list(source_columns) if source_columns is not None else list(df.columns)
        patterns = {k: v for k, v in self.feature_patterns.items() if k in {"email", "phone_fr", "phone_us", "ip_address", "uuid", "iban", "credit_card", "url"}}

        created_total: List[str] = []
        for col in cols:
            if col not in df.columns or df[col].dtype != "object":
                continue
            local_prefix = f"{prefix}{col}__"
            self.add_regex_flags(df, source=col, patterns=patterns, prefix=local_prefix, to_int=True)
            created_total.extend([f"{local_prefix}{k}" for k in patterns.keys()])

        self._op("add_contact_and_id_features", {"columns": cols, "created": created_total})
        return df
