"""Traduction de fichiers tabulaires (CSV/XLSX/TSV/Parquet) sur une colonne ciblée.

But: permettre à l'utilisateur de choisir une colonne et de la traduire uniquement,
avec un cache sur les valeurs uniques pour être rapide et éviter de sur-consommer
l'API du provider.

Conçu pour être import-safe côté Flask: dépendances provider importées en lazy.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

from .math_detection import is_mathematical_content
from .text_translation import translate_text


PathLike = Union[str, Path]


def _sniff_csv(input_path: Path, encoding: str = "utf-8") -> Tuple[str, str]:
    """Retourne (delimiter, quotechar) best-effort."""
    try:
        sample = input_path.read_text(encoding=encoding, errors="ignore")[:8192]
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter, dialect.quotechar
    except Exception:
        return ",", '"'


def _load_table(
    input_path: Path,
    *,
    encoding: str = "utf-8",
    delimiter: Optional[str] = None,
    quotechar: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    suf = input_path.suffix.lower()
    meta: Dict[str, Any] = {"path": str(input_path), "format": suf}

    if suf in {".csv", ".tsv", ".txt"}:
        if delimiter is None or quotechar is None:
            d, q = _sniff_csv(input_path, encoding=encoding)
            delimiter = delimiter or d
            quotechar = quotechar or q
        meta.update({"delimiter": delimiter, "quotechar": quotechar, "encoding": encoding})
        df = pd.read_csv(
            input_path,
            encoding=encoding,
            sep=delimiter,
            quotechar=quotechar,
            dtype=str,
            keep_default_na=True,
            na_values=["", "NULL", "null", "None", "N/A", "n/a", "#N/A", "NaN", "nan"],
        )
        return df, meta

    if suf in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path, dtype=str)
        return df, meta

    if suf == ".parquet":
        df = pd.read_parquet(input_path)
        # parquet conserve les dtypes; on force str pour traduire
        df = df.copy()
        return df, meta

    raise ValueError(f"Format tabulaire non supporté: {suf}")


def _save_table(
    df: pd.DataFrame,
    output_path: Path,
    *,
    encoding: str = "utf-8",
    delimiter: str = ",",
    quotechar: str = '"',
) -> None:
    suf = output_path.suffix.lower()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if suf in {".csv", ".tsv", ".txt"}:
        df.to_csv(output_path, index=False, encoding=encoding, sep=delimiter, quotechar=quotechar)
        return

    if suf in {".xlsx", ".xls"}:
        df.to_excel(output_path, index=False)
        return

    if suf == ".parquet":
        df.to_parquet(output_path, index=False)
        return

    raise ValueError(f"Format de sortie non supporté: {suf}")


def translate_dataframe_column(
    df: pd.DataFrame,
    *,
    column: Union[str, int],
    target_lang: str,
    source_lang: str = "auto",
    output_column: Optional[str] = None,
    replace: bool = False,
    preserve_math: bool = True,
    cache_unique: bool = True,
    preserve_empty: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Traduit une colonne d'un DataFrame.

    Args:
        df: DataFrame source
        column: nom de colonne ou index
        target_lang: ex 'fr', 'en'
        source_lang: 'auto' par défaut
        output_column: si None => "<col>_translated_<lang>" (si replace=False)
        replace: si True remplace la colonne d'origine
        preserve_math: ne traduit pas ce qui ressemble à des formules
        cache_unique: traduit les valeurs uniques une seule fois
        preserve_empty: conserve '' tel quel

    Returns:
        (df_out, report)
    """
    if df is None:
        raise ValueError("df is required")

    df_out = df.copy()

    if isinstance(column, int):
        if column < 0 or column >= len(df_out.columns):
            raise KeyError(f"Invalid column index: {column}")
        col_name = str(df_out.columns[column])
    else:
        col_name = str(column)
        if col_name not in df_out.columns:
            raise KeyError(f"Column not found: {col_name}. Available={list(df_out.columns)}")

    if replace:
        out_col = col_name
    else:
        out_col = output_column or f"{col_name}_translated_{target_lang}"
        # éviter collision
        if out_col in df_out.columns:
            i = 1
            base = out_col
            while f"{base}_{i}" in df_out.columns:
                i += 1
            out_col = f"{base}_{i}"

    # Lazy import du provider
    try:
        from deep_translator import GoogleTranslator
    except Exception as e:
        raise ImportError(
            "deep_translator est requis pour traduire. Installe-le avec `pip install deep-translator`."
        ) from e

    translator = GoogleTranslator(source=source_lang, target=target_lang)

    s = df_out[col_name]
    s_obj = s.where(s.notna(), None).astype(object)

    report: Dict[str, Any] = {
        "column": col_name,
        "output_column": out_col,
        "replace": bool(replace),
        "source_lang": source_lang,
        "target_lang": target_lang,
        "rows": int(len(df_out)),
        "translated_cells": 0,
        "skipped_empty": 0,
        "skipped_math": 0,
        "unique_values": None,
        "cache_unique": bool(cache_unique),
    }

    cache: Dict[str, str] = {}

    def _translate_cell(x: Any) -> Any:
        if x is None:
            return None
        text = str(x)
        if preserve_empty and text.strip() == "":
            report["skipped_empty"] += 1
            return text
        if preserve_math and is_mathematical_content(text):
            report["skipped_math"] += 1
            return text

        if cache_unique:
            key = text
            if key in cache:
                return cache[key]

        try:
            translated = translate_text(text, translator)
        except Exception:
            # best-effort: si le provider refuse, garder le texte original
            translated = text

        report["translated_cells"] += 1
        if cache_unique:
            cache[text] = translated
        return translated

    if cache_unique:
        # Traduire d'abord les uniques (plus rapide)
        uniques = [u for u in pd.unique(s_obj.dropna())]
        report["unique_values"] = int(len(uniques))
        for u in uniques:
            _translate_cell(u)
        df_out[out_col] = s_obj.apply(lambda v: cache.get(str(v), v) if v is not None else None)
    else:
        df_out[out_col] = s_obj.apply(_translate_cell)

    return df_out, report


def translate_dataframe_columns(
    df: pd.DataFrame,
    *,
    columns: Iterable[Union[str, int]],
    target_lang: str,
    source_lang: str = "auto",
    replace: bool = False,
    output_suffix: Optional[str] = None,
    preserve_math: bool = True,
    cache_unique: bool = True,
    preserve_empty: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Traduit plusieurs colonnes d'un DataFrame en une seule passe.

    Optimisation: on charge le fichier une seule fois, on réutilise un seul
    traducteur et un cache partagé de valeurs uniques entre colonnes.
    """
    if df is None:
        raise ValueError("df is required")

    cols_list = list(columns or [])
    if not cols_list:
        raise ValueError("columns is required")

    df_out = df.copy()

    # Lazy import du provider (une seule fois)
    try:
        from deep_translator import GoogleTranslator
    except Exception as e:
        raise ImportError(
            "deep_translator est requis pour traduire. Installe-le avec `pip install deep-translator`."
        ) from e

    translator = GoogleTranslator(source=source_lang, target=target_lang)

    # Cache partagé de valeurs uniques (string -> string) entre colonnes
    shared_cache: Dict[str, str] = {}

    reports: List[Dict[str, Any]] = []

    def _resolve_col_name(c: Union[str, int]) -> str:
        if isinstance(c, int):
            if c < 0 or c >= len(df_out.columns):
                raise KeyError(f"Invalid column index: {c}")
            return str(df_out.columns[c])
        name = str(c)
        if name not in df_out.columns:
            raise KeyError(f"Column not found: {name}. Available={list(df_out.columns)}")
        return name

    def _out_col_name(col_name: str) -> str:
        if replace:
            return col_name
        suffix = output_suffix or f"translated_{target_lang}"
        out_col = f"{col_name}_{suffix}"
        if out_col in df_out.columns:
            i = 1
            base = out_col
            while f"{base}_{i}" in df_out.columns:
                i += 1
            out_col = f"{base}_{i}"
        return out_col

    def _translate_cell_text(text: str) -> str:
        if cache_unique and text in shared_cache:
            return shared_cache[text]
        try:
            translated = translate_text(text, translator)
        except Exception:
            translated = text
        if cache_unique:
            shared_cache[text] = translated
        return translated

    for c in cols_list:
        col_name = _resolve_col_name(c)
        out_col = _out_col_name(col_name)

        s = df_out[col_name]
        s_obj = s.where(s.notna(), None).astype(object)

        rep: Dict[str, Any] = {
            "column": col_name,
            "output_column": out_col,
            "replace": bool(replace),
            "source_lang": source_lang,
            "target_lang": target_lang,
            "rows": int(len(df_out)),
            "translated_cells": 0,
            "skipped_empty": 0,
            "skipped_math": 0,
            "unique_values": None,
            "cache_unique": bool(cache_unique),
        }

        def _translate_cell(x: Any) -> Any:
            if x is None:
                return None
            txt = str(x)
            if preserve_empty and txt.strip() == "":
                rep["skipped_empty"] += 1
                return txt
            if preserve_math and is_mathematical_content(txt):
                rep["skipped_math"] += 1
                return txt
            rep["translated_cells"] += 1
            return _translate_cell_text(txt)

        if cache_unique:
            uniques = [u for u in pd.unique(s_obj.dropna())]
            rep["unique_values"] = int(len(uniques))
            for u in uniques:
                _translate_cell(u)
            df_out[out_col] = s_obj.apply(
                lambda v: shared_cache.get(str(v), v) if v is not None else None
            )
        else:
            df_out[out_col] = s_obj.apply(_translate_cell)

        reports.append(rep)

    summary: Dict[str, Any] = {
        "columns": [r["column"] for r in reports],
        "output_columns": [r["output_column"] for r in reports],
        "source_lang": source_lang,
        "target_lang": target_lang,
        "replace": bool(replace),
        "cache_unique": bool(cache_unique),
        "shared_cache_size": int(len(shared_cache)),
        "reports": reports,
    }

    return df_out, summary


def translate_table_column(
    input_path: PathLike,
    output_path: PathLike,
    *,
    column: Union[str, int],
    target_lang: str,
    source_lang: str = "auto",
    output_column: Optional[str] = None,
    replace: bool = False,
    preserve_math: bool = True,
    cache_unique: bool = True,
    preserve_empty: bool = True,
    encoding: str = "utf-8",
    delimiter: Optional[str] = None,
    quotechar: Optional[str] = None,
) -> Dict[str, Any]:
    """Traduit une colonne d'un fichier tabulaire et écrit le fichier de sortie.

    Retourne un report JSON-friendly.
    """
    in_path = Path(input_path)
    out_path = Path(output_path)

    df, meta = _load_table(in_path, encoding=encoding, delimiter=delimiter, quotechar=quotechar)

    df_out, rep = translate_dataframe_column(
        df,
        column=column,
        target_lang=target_lang,
        source_lang=source_lang,
        output_column=output_column,
        replace=replace,
        preserve_math=preserve_math,
        cache_unique=cache_unique,
        preserve_empty=preserve_empty,
    )

    # réutilise la config csv si applicable
    d = meta.get("delimiter") or delimiter or ","
    q = meta.get("quotechar") or quotechar or '"'

    _save_table(df_out, out_path, encoding=encoding, delimiter=d, quotechar=q)

    return {
        "input": meta,
        "output": {"path": str(out_path), "format": out_path.suffix.lower()},
        "translation": rep,
    }


def translate_table_columns(
    input_path: PathLike,
    output_path: PathLike,
    *,
    columns: Iterable[Union[str, int]],
    target_lang: str,
    source_lang: str = "auto",
    replace: bool = False,
    output_suffix: Optional[str] = None,
    preserve_math: bool = True,
    cache_unique: bool = True,
    preserve_empty: bool = True,
    encoding: str = "utf-8",
    delimiter: Optional[str] = None,
    quotechar: Optional[str] = None,
) -> Dict[str, Any]:
    """Traduit plusieurs colonnes d'un fichier tabulaire et écrit un seul fichier de sortie."""
    in_path = Path(input_path)
    out_path = Path(output_path)

    df, meta = _load_table(in_path, encoding=encoding, delimiter=delimiter, quotechar=quotechar)

    df_out, rep = translate_dataframe_columns(
        df,
        columns=columns,
        target_lang=target_lang,
        source_lang=source_lang,
        replace=replace,
        output_suffix=output_suffix,
        preserve_math=preserve_math,
        cache_unique=cache_unique,
        preserve_empty=preserve_empty,
    )

    d = meta.get("delimiter") or delimiter or ","
    q = meta.get("quotechar") or quotechar or '"'
    _save_table(df_out, out_path, encoding=encoding, delimiter=d, quotechar=q)

    return {
        "input": meta,
        "output": {"path": str(out_path), "format": out_path.suffix.lower()},
        "translation": rep,
    }
