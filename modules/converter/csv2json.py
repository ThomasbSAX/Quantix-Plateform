from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

Logger = logging.getLogger(__name__)


class CSV2JSONError(RuntimeError):
    pass


# -----------------------
# Inference & utilities
# -----------------------

_NULLS = {"null", "none", "nan", ""}


def _infer_value(v: Optional[str]) -> Any:
    if v is None:
        return None
    s = v.strip()
    if s.lower() in _NULLS:
        return None
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
    except Exception:
        pass
    try:
        return float(s)
    except Exception:
        return s


def _detect_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample).delimiter
    except Exception:
        return ","


def _unique_fields(fields: List[str]) -> List[str]:
    seen: dict[str, int] = {}
    out: List[str] = []
    for f in fields:
        base = f.strip() or "field"
        if base in seen:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
        else:
            seen[base] = 0
            out.append(base)
    return out


# -----------------------
# Core conversion
# -----------------------

def convert(
    csv_path: str | Path,
    json_path: str | Path,
    *,
    encoding: str = "utf-8",
    delimiter: Optional[str] = None,
    has_header: Optional[bool] = None,
    fieldnames: Optional[List[str]] = None,
    infer_types: bool = True,
    ndjson: bool = False,
    skip_blank_lines: bool = True,
    on_error: str = "raise",          # "raise" | "skip"
    max_bad_rows: int = 100,
    error_log: Optional[str | Path] = None,
) -> Path:
    """
    Convertit un CSV en JSON ou NDJSON.

    - streaming réel (faible mémoire)
    - détection automatique du délimiteur et du header
    - inférence de types optionnelle
    - NDJSON strict (1 objet / ligne)
    """

    csv_path = Path(csv_path)
    json_path = Path(json_path)
    error_log = Path(error_log) if error_log else None

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    # ouverture CSV avec fallback encodage
    try:
        fin = open(csv_path, "r", encoding=encoding, newline="")
    except UnicodeDecodeError:
        fin = open(csv_path, "r", encoding="latin-1", newline="")

    errors: List[str] = []

    with fin:
        sample = fin.read(8192)
        fin.seek(0)

        delim = delimiter or _detect_delimiter(sample)
        reader = csv.reader(fin, delimiter=delim)

        try:
            first_row = next(reader)
        except StopIteration:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding=encoding) as fout:
                if ndjson:
                    pass
                else:
                    json.dump([], fout)
            return json_path

        # Header detection
        if has_header is None:
            has_header = any(
                not c.replace(".", "").replace("-", "").isdigit()
                for c in first_row
            )

        if has_header:
            fields = _unique_fields(first_row)
        else:
            fields = fieldnames or [f"field{i+1}" for i in range(len(first_row))]
            fin.seek(0)
            reader = csv.reader(fin, delimiter=delim)

        def iter_rows() -> Iterable[Dict[str, Any]]:
            bad = 0
            for line_no, row in enumerate(reader, start=1):
                try:
                    if skip_blank_lines and all(not (c or "").strip() for c in row):
                        continue

                    if len(row) < len(fields):
                        row += [""] * (len(fields) - len(row))
                    elif len(row) > len(fields):
                        row = row[: len(fields)]

                    yield {
                        k: (_infer_value(v) if infer_types else (v or None))
                        for k, v in zip(fields, row)
                    }

                except Exception as exc:
                    bad += 1
                    msg = f"Ligne {line_no}: {exc}"
                    errors.append(msg)
                    Logger.debug(msg)
                    if on_error == "raise":
                        raise CSV2JSONError(msg) from exc
                    if bad >= max_bad_rows:
                        raise CSV2JSONError("Nombre maximal de lignes invalides dépassé")

        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding=encoding) as fout:
            if ndjson:
                for obj in iter_rows():
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            else:
                json.dump(list(iter_rows()), fout, ensure_ascii=False, indent=2)

    if errors and error_log:
        try:
            error_log.write_text("\n".join(errors), encoding=encoding)
        except Exception:
            Logger.warning("Impossible d’écrire le fichier d’erreurs")

    return json_path


__all__ = ["convert", "CSV2JSONError"]
