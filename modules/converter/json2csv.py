from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json
import csv
import logging
import traceback

try:
    import pandas as pd  # optionnel
except Exception:
    pd = None

Logger = logging.getLogger(__name__)


# =========================
# Utils
# =========================

class JSON2CSVError(RuntimeError):
    pass


def _flatten_dict(
    obj: Dict[str, Any],
    *,
    sep: str = ".",
) -> Dict[str, Any]:
    """
    Aplatissement déterministe sans pandas.
    Les listes sont sérialisées en JSON.
    """
    out: Dict[str, Any] = {}

    def _rec(x: Any, prefix: str) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                _rec(v, f"{prefix}{sep}{k}" if prefix else k)
        elif isinstance(x, list):
            out[prefix] = json.dumps(x, ensure_ascii=False)
        else:
            out[prefix] = x

    _rec(obj, "")
    return out


def _read_json(
    path: Path,
    *,
    encoding: str,
    is_jsonl: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if is_jsonl:
        with open(path, "r", encoding=encoding) as f:
            for i, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception as exc:
                    raise JSON2CSVError(f"Ligne {i}: JSON invalide") from exc

                if isinstance(obj, dict):
                    rows.append(obj)
                else:
                    rows.append({"value": obj})
        return rows

    with open(path, "r", encoding=encoding) as f:
        data = json.load(f)

    if isinstance(data, list):
        for item in data:
            rows.append(item if isinstance(item, dict) else {"value": item})
    elif isinstance(data, dict):
        if all(isinstance(v, dict) for v in data.values()):
            rows.extend(data.values())
        else:
            rows.extend({"key": k, "value": v} for k, v in data.items())
    else:
        rows.append({"value": data})

    return rows


# =========================
# API principale
# =========================

def convert(
    json_input: str | Path,
    csv_output: str | Path,
    *,
    encoding: str = "utf-8",
    delimiter: str = ",",
    quotechar: str = '"',
    flatten: bool = False,
    flat_sep: str = ".",
    input_is_jsonl: Optional[bool] = None,
    fieldnames: Optional[List[str]] = None,
    on_error: str = "raise",  # "raise" | "warn" | "ignore"
    error_log: Optional[str | Path] = None,
) -> bool:
    """
    Convertit JSON ou JSONL vers CSV.

    - support JSON array / dict / JSON Lines
    - flatten optionnel (pandas si dispo)
    - structure stable et déterministe
    """

    json_path = Path(json_input)
    csv_path = Path(csv_output)

    if not json_path.exists():
        msg = f"Fichier JSON introuvable: {json_path}"
        if on_error == "raise":
            raise FileNotFoundError(json_path)
        Logger.warning(msg)
        return False

    # Détection JSONL
    is_jsonl = (
        input_is_jsonl
        if input_is_jsonl is not None
        else json_path.suffix.lower() in (".jsonl", ".ndjson")
    )

    try:
        rows = _read_json(json_path, encoding=encoding, is_jsonl=is_jsonl)
    except Exception as exc:
        msg = f"Erreur lecture JSON: {exc}"
        Logger.exception(msg)
        if error_log:
            Path(error_log).write_text(msg + "\n" + traceback.format_exc())
        if on_error == "raise":
            raise
        return False

    if not rows:
        msg = "Aucune donnée JSON exploitable"
        if on_error == "raise":
            raise JSON2CSVError(msg)
        Logger.warning(msg)
        return False

    # Flatten
    if flatten:
        try:
            if pd is not None:
                df = pd.json_normalize(rows, sep=flat_sep)
                rows = df.to_dict(orient="records")
            else:
                rows = [_flatten_dict(r, sep=flat_sep) for r in rows]
        except Exception as exc:
            msg = f"Erreur flatten JSON: {exc}"
            Logger.exception(msg)
            if error_log:
                Path(error_log).write_text(msg + "\n" + traceback.format_exc())
            if on_error == "raise":
                raise
            return False

    # Champs CSV
    if fieldnames is None:
        fieldnames = sorted({k for r in rows for k in r.keys()})

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(csv_path, "w", newline="", encoding=encoding) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                delimiter=delimiter,
                quotechar=quotechar,
                extrasaction="ignore",
            )
            writer.writeheader()

            for i, row in enumerate(rows, start=1):
                try:
                    safe = {
                        k: (
                            json.dumps(v, ensure_ascii=False)
                            if isinstance(v, (dict, list))
                            else v
                        )
                        for k, v in row.items()
                    }
                    writer.writerow(safe)
                except Exception as exc:
                    msg = f"Ligne {i}: erreur écriture CSV: {exc}"
                    Logger.debug(msg)
                    if on_error == "raise":
                        raise

    except Exception as exc:
        msg = f"Erreur écriture CSV: {exc}"
        Logger.exception(msg)
        if error_log:
            Path(error_log).write_text(msg + "\n" + traceback.format_exc())
        if on_error == "raise":
            raise
        return False

    return True


# =========================
# Aliases
# =========================

def json_to_csv(*a, **k):
    return convert(*a, **k)


def jsonl_to_csv(jsonl_path: str | Path, csv_path: str | Path, **kwargs):
    return convert(jsonl_path, csv_path, input_is_jsonl=True, **kwargs)


def json_to_csv_flat(json_path: str | Path, csv_path: str | Path, **kwargs):
    return convert(json_path, csv_path, flatten=True, **kwargs)


__all__ = [
    "convert",
    "json_to_csv",
    "jsonl_to_csv",
    "json_to_csv_flat",
    "JSON2CSVError",
]
