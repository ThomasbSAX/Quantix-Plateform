from __future__ import annotations

from pathlib import Path
import csv
import logging
import traceback
from typing import Optional, Iterable

Logger = logging.getLogger(__name__)


class CSVConversionError(RuntimeError):
    pass


def _is_numeric(s: str) -> bool:
    try:
        float(s.replace(",", "."))
        return True
    except Exception:
        return False


def _detect_header(row: Iterable[str]) -> bool:
    # Heuristique classique : majorité de non-numérique
    cells = list(row)
    if not cells:
        return False
    non_numeric = sum(not _is_numeric(c.strip()) for c in cells if c.strip())
    return non_numeric >= len(cells) / 2


def _make_unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        base = name.strip() or "field"
        if base in seen:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
        else:
            seen[base] = 0
            out.append(base)
    return out


def _handle_error(
    msg: str,
    *,
    exc: Optional[Exception],
    on_error: str,
    error_log: Optional[Path],
) -> bool:
    Logger.error(msg)
    if exc:
        Logger.debug(traceback.format_exc())
    if error_log:
        try:
            error_log.write_text(
                msg + ("\n\n" + traceback.format_exc() if exc else ""),
                encoding="utf-8",
            )
        except Exception:
            Logger.warning("Impossible d’écrire error_log: %s", error_log)
    if on_error == "raise":
        raise CSVConversionError(msg) from exc
    return False


def convert(
    csv_path: str | Path,
    tsv_path: str | Path,
    *,
    encoding: str = "utf-8",
    input_delimiter: str = ",",
    output_delimiter: str = "\t",
    quotechar: str = '"',
    has_header: Optional[bool] = None,
    fieldnames: Optional[list[str]] = None,
    skip_blank_lines: bool = True,
    on_error: str = "raise",  # "raise" | "skip"
    error_log: Optional[str | Path] = None,
) -> bool:
    """
    Conversion CSV -> TSV robuste.
    - streaming (pas de chargement complet en mémoire)
    - détection header
    - normalisation des champs
    - gestion d’erreurs configurable
    """

    csv_path = Path(csv_path)
    tsv_path = Path(tsv_path)
    error_log = Path(error_log) if error_log else None

    if not csv_path.exists():
        return _handle_error(
            f"Fichier source introuvable: {csv_path}",
            exc=None,
            on_error=on_error,
            error_log=error_log,
        )

    # ouverture avec fallback d'encodage
    try:
        fin = open(csv_path, "r", encoding=encoding, newline="")
    except UnicodeDecodeError:
        Logger.warning("Encodage %s invalide, tentative latin-1", encoding)
        try:
            fin = open(csv_path, "r", encoding="latin-1", newline="")
        except Exception as exc:
            return _handle_error(
                f"Impossible d’ouvrir le fichier {csv_path}",
                exc=exc,
                on_error=on_error,
                error_log=error_log,
            )
    except Exception as exc:
        return _handle_error(
            f"Impossible d’ouvrir le fichier {csv_path}",
            exc=exc,
            on_error=on_error,
            error_log=error_log,
        )

    with fin, open(tsv_path, "w", encoding=encoding, newline="") as fout:
        reader = csv.reader(fin, delimiter=input_delimiter)
        writer: Optional[csv.DictWriter] = None

        try:
            first_row = next(reader)
        except StopIteration:
            return True
        except csv.Error as exc:
            return _handle_error(
                "Erreur lecture CSV",
                exc=exc,
                on_error=on_error,
                error_log=error_log,
            )

        if has_header is None:
            has_header = _detect_header(first_row)

        if has_header:
            header = fieldnames or _make_unique(first_row)
        else:
            header = fieldnames or [f"col{i+1}" for i in range(len(first_row))]
            # on réinjecte la première ligne comme données
            reader = iter([first_row, *reader])

        writer = csv.DictWriter(
            fout,
            fieldnames=header,
            delimiter=output_delimiter,
            quotechar=quotechar,
            extrasaction="ignore",
        )
        writer.writeheader()

        for line_no, row in enumerate(reader, start=2 if has_header else 1):
            try:
                if skip_blank_lines and all(not (c or "").strip() for c in row):
                    continue

                if len(row) < len(header):
                    row += [""] * (len(header) - len(row))
                elif len(row) > len(header):
                    row = row[: len(header)]

                writer.writerow(dict(zip(header, row)))
            except Exception as exc:
                msg = f"Erreur ligne {line_no}: {exc}"
                if on_error == "raise":
                    raise CSVConversionError(msg) from exc
                Logger.debug(msg)

    return True


def csv_to_tsv(*args, **kwargs) -> bool:
    return convert(*args, **kwargs)


__all__ = ["convert", "csv_to_tsv", "CSVConversionError"]
