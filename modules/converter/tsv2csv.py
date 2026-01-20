from pathlib import Path
import csv
import io
import logging
import traceback
from typing import Optional

Logger = logging.getLogger(__name__)


def convert(
    tsv_path: str | Path,
    csv_path: str | Path,
    *,
    encoding: str = "utf-8",
    input_delimiter: str = "\t",
    output_delimiter: str = ",",
    quotechar: str = '"',
    has_header: Optional[bool] = None,
    fieldnames: Optional[list] = None,
    skip_blank_lines: bool = True,
    on_error: str = "raise",
    error_log: Optional[str] = None,
) -> bool:
    """Convertit un TSV en CSV en gérant les erreurs courantes.

    - `on_error`: 'raise'|'warn'|'ignore'.
    - Retourne True si succès, False sinon (si `on_error` != 'raise').
    """

    tsv_path = Path(tsv_path)
    csv_path = Path(csv_path)

    if not tsv_path.exists():
        msg = f"Fichier source introuvable: {tsv_path}"
        if on_error == "raise":
            raise FileNotFoundError(tsv_path)
        Logger.warning(msg)
        if error_log:
            Path(error_log).write_text(msg)
        return False

    # open with encoding fallback
    try:
        with open(tsv_path, "r", encoding=encoding, newline="") as fin:
            sample = fin.read(8192)
            fin.seek(0)
            rest = fin.read()
    except UnicodeDecodeError:
        Logger.warning("Échec encodage %s, tentative latin-1", encoding)
        try:
            with open(tsv_path, "r", encoding="latin-1", newline="") as fin:
                sample = fin.read(8192)
                fin.seek(0)
                rest = fin.read()
        except Exception as exc:
            msg = f"Impossible d'ouvrir le fichier: {exc}"
            if on_error == "raise":
                raise
            Logger.exception(msg)
            if error_log:
                Path(error_log).write_text(msg + "\n" + traceback.format_exc())
            return False
    except Exception as exc:
        msg = f"Impossible d'ouvrir le fichier: {exc}"
        if on_error == "raise":
            raise
        Logger.exception(msg)
        if error_log:
            Path(error_log).write_text(msg + "\n" + traceback.format_exc())
        return False

    buf = io.StringIO(sample + rest)

    reader = csv.reader(buf, delimiter=input_delimiter)

    # peek
    try:
        first = next(reader)
    except StopIteration:
        # empty file -> create empty CSV
        with open(csv_path, "w", encoding=encoding, newline="") as fout:
            pass
        return True
    except csv.Error as exc:
        msg = f"Erreur lecture TSV: {exc}"
        if on_error == "raise":
            raise
        Logger.exception(msg)
        if error_log:
            Path(error_log).write_text(msg)
        return False

    # determine header
    if has_header is None:
        non_numeric = any(not (cell.replace("-", "").replace(".", "").isdigit()) for cell in first)
        has_header = non_numeric

    if has_header:
        raw_fieldnames = [h.strip() for h in first]
        if fieldnames is None:
            # ensure unique
            seen = {}
            fieldnames_final = []
            for name in raw_fieldnames:
                base = name or ""
                if base in seen:
                    seen[base] += 1
                    name2 = f"{base}_{seen[base]}"
                else:
                    seen[base] = 0
                    name2 = base
                fieldnames_final.append(name2)
        else:
            fieldnames_final = fieldnames
    else:
        if fieldnames:
            fieldnames_final = fieldnames
        else:
            fieldnames_final = [f"col{i+1}" for i in range(len(first))]
        # rewind
        buf = io.StringIO(sample + rest)
        reader = csv.reader(buf, delimiter=input_delimiter)

    errors: list = []

    try:
        with open(csv_path, "w", newline="", encoding=encoding) as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames_final, delimiter=output_delimiter, quotechar=quotechar, extrasaction="ignore")
            writer.writeheader()
            # if we had header originally, skip first row
            if has_header:
                # consume header
                pass
            for nr, row in enumerate(reader, start=1):
                try:
                    if skip_blank_lines and all((cell or "").strip() == "" for cell in row):
                        continue
                    # normalize length
                    if len(row) < len(fieldnames_final):
                        row = row + [""] * (len(fieldnames_final) - len(row))
                    elif len(row) > len(fieldnames_final):
                        row = row[: len(fieldnames_final)]
                    obj = {k: v for k, v in zip(fieldnames_final, row)}
                    writer.writerow(obj)
                except Exception as exc:
                    msg = f"Erreur ligne {nr}: {exc}"
                    errors.append(msg)
                    Logger.debug(msg)
                    if on_error == "raise":
                        raise
    except Exception as exc:
        msg = f"Erreur écriture CSV: {exc}\n{traceback.format_exc()}"
        if on_error == "raise":
            raise
        Logger.exception(msg)
        if error_log:
            Path(error_log).write_text(msg)
        return False

    if errors and error_log:
        try:
            Path(error_log).write_text("\n".join(errors))
        except Exception:
            Logger.warning("Impossible d'écrire error_log: %s", error_log)

    return True


def tsv_to_csv(*a, **k):
    return convert(*a, **k)


__all__ = ["convert", "tsv_to_csv"]
