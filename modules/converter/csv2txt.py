from __future__ import annotations

from pathlib import Path
from typing import Optional, Iterable
import csv


class CSV2TextError(RuntimeError):
    pass


def convert(
    csv_path: str | Path,
    txt_path: str | Path,
    *,
    delimiter: str = ",",
    encoding: str = "utf-8",
    output_delimiter: Optional[str] = None,
    skip_blank_lines: bool = True,
    strip_cells: bool = False,
    line_ending: str = "\n",
) -> Path:
    """
    Convertit un CSV en fichier texte simple (1 ligne texte par ligne CSV).

    - streaming (faible mémoire)
    - support UTF-8 avec BOM
    - contrôle du délimiteur de sortie
    - normalisation légère optionnelle
    """

    csv_path = Path(csv_path)
    txt_path = Path(txt_path)
    out_delim = output_delimiter if output_delimiter is not None else delimiter

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    txt_path.parent.mkdir(parents=True, exist_ok=True)

    # ouverture avec fallback BOM
    try:
        fin = open(csv_path, newline="", encoding=encoding)
    except UnicodeDecodeError:
        fin = open(csv_path, newline="", encoding="utf-8-sig")

    with fin, open(txt_path, "w", encoding=encoding, newline="") as fout:
        reader = csv.reader(fin, delimiter=delimiter)

        for row in reader:
            if skip_blank_lines and all(not (c or "").strip() for c in row):
                continue

            if strip_cells:
                row = [c.strip() if isinstance(c, str) else str(c) for c in row]
            else:
                row = [str(c) for c in row]

            fout.write(out_delim.join(row) + line_ending)

    return txt_path


__all__ = ["convert", "CSV2TextError"]
