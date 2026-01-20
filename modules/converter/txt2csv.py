from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Optional, List, Dict

Logger = logging.getLogger(__name__)


_BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+\.)\s+(.*)$")


def txt_to_csv(
    txt_path: str | Path,
    csv_path: str | Path,
    *,
    encoding: str = "utf-8",
    delimiter: Optional[str] = None,
    quotechar: str = '"',
    on_error: str = "raise",
) -> Path:
    """
    Convertit un fichier texte / JSON / liste / CSV implicite vers CSV.

    Hiérarchie de détection :
    1. JSON (list / dict)
    2. Liste à puces ou numérotée
    3. CSV / TSV
    4. Texte brut (1 colonne)

    Garantie : le CSV produit est toujours cohérent.
    """

    txt_path = Path(txt_path)
    csv_path = Path(csv_path)

    if not txt_path.exists():
        raise FileNotFoundError(txt_path)

    text = txt_path.read_text(encoding=encoding).strip()
    lines = [ln for ln in text.splitlines() if ln.strip()]

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────
    # 1. JSON
    # ─────────────────────────────
    try:
        data = json.loads(text)

        if isinstance(data, list):
            if all(isinstance(x, dict) for x in data):
                fieldnames = sorted({k for r in data for k in r})
                with open(csv_path, "w", newline="", encoding=encoding) as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerows(data)
                return csv_path

            with open(csv_path, "w", newline="", encoding=encoding) as f:
                w = csv.writer(f)
                w.writerow(["value"])
                for x in data:
                    w.writerow([json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x])
            return csv_path

        if isinstance(data, dict):
            with open(csv_path, "w", newline="", encoding=encoding) as f:
                w = csv.DictWriter(f, fieldnames=data.keys())
                w.writeheader()
                w.writerow(data)
            return csv_path

    except Exception:
        pass  # pas du JSON

    # ─────────────────────────────
    # 2. Liste à puces / numérotée
    # ─────────────────────────────
    bullets = []
    for ln in lines:
        m = _BULLET_RE.match(ln)
        if not m:
            bullets = []
            break
        bullets.append(m.group(1).strip())

    if bullets:
        with open(csv_path, "w", newline="", encoding=encoding) as f:
            w = csv.writer(f)
            w.writerow(["item"])
            for b in bullets:
                w.writerow([b])
        return csv_path

    # ─────────────────────────────
    # 3. CSV / TSV implicite
    # ─────────────────────────────
    if delimiter is None:
        for d in ("\t", ";", ","):
            if all(d in ln for ln in lines[:3]):
                delimiter = d
                break

    if delimiter:
        rows = [ln.split(delimiter) for ln in lines]
        width = max(len(r) for r in rows)

        with open(csv_path, "w", newline="", encoding=encoding) as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow(r + [""] * (width - len(r)))
        return csv_path

    # ─────────────────────────────
    # 4. Texte brut → 1 colonne
    # ─────────────────────────────
    with open(csv_path, "w", newline="", encoding=encoding) as f:
        w = csv.writer(f)
        w.writerow(["line"])
        for ln in lines:
            w.writerow([ln])

    return csv_path


__all__ = ["txt_to_csv"]
