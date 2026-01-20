from __future__ import annotations

from pathlib import Path
from typing import Optional, List
import csv

# -----------------------
# Exceptions
# -----------------------

class CSV2PDFError(RuntimeError):
    pass


# -----------------------
# Core conversion
# -----------------------

def convert(
    csv_path: str | Path,
    pdf_path: str | Path,
    *,
    delimiter: str = ",",
    encoding: str = "utf-8",
    page_size: str = "A4",          # "A4" | "LETTER"
    landscape_mode: bool = False,
    font_name: str = "Helvetica",
    font_size: int = 10,
    repeat_header: bool = True,
    margin_mm: int = 12,
) -> Path:
    """
    Convertit un CSV en PDF via reportlab.

    - pagination automatique
    - largeur de colonnes ajustée au contenu
    - header stylé et répétable
    - orientation portrait / paysage
    """

    csv_path = Path(csv_path)
    pdf_path = Path(pdf_path)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    try:
        from reportlab.lib.pagesizes import A4, LETTER, landscape as rl_landscape
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.pdfbase import pdfmetrics
        from reportlab.lib.units import mm
    except Exception as exc:
        raise CSV2PDFError(
            "reportlab requis. Installez-le avec : pip install reportlab"
        ) from exc

    # -----------------------
    # Lecture CSV (streaming)
    # -----------------------

    rows: List[List[str]] = []
    try:
        with open(csv_path, newline="", encoding=encoding) as f:
            reader = csv.reader(f, delimiter=delimiter)
            for r in reader:
                rows.append([str(c) for c in r])
    except UnicodeDecodeError:
        with open(csv_path, newline="", encoding="latin-1") as f:
            reader = csv.reader(f, delimiter=delimiter)
            for r in reader:
                rows.append([str(c) for c in r])

    if not rows:
        raise CSV2PDFError("CSV vide")

    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]

    # -----------------------
    # Page size
    # -----------------------

    ps = A4 if page_size.upper() == "A4" else LETTER
    if landscape_mode:
        ps = rl_landscape(ps)

    # -----------------------
    # Column width estimation
    # -----------------------

    col_widths: List[float] = []
    try:
        for j in range(max_cols):
            max_w = 0.0
            for r in rows:
                w = pdfmetrics.stringWidth(r[j], font_name, font_size)
                max_w = max(max_w, w)
            col_widths.append(max_w + 8)
    except Exception:
        col_widths = None  # reportlab auto-layout

    # -----------------------
    # PDF build
    # -----------------------

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=ps,
        leftMargin=margin_mm * mm,
        rightMargin=margin_mm * mm,
        topMargin=margin_mm * mm,
        bottomMargin=margin_mm * mm,
    )

    table = Table(rows, colWidths=col_widths, repeatRows=1 if repeat_header else 0)

    style = TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
    ])

    table.setStyle(style)
    doc.build([table])

    return pdf_path


__all__ = ["convert", "CSV2PDFError"]
