from __future__ import annotations

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth


def txt_to_pdf(
    txt_path: str | Path,
    pdf_path: str | Path,
    *,
    encoding: str = "utf-8",
    font_name: str = "Helvetica",
    font_size: int = 11,
    margin_mm: int = 20,
    line_spacing: float = 1.25,
) -> Path:
    """
    Convertit un fichier TXT en PDF lisible avec word-wrapping et pagination.

    - Gère les textes longs
    - Respecte les paragraphes
    - Pas de débordement horizontal
    """

    txt_path = Path(txt_path)
    pdf_path = Path(pdf_path)

    if not txt_path.exists():
        raise FileNotFoundError(txt_path)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4

    margin = margin_mm * 72 / 25.4  # mm → points
    max_width = width - 2 * margin
    line_height = font_size * line_spacing

    c.setFont(font_name, font_size)
    text = c.beginText(margin, height - margin)

    def wrap_line(line: str) -> list[str]:
        words = line.split()
        if not words:
            return [""]

        lines = []
        current = words[0]

        for w in words[1:]:
            test = current + " " + w
            if stringWidth(test, font_name, font_size) <= max_width:
                current = test
            else:
                lines.append(current)
                current = w

        lines.append(current)
        return lines

    with open(txt_path, "r", encoding=encoding) as f:
        for raw_line in f:
            line = raw_line.rstrip()

            # paragraphe vide
            if not line:
                text.textLine("")
                continue

            wrapped = wrap_line(line)
            for wline in wrapped:
                if text.getY() <= margin:
                    c.drawText(text)
                    c.showPage()
                    c.setFont(font_name, font_size)
                    text = c.beginText(margin, height - margin)

                text.textLine(wline)

    c.drawText(text)
    c.save()

    return pdf_path


__all__ = ["txt_to_pdf"]
