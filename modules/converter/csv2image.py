from __future__ import annotations

from pathlib import Path
from typing import Optional, List
import csv

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = None


class CSV2ImageError(RuntimeError):
    pass


def _require_pillow() -> None:
    if Image is None:
        raise CSV2ImageError(
            "Pillow non installé. Installez-le avec: pip install Pillow"
        )


def _load_font(font_path: Optional[str], size: int) -> ImageFont.ImageFont:
    try:
        if font_path:
            return ImageFont.truetype(font_path, size)
    except Exception:
        pass

    # fallback raisonnable (monospace si possible)
    try:
        return ImageFont.truetype("DejaVuSansMono.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    try:
        box = draw.textbbox((0, 0), text, font=font)
        return box[2] - box[0], box[3] - box[1]
    except Exception:
        return draw.textlength(text, font=font), font.size


def convert(
    csv_path: str | Path,
    image_path: str | Path,
    *,
    delimiter: str = ",",
    font_path: Optional[str] = None,
    font_size: int = 12,
    encoding: str = "utf-8",
    has_header: Optional[bool] = None,
    padding_x: int = 8,
    padding_y: int = 6,
    draw_grid: bool = True,
    bg_color: str = "white",
    fg_color: str = "black",
    grid_color: str = "#cccccc",
) -> Path:
    """
    Rend un CSV sous forme d’image (tableau) via Pillow.

    - layout déterministe
    - largeur par colonne calculée précisément
    - option header + grille
    - police monospace prioritaire (lisibilité / OCR)
    """

    _require_pillow()

    csv_path = Path(csv_path)
    image_path = Path(image_path)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    rows: List[List[str]] = []
    try:
        with open(csv_path, newline="", encoding=encoding) as f:
            reader = csv.reader(f, delimiter=delimiter)
            for r in reader:
                rows.append([str(c) for c in r])
    except UnicodeDecodeError:
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter=delimiter)
            for r in reader:
                rows.append([str(c) for c in r])

    if not rows:
        raise CSV2ImageError("CSV vide")

    # Header detection simple
    if has_header is None:
        has_header = any(not c.replace(".", "").isdigit() for c in rows[0])

    cols = max(len(r) for r in rows)
    rows = [r + [""] * (cols - len(r)) for r in rows]

    font = _load_font(font_path, font_size)
    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)

    # Largeur colonnes
    col_widths: List[int] = []
    for j in range(cols):
        max_w = 0
        for r in rows:
            w, _ = _text_size(draw, r[j], font)
            max_w = max(max_w, int(w))
        col_widths.append(max_w + 2 * padding_x)

    # Hauteur ligne
    _, text_h = _text_size(draw, "Ay", font)
    row_h = text_h + 2 * padding_y

    img_w = sum(col_widths)
    img_h = row_h * len(rows)

    img = Image.new("RGB", (img_w, img_h), bg_color)
    draw = ImageDraw.Draw(img)

    y = 0
    for i, row in enumerate(rows):
        x = 0
        for j, cell in enumerate(row):
            draw.text(
                (x + padding_x, y + padding_y),
                cell,
                fill=fg_color,
                font=font,
            )
            if draw_grid:
                draw.rectangle(
                    [x, y, x + col_widths[j], y + row_h],
                    outline=grid_color,
                )
            x += col_widths[j]
        y += row_h

    image_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(image_path)
    return image_path


__all__ = ["convert", "CSV2ImageError"]
