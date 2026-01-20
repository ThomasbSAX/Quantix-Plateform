from __future__ import annotations

from pathlib import Path
from typing import Optional, List
import csv
import re
import logging

Logger = logging.getLogger(__name__)

# =========================
# Image -> Markdown (Granite / Docling)
# =========================

def image_to_markdown(
    image_path: str | Path,
    output_md: Optional[str | Path] = None,
    *,
    do_ocr: bool = True,
) -> Path:
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import ImageFormatOption
    except Exception as exc:
        raise RuntimeError("Docling / Granite non installé") from exc

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    output_md = Path(output_md or image_path.with_suffix(".md"))
    output_md.parent.mkdir(parents=True, exist_ok=True)

    # Selon la version Docling, il n'existe pas forcément d'options dédiées aux images.
    # Les images passent par le pipeline PDF standard, donc PdfPipelineOptions est compatible.
    opts = PdfPipelineOptions(
        do_ocr=do_ocr,
        do_table_structure=True,
        do_formula_enrichment=False,
        do_code_enrichment=False,
        generate_page_images=False,
        generate_picture_images=False,
        generate_table_images=False,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts)
        }
    )

    result = converter.convert(str(image_path))
    md = result.document.export_to_markdown()

    if "|" not in md:
        raise RuntimeError("Granite n’a produit aucune structure tabulaire exploitable")

    output_md.write_text(md, encoding="utf-8")
    return output_md


# =========================
# Markdown -> CSV (tables strictes)
# =========================

_ROW = re.compile(r"^\s*\|(.+?)\|\s*$")
_SEP = re.compile(r"^\s*\|(?:\s*:?-+:?\s*\|)+\s*$")


def _parse_md_tables(md: str) -> List[List[List[str]]]:
    tables: List[List[List[str]]] = []
    current: List[List[str]] = []

    for line in md.splitlines():
        if _SEP.match(line):
            continue

        m = _ROW.match(line)
        if m:
            cells = [c.strip() for c in m.group(1).split("|")]
            current.append(cells)
        else:
            if current:
                tables.append(current)
                current = []

    if current:
        tables.append(current)

    return tables


def markdown_to_csv(
    md_path: str | Path,
    csv_path: str | Path,
    *,
    table_index: int = 0,
    delimiter: str = ",",
) -> Path:
    md_path = Path(md_path)
    csv_path = Path(csv_path)

    if not md_path.exists():
        raise FileNotFoundError(md_path)

    md = md_path.read_text(encoding="utf-8")
    tables = _parse_md_tables(md)

    if not tables:
        raise RuntimeError("Aucune table Markdown valide détectée")

    if table_index >= len(tables):
        raise IndexError(
            f"Table {table_index} inexistante (tables détectées: {len(tables)})"
        )

    table = tables[table_index]
    width = max(len(r) for r in table)

    # normalisation stricte
    norm = [
        r + [""] * (width - len(r)) if len(r) < width else r[:width]
        for r in table
    ]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerows(norm)

    return csv_path


# =========================
# Pipeline complet
# =========================

def image_to_csv(
    image_path: str | Path,
    csv_path: str | Path,
    *,
    table_index: int = 0,
    delimiter: str = ",",
    keep_markdown: bool = False,
) -> Path:
    md_path = Path(csv_path).with_suffix(".md")

    image_to_markdown(image_path, md_path)
    markdown_to_csv(md_path, csv_path, table_index=table_index, delimiter=delimiter)

    if not keep_markdown:
        md_path.unlink(missing_ok=True)

    return Path(csv_path)


__all__ = ["image_to_markdown", "markdown_to_csv", "image_to_csv"]


def convert(
    src_path: str | Path,
    out_path: str | Path,
    **kwargs,
) -> Path:
    """Entrée standard pour le routeur (src_path, out_path).

    kwargs supportés:
    - do_ocr: bool (par défaut True)
    """
    do_ocr = kwargs.get("do_ocr", True)
    return image_to_markdown(src_path, out_path, do_ocr=bool(do_ocr))

