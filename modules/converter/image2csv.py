"""
Image -> CSV (tableur photographié) via PaddleOCR PP-Structure.

Pipeline robuste :
image
 └── PaddleOCR (table detection + OCR)
       ├── extraction cellules
       ├── reconstruction lignes / colonnes
       └── CSV

Dépendances :
- paddlepaddle
- paddleocr
- opencv-python
- pandas (optionnel, fallback HTML)

Installation :
pip install paddlepaddle paddleocr opencv-python pandas
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union
import csv
import logging

Logger = logging.getLogger(__name__)


# =========================
# Exceptions
# =========================

class Image2CSVError(RuntimeError):
    pass


# =========================
# PaddleOCR – Table detection
# =========================

def image_to_tables_paddle(
    image_path: Union[str, Path],
    *,
    lang: str = "en",
) -> List[dict]:
    """
    Détecte et extrait les tables d'une image via PaddleOCR PP-Structure.

    Retour :
    - liste de tables (dicts) contenant :
      - res["cells"] : lignes -> cellules OCR
      - res["html"]  : table HTML
      - bbox         : bounding box dans l'image
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    try:
        import cv2
        from paddleocr import PPStructure
    except Exception as exc:
        raise Image2CSVError(
            "PaddleOCR requis. Installez : pip install paddlepaddle paddleocr opencv-python"
        ) from exc

    engine = PPStructure(
        show_log=False,
        lang=lang,
    )

    img = cv2.imread(str(image_path))
    if img is None:
        raise Image2CSVError("Impossible de charger l'image (cv2.imread a échoué)")

    results = engine(img)

    tables = [r for r in results if r.get("type") == "table"]
    if not tables:
        raise Image2CSVError("Aucun tableau détecté dans l'image")

    return tables


def _fallback_docling_markdown_tables(image_path: Path) -> List[List[List[str]]]:
    """Fallback: Image -> Markdown (Docling) -> détection de tables.

    Objectif: améliorer les cas où PaddleOCR n'est pas installé ou échoue.
    """

    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import ImageFormatOption
    except Exception as exc:
        raise Image2CSVError(
            "Docling requis pour le fallback Image→Markdown. Installez : pip install docling"
        ) from exc

    try:
        from .markdown2csv import extract_tables_from_text
    except Exception as exc:
        raise Image2CSVError("Impossible d'importer l'extracteur Markdown→tables") from exc

    # Docling ne fournit pas toujours d'options spécifiques "ImagePipelineOptions" selon la version.
    # L'API stable (et utilisée aussi pour les images via StandardPdfPipeline) est PdfPipelineOptions.
    opts = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        do_formula_enrichment=False,
        do_code_enrichment=False,
        generate_page_images=False,
        generate_picture_images=False,
        generate_table_images=False,
    )

    converter = DocumentConverter(
        format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts)}
    )
    result = converter.convert(str(image_path))
    md = result.document.export_to_markdown() or ""

    tables = extract_tables_from_text(md)
    if not tables:
        raise Image2CSVError("Aucun tableau détecté dans l'image (fallback Docling)")
    return tables


# =========================
# Conversion Table -> CSV
# =========================

def table_cells_to_csv(
    table: dict,
    csv_path: Union[str, Path],
    *,
    delimiter: str = ",",
) -> Path:
    """
    Conversion directe via les cellules PaddleOCR (méthode préférée).
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    cells = table.get("res", {}).get("cells")
    if not cells:
        raise Image2CSVError("Structure de table invalide (cells absentes)")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=delimiter)
        for row in cells:
            writer.writerow([cell.get("text", "") for cell in row])

    return csv_path


def table_html_to_csv(
    table: dict,
    csv_path: Union[str, Path],
) -> Path:
    """
    Fallback via HTML -> pandas.
    """
    html = table.get("res", {}).get("html")
    if not html:
        raise Image2CSVError("HTML de table absent (fallback impossible)")

    try:
        import pandas as pd
        from io import StringIO
    except Exception as exc:
        raise Image2CSVError(
            "pandas requis pour le fallback HTML. Installez : pip install pandas"
        ) from exc

    dfs = pd.read_html(StringIO(html))
    if not dfs:
        raise Image2CSVError("Impossible de parser la table HTML")

    df = dfs[0]
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    return csv_path


# =========================
# Pipeline complet
# =========================

def image_to_csv(
    image_path: Union[str, Path],
    csv_path: Union[str, Path],
    *,
    table_index: int = 0,
    delimiter: str = ",",
    lang: str = "en",
) -> Path:
    """
    Pipeline complet :
    image -> PaddleOCR -> table -> CSV

    - table_index : index du tableau à extraire si plusieurs détectés
    """
    image_path_p = Path(image_path)

    # Méthode 1 (préférée): PaddleOCR
    tables: List[dict]
    try:
        tables = image_to_tables_paddle(image_path_p, lang=lang)
    except Exception as exc:
        Logger.debug("PaddleOCR indisponible/échec (%s). Fallback Docling.", exc)
        tables = []

    if tables:
        if table_index >= len(tables):
            raise IndexError(
                f"Table {table_index} inexistante (tables détectées: {len(tables)})"
            )

        table = tables[table_index]

        try:
            return table_cells_to_csv(table, csv_path, delimiter=delimiter)
        except Exception as exc:
            Logger.debug("Conversion via cells échouée, fallback HTML: %s", exc)
            return table_html_to_csv(table, csv_path)

    # Méthode 2: Docling -> Markdown -> parse tables
    md_tables = _fallback_docling_markdown_tables(image_path_p)
    if table_index >= len(md_tables):
        raise IndexError(
            f"Table {table_index} inexistante (tables détectées: {len(md_tables)})"
        )

    chosen = md_tables[table_index]

    # Heuristique OCR fréquente: "S" confondu avec "5" ou "$" (ex: 5002 au lieu de S002).
    # On corrige uniquement ces cas très spécifiques.
    normalized: List[List[str]] = []
    for row in chosen:
        if not row:
            continue
        first = (row[0] or "").strip()
        if first.startswith("$") and len(first) == 4 and first[1:].isdigit():
            first = "S" + first[1:]
        elif first.startswith("5") and len(first) == 4 and first[1:].isdigit():
            first = "S" + first[1:]
        normalized.append([first] + row[1:])
    chosen = normalized or chosen
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerows(chosen)
    return csv_path


def convert(src_path: str | Path, out_path: str | Path, **kwargs):
    """Wrapper compatibilité pour le routeur: image -> csv."""

    table_index = int(kwargs.get("table_index", 0))
    delimiter = str(kwargs.get("delimiter", ","))
    lang = str(kwargs.get("lang", "en"))
    return image_to_csv(src_path, out_path, table_index=table_index, delimiter=delimiter, lang=lang)


# =========================
# CLI
# =========================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Convertir une image de tableau (photo de tableur) en CSV via PaddleOCR"
    )
    ap.add_argument("image", help="chemin vers l'image (jpg/png)")
    ap.add_argument("output", help="chemin du CSV de sortie")
    ap.add_argument("--table-index", type=int, default=0, help="index du tableau à extraire")
    ap.add_argument("--delimiter", default=",", help="délimiteur CSV")
    ap.add_argument("--lang", default="en", help="langue OCR (en, fr, etc.)")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    out = image_to_csv(
        args.image,
        args.output,
        table_index=args.table_index,
        delimiter=args.delimiter,
        lang=args.lang,
    )
    print(out)
