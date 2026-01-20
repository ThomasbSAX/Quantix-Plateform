from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import csv
import re
import logging

Logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Étape 1 — PDF → Markdown (Granite)
# -------------------------------------------------------------------

def _pdf_to_markdown_granite(pdf_path: Path) -> str:
    try:
        from .graniteIBM import granite
    except Exception as exc:
        raise RuntimeError("Granite / Docling non disponible") from exc

    md_path = granite(
        input_path=str(pdf_path),
        output_format="markdown"
    )

    md_path = Path(md_path)
    if not md_path.exists():
        raise RuntimeError("Granite n'a pas produit de Markdown")

    return md_path.read_text(encoding="utf-8")


# -------------------------------------------------------------------
# Étape 2 — Détection de tableaux dans le Markdown
# -------------------------------------------------------------------


def _extract_tables(md: str) -> List[List[List[str]]]:
    """Détecte des tableaux dans un Markdown Docling/Granite (best-effort).

    On s'appuie sur l'extracteur générique (pipe tables, listes, colonnes alignées,
    CSV-like). Cela améliore la détection sur des PDFs hétérogènes.
    """

    try:
        from .markdown2csv import extract_tables_from_text

        return extract_tables_from_text(md)
    except Exception:
        # Fallback ultra-simple si l'import échoue.
        # (On garde un comportement minimal plutôt que de casser la conversion.)
        tables: List[List[List[str]]] = []
        lines = (md or "").splitlines()
        current: List[List[str]] = []
        for line in lines:
            if line.lstrip().startswith("|"):
                row = [c.strip() for c in line.strip().strip("|").split("|")]
                if row:
                    current.append(row)
            else:
                if len(current) >= 2:
                    tables.append(current)
                current = []
        if len(current) >= 2:
            tables.append(current)
        return tables


def _pdf_to_markdown_docling_tables(pdf_path: Path) -> str:
    """PDF -> Markdown via Docling avec options orientées tableaux.

    Objectif: améliorer la détection de tableaux sur PDF où Granite/Markdown ne
    produit pas de tables exploitables (ex: table scannée / structure faible).
    """

    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption
    except Exception as exc:
        raise RuntimeError("Docling non disponible pour le fallback PDF→Markdown") from exc

    opts = PdfPipelineOptions()
    # OCR + structure de tableau (best-effort selon version Docling)
    if hasattr(opts, "do_ocr"):
        opts.do_ocr = True
    if hasattr(opts, "do_table_structure"):
        opts.do_table_structure = True
    if hasattr(opts, "generate_page_images"):
        opts.generate_page_images = False
    if hasattr(opts, "generate_picture_images"):
        opts.generate_picture_images = False
    if hasattr(opts, "do_formula_enrichment"):
        opts.do_formula_enrichment = False

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    result = converter.convert(str(pdf_path))
    return result.document.export_to_markdown() or ""


def _pdf_to_csv_via_ocr_first_page(
    pdf_path: Path,
    output_csv: Path,
    *,
    table_index: int,
    delimiter: str,
) -> Path:
    """Dernier recours: PDF (page 1) -> image -> CSV (OCR table)."""

    try:
        from .pdf2image import pdf_to_images
    except Exception as exc:
        raise RuntimeError("Fallback OCR indisponible (pdf2image manquant)") from exc

    try:
        from .image2csv import image_to_csv
    except Exception as exc:
        raise RuntimeError("Fallback OCR indisponible (image2csv manquant)") from exc

    tmp_dir = output_csv.parent / f"{output_csv.stem}_ocr_pages"
    imgs = pdf_to_images(
        pdf_path,
        tmp_dir,
        dpi=300,
        image_format="png",
        first_page=1,
        last_page=1,
        on_error="raise",
    )
    if not imgs:
        raise RuntimeError("Fallback OCR: aucune image générée depuis le PDF")

    # OCR sur la 1ère page
    return image_to_csv(
        imgs[0],
        output_csv,
        table_index=table_index,
        delimiter=delimiter,
        lang="en",
    )


# -------------------------------------------------------------------
# Étape 3 — Écriture CSV
# -------------------------------------------------------------------

def _write_csv(
    table: List[List[str]],
    path: Path,
    delimiter: str = ","
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerows(table)


# -------------------------------------------------------------------
# Pipeline complet
# -------------------------------------------------------------------

def pdf_to_csv_tables(
    pdf_path: str | Path,
    output_dir: Optional[str | Path] = None,
    *,
    delimiter: str = ",",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Extrait tous les tableaux détectables d'un PDF vers des CSV.

    Pipeline:
        PDF → Markdown (Granite)
            → détection robuste des tableaux Markdown
            → CSV (1 fichier par tableau)

    Returns:
        dict avec chemins CSV et métadonnées
    """

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    output_dir = Path(output_dir or "result/csv") / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        Logger.info("PDF → CSV tables : %s", pdf_path)

    # 1. Granite → Markdown
    md = _pdf_to_markdown_granite(pdf_path)

    # 2. Détection des tableaux
    tables = _extract_tables(md)

    csv_files: List[Path] = []

    for idx, table in enumerate(tables, start=1):
        csv_path = output_dir / f"table_{idx:03d}.csv"
        _write_csv(table, csv_path, delimiter=delimiter)
        csv_files.append(csv_path)

    # manifest
    manifest = {
        "source_pdf": str(pdf_path),
        "tables_detected": len(tables),
        "csv_files": [str(p) for p in csv_files],
        "output_dir": str(output_dir),
    }

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if verbose:
        Logger.info("✓ %d tableaux extraits", len(csv_files))

    return {
        "success": True,
        "tables": len(csv_files),
        "csv_files": [str(p) for p in csv_files],
        "output_dir": str(output_dir),
    }


def pdf_to_csv(
    pdf_path: str | Path,
    output_csv: str | Path,
    *,
    table_index: int = 0,
    delimiter: str = ",",
    combine_tables: bool = False,
    verbose: bool = False,
) -> Path:
    """Extrait une table (ou concatène toutes les tables) d'un PDF vers un seul CSV.

    Par défaut, on exporte la table `table_index` (0) car c'est ce qu'attend
    l'usage le plus courant dans l'UI.
    """

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if table_index < 0:
        raise ValueError("table_index doit être >= 0")

    md = _pdf_to_markdown_granite(pdf_path)
    tables = _extract_tables(md)

    # Fallback 1: Docling avec table-structure
    if not tables:
        try:
            md2 = _pdf_to_markdown_docling_tables(pdf_path)
            tables = _extract_tables(md2)
        except Exception:
            pass

    # Fallback 2: OCR via page image (utile pour PDF scannés)
    if not tables:
        try:
            return _pdf_to_csv_via_ocr_first_page(
                pdf_path,
                output_csv,
                table_index=table_index,
                delimiter=delimiter,
            )
        except Exception as exc:
            raise RuntimeError(
                "Aucun tableau détecté dans le PDF. "
                "Ce PDF semble être scanné (image) ou sans structure de tableau exploitable. "
                f"Détails fallback OCR: {exc}"
            )

    if combine_tables and len(tables) > 1:
        combined: List[List[str]] = []
        for idx, t in enumerate(tables):
            if idx:
                combined.append([])
            combined.extend(t)
        _write_csv(combined, output_csv, delimiter=delimiter)
        return output_csv

    if table_index >= len(tables):
        raise IndexError(
            f"Table {table_index} inexistante (tables détectées: {len(tables)})"
        )

    _write_csv(tables[table_index], output_csv, delimiter=delimiter)
    return output_csv


__all__ = ["pdf_to_csv_tables"]


def convert(pdf_path: str | Path, out_path: str | Path, **kwargs):
    """Wrapper compatibilité: PDF -> CSV (multi-tables).

    Par défaut, on produit un CSV unique (table 0). Pour conserver l'ancien
    comportement multi-fichiers (1 CSV par table), passer `multi_tables=true`.

    kwargs supportés:
    - delimiter: str
    - table_index: int
    - combine_tables: bool
    - multi_tables: bool
    """

    out_path = Path(out_path)

    delimiter = str(kwargs.get("delimiter", ","))
    table_index = int(kwargs.get("table_index", 0))
    combine_tables = bool(kwargs.get("combine_tables", False))
    multi_tables = bool(kwargs.get("multi_tables", False))
    verbose = bool(kwargs.get("verbose", False))

    if multi_tables and not combine_tables:
        # On interprète out_path comme un dossier de sortie (pour les tables)
        output_dir = out_path.parent / f"{out_path.stem}_tables"
        return pdf_to_csv_tables(pdf_path, output_dir=output_dir, delimiter=delimiter, verbose=verbose)

    # CSV unique (table_index) ou concat (combine_tables)
    return pdf_to_csv(
        pdf_path,
        out_path,
        table_index=table_index,
        delimiter=delimiter,
        combine_tables=combine_tables,
        verbose=verbose,
    )


__all__.append("convert")
